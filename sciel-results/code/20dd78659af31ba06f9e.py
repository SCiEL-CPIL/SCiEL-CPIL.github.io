"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_full(h):
    """Compute full autocorrelation r[k] = sum_i h[i]*h[i+k] for k=0..N-1 using FFT."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return r[:N]


def _score_from_autocorr(r):
    """Compute peak sidelobe from autocorrelation array."""
    return np.max(np.abs(r[1:N]))


def _fast_flip_autocorr_update(h, idx, r):
    """
    Update autocorrelation after flipping h[idx].
    h[idx] changes from h[idx] to -h[idx].
    Delta at position idx is -2*h[idx].
    
    For each lag k, r[k] = sum_i h[i]*h[i+k].
    Flipping h[idx] affects terms where idx or idx+k is involved.
    
    More precisely:
    r_new[k] = r_old[k] - 2*h[idx]*h[(idx-k) mod N] - 2*h[(idx+k) mod N]*h[idx] + ... 
    Actually let's think carefully:
    
    r[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]  (for linear autocorrelation)
    
    When we flip h[idx], the change in r[k] is:
    - Terms where i = idx: h[idx]*h[idx+k] -> -h[idx]*h[idx+k], change = -2*h[idx]*h[idx+k]
    - Terms where i+k = idx, i.e., i = idx-k: h[idx-k]*h[idx] -> h[idx-k]*(-h[idx]), change = -2*h[idx-k]*h[idx]
    
    But we need to be careful about boundaries.
    
    Actually, for the FFT-based autocorrelation (which computes circular autocorrelation of the zero-padded signal),
    it's easier to just recompute. But for speed, let's use the linear autocorrelation formula.
    
    Let's use: r[k] = sum_{i=max(0,k-N+1)}^{min(N-1-k, N-1)} h[i]*h[i+k]
    
    When flipping h[idx]:
    - For lag k: 
      * If idx >= k and idx < N-k: term h[idx]*h[idx+k] is present, changes by -2*h[idx]*h[idx+k]
      * If idx-k >= 0 and idx-k < N-k: term h[idx-k]*h[idx] is present, changes by -2*h[idx-k]*h[idx]
      * Note: if both conditions hold (i.e., k <= idx < N-k), both terms are present but they are different terms
        (one is h[idx]*h[idx+k], other is h[idx-k]*h[idx])
      * Special case k=0: h[idx]*h[idx] = h[idx]^2 = 1, flipping doesn't change it.
    
    So delta_r[k] = -2*h[idx]*h[(idx+k) % N] - 2*h[(idx-k) % N]*h[idx] for k > 0, with boundary checks.
    For k=0, delta = 0.
    """
    delta = np.zeros(N, dtype=np.float64)
    old_val = h[idx]
    
    for k in range(1, N):
        dr = 0.0
        # Term where i = idx: need idx < N - k
        if idx < N - k:
            dr += -2.0 * old_val * h[idx + k]
        # Term where i = idx - k: need idx - k >= 0
        if idx - k >= 0:
            dr += -2.0 * h[idx - k] * old_val
        delta[k] = dr
    
    r_new = r + delta
    return r_new


def _generate_m_sequence(poly, length, init_state):
    """Generate m-sequence bits from LFSR."""
    degree = poly.bit_length()
    register = init_state & ((1 << degree) - 1)
    if register == 0:
        register = 1
    bits = []
    for _ in range(length):
        bit = register & 1
        bits.append(bit)
        feedback = register & 1
        register >>= 1
        if feedback:
            register ^= poly
    return bits


def _get_primitives():
    """Known primitive polynomials for LFSRs up to degree 16."""
    return [
        0b10011,      # x^4 + x + 1
        0b100011,     # x^5 + x^2 + 1
        0b1000101,    # x^6 + x + 1
        0b10000101,   # x^7 + x + 1
        0b100000111,  # x^8 + x^4 + x^3 + x^2 + 1
        0b1000000011, # x^9 + x^4 + x^3 + x^2 + 1
        0b10000000101,# x^10 + x^3 + x^2 + x + 1
        0b100000000011, # x^11 + x^2 + 1
        0b1000000000011, # x^12 + x^6 + x^4 + x + 1
        0b100000000000011, # x^13 + x^4 + x^3 + x + 1
        0b1000000000000011, # x^14 + x^5 + x^3 + x^2 + 1
        0b100000000000000011, # x^15 + x + 1
        0b1000000000000000011, # x^16 + x^5 + x^3 + x^2 + 1
    ]


def _simulated_annealing(h, r, start_temp, end_temp, n_iter, deadline, rng):
    """
    Run simulated annealing on binary sequence h with autocorrelation r.
    Returns (best_h, best_score, final_h, final_r).
    """
    current_h = h.copy()
    current_r = r.copy()
    current_score = _score_from_autocorr(current_r)
    
    best_h = current_h.copy()
    best_r = current_r.copy()
    best_score = current_score
    
    T = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max(1, n_iter))
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Pick a random position to flip
        idx = rng.randint(N)
        old_val = current_h[idx]
        
        # Compute delta in autocorrelation
        delta_r = np.zeros(N, dtype=np.float64)
        for k in range(1, N):
            dr = 0.0
            if idx < N - k:
                dr += -2.0 * old_val * current_h[idx + k]
            if idx - k >= 0:
                dr += -2.0 * current_h[idx - k] * old_val
            delta_r[k] = dr
        
        new_r = current_r + delta_r
        new_score = _score_from_autocorr(new_r)
        
        delta = new_score - current_score
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_h[idx] = -old_val
            current_r = new_r
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
                best_r = current_r.copy()
        
        T *= cooling
    
    return best_h, best_score, current_h, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_score = np.inf
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1a: M-sequences from LFSRs
    primitives = _get_primitives()
    for poly in primitives:
        degree = poly.bit_length()
        period = (1 << degree) - 1
        # Use first N bits of m-sequence
        for init_state in [1, 3, 7, 15, 31]:
            if init_state >= period:
                continue
            bits = _generate_m_sequence(poly, N, init_state)
            h = np.array([2 * b - 1 for b in bits], dtype=np.float64)
            r = _autocorr_full(h)
            candidates.append((h, r))
    
    # 1b: Random sequences
    for _ in range(20):
        h = rng.choice([-1, 1], N).astype(np.float64)
        r = _autocorr_full(h)
        candidates.append((h, r))
    
    # 1c: Walsh-Hadamard based (first 400 rows of a large Hadamard matrix)
    try:
        # Use a simple construction: take first N rows of Hadamard of order 512
        from scipy.linalg import hadamard
        H = hadamard(512)
        for row_start in range(0, 512 - N + 1, 50):
            h = H[row_start:row_start + N, :N].astype(np.float64)
            # Normalize to ±1
            h = np.sign(h)
            h[h == 0] = 1
            r = _autocorr_full(h)
            candidates.append((h, r))
    except Exception:
        pass
    
    # Score all candidates
    scored_candidates = []
    for h, r in candidates:
        s = _score_from_autocorr(r)
        scored_candidates.append((s, h, r))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Simulated annealing from top candidates
    top_k = min(15, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 20.0:
            break
        
        s_init, h_init, r_init = scored_candidates[i]
        
        if s_init < best_score:
            best_score = s_init
            best_h = h_init.copy()
        
        # Run SA with decreasing temperature
        # Estimate typical score scale
        T_start = max(5.0, s_init * 0.5)
        T_end = 0.1
        n_iter = 5000
        
        h_best, s_best, h_final, r_final = _simulated_annealing(
            h_init, r_init, T_start, T_end, n_iter, deadline - 10.0, rng
        )
        
        if s_best < best_score:
            best_score = s_best
            best_h = h_best.copy()
        
        # If we're doing well, try more iterations from the best
        if s_best < best_score * 0.95 and time.time() < deadline - 10.0:
            T_start2 = max(2.0, s_best * 0.3)
            T_end2 = 0.05
            n_iter2 = 3000
            h_best2, s_best2, _, _ = _simulated_annealing(
                h_best, _autocorr_full(h_best), T_start2, T_end2, n_iter2, deadline - 5.0, rng
            )
            if s_best2 < best_score:
                best_score = s_best2
                best_h = h_best2.copy()
    
    # Phase 3: Final greedy polish from best
    if best_h is not None and time.time() < deadline - 2.0:
        current_h = best_h.copy()
        current_r = _autocorr_full(current_h)
        current_score = _score_from_autocorr(current_r)
        
        # Simple greedy: try flipping each position, keep if improves
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(N):
                if time.time() > deadline - 2.0:
                    break
                old_val = current_h[idx]
                
                delta_r = np.zeros(N, dtype=np.float64)
                for k in range(1, N):
                    dr = 0.0
                    if idx < N - k:
                        dr += -2.0 * old_val * current_h[idx + k]
                    if idx - k >= 0:
                        dr += -2.0 * current_h[idx - k] * old_val
                    delta_r[k] = dr
                
                new_r = current_r + delta_r
                new_score = _score_from_autocorr(new_r)
                
                if new_score < current_score:
                    current_h[idx] = -old_val
                    current_r = new_r
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
