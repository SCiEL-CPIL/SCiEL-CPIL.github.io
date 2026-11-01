import numpy as np
import time

N = 400

def _max_autocorr_sidelobe(h):
    """Compute max autocorrelation at non-zero lags for a ±1 sequence."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft)[:N]
    return np.max(np.abs(acf[1:]))

def _de_bruijn_sequence(n_bits):
    """Generate a de Bruijn sequence of order n_bits over alphabet {0, 1}."""
    k = 2
    a = [0] * (k * (k + 1))
    sequence = []
    
    def db(t, p):
        if t > n_bits:
            if n_bits % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
                
    db(1, 1)
    return sequence

def _construct_de_bruijn_candidate(order):
    """
    Construct a candidate signed step function based on a de Bruijn sequence.
    Maps the de Bruijn sequence (which has period 2^order) to a length-N sequence.
    We map 0 -> 1, 1 -> -1.
    """
    seq = _de_bruijn_sequence(order)
    # seq length is 2^order
    # We need N elements.
    # If N > len(seq), we repeat. If N < len(seq), we truncate.
    # Since N=400 is not a power of 2, this creates a phase mismatch if we just repeat,
    # but the "local" structure is preserved.
    
    # A better approach for "balanced de Bruijn graph traversal" might be to use
    # the de Bruijn sequence to define the *transitions* or simply use the raw bits.
    # Let's just map bits to signs.
    
    # To ensure we use the full periodicity as much as possible, we might want to
    # align the start.
    
    h = np.zeros(N)
    for i in range(N):
        bit = seq[i % len(seq)]
        h[i] = 1.0 if bit == 0 else -1.0
        
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Strategy 1: De Bruijn based initialization
    # Try different orders. 2^8 = 256, 2^9 = 512.
    # N=400 is between 256 and 512.
    # Order 9 gives a sequence of length 512. We take the first 400.
    # Order 8 gives length 256. We repeat it ~1.5 times.
    
    candidates = []
    
    # Order 9 (length 512)
    try:
        h_db9 = _construct_de_bruijn_candidate(9)
        candidates.append(h_db9)
    except Exception:
        pass
        
    # Order 8 (length 256)
    try:
        h_db8 = _construct_de_bruijn_candidate(8)
        candidates.append(h_db8)
    except Exception:
        pass

    # Also try a random initialization as a fallback/comparison
    rng = np.random.RandomState(42)
    h_rand = rng.choice([1, -1], N)
    candidates.append(h_rand)
    
    # Evaluate candidates
    for h in candidates:
        score = _max_autocorr_sidelobe(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        return h_rand

    # Strategy 2: Local Search / Simulated Annealing Refinement
    # We use the best candidate as the starting point for iterative improvement.
    
    def local_search_with_sa(h, max_iter=50000, time_limit=60.0):
        h = h.copy()
        score = _max_autocorr_sidelobe(h)
        best_h_local = h.copy()
        best_score_local = score
        
        T = max(0.1, score * 0.1)
        T_min = 0.01
        cooling = 0.999
        
        start_ls = time.time()
        
        for _ in range(max_iter):
            if time.time() - start_ls > time_limit:
                break
            if time.time() > deadline - 10.0:
                break
                
            idx = rng.randint(N)
            h_new = h.copy()
            h_new[idx] = -h_new[idx]
            
            new_score = _max_autocorr_sidelobe(h_new)
            delta = new_score - score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h = h_new
                score = new_score
                
                if score < best_score_local:
                    best_score_local = score
                    best_h_local = h.copy()
                    
            T = max(T * cooling, T_min)
            
        return best_h_local, best_score_local

    # Run refinement
    if time.time() < deadline - 20.0:
        h_ref, score_ref = local_search_with_sa(best_h, max_iter=100000, time_limit=60.0)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref

    # If we have time, try a second round of refinement from a slightly perturbed state
    # to see if we can escape a local minimum.
    if time.time() < deadline - 10.0 and best_score > 1.0: # If score is high (bad), try harder
        # Perturb best_h
        h_pert = best_h.copy()
        # Flip 5% of bits
        num_flips = int(0.05 * N)
        flip_indices = rng.choice(N, num_flips, replace=False)
        h_pert[flip_indices] *= -1
        
        h_ref2, score_ref2 = local_search_with_sa(h_pert, max_iter=50000, time_limit=30.0)
        if score_ref2 < best_score:
            best_score = score_ref2
            best_h = h_ref2

    return best_h
