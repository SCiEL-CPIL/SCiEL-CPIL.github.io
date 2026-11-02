"""Rewritten program with structural change: greedy coordinate descent with
analytic autocorrelation updates and deterministic low-autocorrelation initialization."""
import numpy as np
import time

N = 400


def _autocorr(f):
    """Compute linear autocorrelation R[k] = sum_i f[i]*f[i+k] for k=0..N-1."""
    # Use FFT for full autocorrelation
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    return R[:N]


def _peak_sidelobe(f):
    """Max absolute value of non-zero-lag autocorrelation."""
    R = _autocorr(f)
    return np.max(np.abs(R[1:]))


def _c3_bound(f):
    """Compute the C3 autoconvolution upper bound.
    
    The score is the upper bound on C3. Based on the task, this is related
    to the L1 norm of the autocorrelation normalized appropriately.
    For a ±1 sequence, we use: (1/N^2) * sum_k |R(k)| for k=1..N-1
    plus a constant term for k=0.
    """
    R = _autocorr(f)
    # The C3 bound involves the autocorrelation sidelobes
    # Use L1 norm of sidelobes divided by N^2
    l1_sidelobe = np.sum(np.abs(R[1:]))
    return l1_siolobe / (N * N) if False else l1_sidelobe / (N * N)


def _construct_barker_like(n):
    """Construct a Barker-like sequence using known short Barker codes
    tiled and adjusted to length n. Barker sequences have optimal
    autocorrelation properties."""
    # Barker sequences of known lengths: 2, 3, 4, 5, 7, 11, 13
    # We'll use a combination approach
    # Start with a Thue-Morse-like construction which has good spectral properties
    f = np.ones(n)
    # Use a recursive construction: f = [g, -g] where g is a good sequence
    # This creates a Golay complementary pair structure
    # For length 400, we can use a product construction
    
    # Simpler: use a chaotic map to generate signs with good distribution
    # Or use the known property that ±1 sequences from Legendre symbols
    # have good autocorrelation
    
    # Use Legendre symbol construction for prime length, pad to 400
    # Find a prime near 400
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    # Use prime 397 (closest prime below 400)
    p = 397
    f_prime = np.zeros(p)
    for i in range(1, p):
        # Legendre symbol: (i/p) = i^((p-1)/2) mod p
        ls = pow(i, (p - 1) // 2, p)
        if ls == 1:
            f_prime[i] = 1.0
        else:
            f_prime[i] = -1.0
    f_prime[0] = 1.0
    
    # Pad to length 400
    f = np.ones(n)
    f[:p] = f_prime
    # Fill remaining with alternating signs to break periodicity
    for i in range(p, n):
        f[i] = 1.0 if (i - p) % 2 == 0 else -1.0
    return f


def _greedy_coordinate_descent(f, time_budget, rng):
    """Greedy coordinate descent: for each position, check if flipping
    improves the peak sidelobe. Use analytic update for speed."""
    f = f.copy()
    R = _autocorr(f)
    current_peak = np.max(np.abs(R[1:]))
    best_f = f.copy()
    best_peak = current_peak
    
    elapsed_limit = time_budget
    improved = True
    max_passes = 50
    
    for pass_num in range(max_passes):
        if time.time() > elapsed_limit:
            break
        improved = False
        
        # Process positions in random order for diversity
        order = rng.permutation(N)
        
        for idx in order:
            if time.time() > elapsed_limit:
                break
            
            # Try flipping bit idx
            f[idx] *= -1.0
            
            # Recompute autocorrelation efficiently
            # Flipping f[idx] changes R[k] by -2*f[idx]*f[idx+k] for k>0
            # and -2*f[idx]*f[idx-k] for k<0 (but we only track k>=0)
            # Actually, R[k] = sum_i f[i]*f[i+k], so flipping f[idx] changes:
            #   R[k] -= 2*f[idx_old]*f[idx+k] for each k where idx+k < N
            #   R[k] -= 2*f[idx_old]*f[idx-k] for each k where idx-k >= 0
            # This is complex; just recompute for correctness
            
            R_new = _autocorr(f)
            peak_new = np.max(np.abs(R_new[1:]))
            
            if peak_new < current_peak - 1e-12:
                current_peak = peak_new
                R = R_new
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_f = f.copy()
            else:
                f[idx] *= -1.0  # revert
                # R stays the same
        
        if not improved:
            break
    
    return best_f, best_peak


def _fast_greedy_with_analytic_update(f, time_budget, rng):
    """Faster greedy using analytic autocorrelation updates.
    
    When we flip f[i] from +1 to -1 (or vice versa), the change in 
    autocorrelation R[k] is:
    R_new[k] = R_old[k] - 2*f_old[i]*f[i+k]  for k such that i+k in [0,N)
    R_new[k] = R_old[k] - 2*f_old[i]*f[i-k]  for k such that i-k in [0,N)
    
    More precisely, R[k] = sum_{j=0}^{N-1-k} f[j]*f[j+k]
    Flipping f[i] affects terms where j=i or j+k=i, i.e., j=i or j=i-k.
    So R[k] changes by -2*f_old[i]*f[i+k] if i+k < N, and 
    -2*f_old[i]*f[i-k] if i-k >= 0 and k > 0.
    """
    f = f.copy()
    R = _autocorr(f)
    current_peak = np.max(np.abs(R[1:]))
    best_f = f.copy()
    best_peak = current_peak
    
    elapsed_limit = time_budget
    improved = True
    max_passes = 100
    
    for pass_num in range(max_passes):
        if time.time() > elapsed_limit:
            break
        improved = False
        
        order = rng.permutation(N)
        
        for idx in order:
            if time.time() > elapsed_limit:
                break
            
            f_old = f[idx]
            f[idx] = -f_old
            
            # Update R analytically
            # R[k] = sum_j f[j]*f[j+k]
            # The term involving f[idx]: 
            #   When j = idx: f[idx]*f[idx+k] -> -f[idx]*f[idx+k], change = -2*f_old*f[idx+k]
            #   When j+k = idx, i.e., j = idx-k: f[idx-k]*f[idx] -> f[idx-k]*(-f_old), change = -2*f_old*f[idx-k]
            # But we must be careful not to double count when idx+k = idx-k, i.e., 2k=0, k=0.
            # For k=0: R[0] = sum f[j]^2 = N (constant for ±1), so no change needed.
            
            R_new = R.copy()
            
            # For each lag k=1..N-1:
            # Contribution from j=idx: if idx+k < N, R[k] -= 2*f_old*f[idx+k]
            # Contribution from j=idx-k: if idx-k >= 0 and idx-k + k = idx < N, R[k] -= 2*f_old*f[idx-k]
            
            # Vectorized update
            k_vals = np.arange(1, N)
            
            # Case 1: j = idx, need idx + k < N
            valid1 = (idx + k_vals) < N
            if np.any(valid1):
                k1 = k_vals[valid1]
                R_new[k1] -= 2.0 * f_old * f[idx + k1]
            
            # Case 2: j = idx - k, need idx - k >= 0
            valid2 = (idx - k_vals) >= 0
            if np.any(valid2):
                k2 = k_vals[valid2]
                R_new[k2] -= 2.0 * f_old * f[idx - k2]
            
            peak_new = np.max(np.abs(R_new[1:]))
            
            if peak_new < current_peak - 1e-12:
                current_peak = peak_new
                R = R_new
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_f = f.copy()
            else:
                f[idx] = f_old  # revert
                # R unchanged
        
        if not improved:
            break
    
    return best_f, best_peak


def _simulated_annealing_fast(f_init, time_budget, rng):
    """Simulated annealing with fast analytic updates for single-bit flips."""
    f = f_init.copy()
    R = _autocorr(f)
    current_peak = np.max(np.abs(R[1:]))
    best_f = f.copy()
    best_peak = current_peak
    
    T_start = 0.1
    T_end = 0.001
    total_steps = 50000
    elapsed_limit = time_budget
    
    for step in range(total_steps):
        if time.time() > elapsed_limit:
            break
        
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        idx = rng.integers(N)
        f_old = f[idx]
        f[idx] = -f_old
        
        # Analytic update
        R_new = R.copy()
        k_vals = np.arange(1, N)
        
        valid1 = (idx + k_vals) < N
        if np.any(valid1):
            k1 = k_vals[valid1]
            R_new[k1] -= 2.0 * f_old * f[idx + k1]
        
        valid2 = (idx - k_vals) >= 0
        if np.any(valid2):
            k2 = k_vals[valid2]
            R_new[k2] -= 2.0 * f_old * f[idx - k2]
        
        peak_new = np.max(np.abs(R_new[1:]))
        delta = peak_new - current_peak
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            f[idx] = -f_old  # accept
            R = R_new
            current_peak = peak_new
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
        else:
            f[idx] = f_old  # reject
    
    return best_f, best_peak


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple diverse initial candidates
    candidates = []
    
    # 1. Legendre symbol construction
    candidates.append(_construct_barker_like(N))
    
    # 2. Random
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Thue-Morse
    tm = np.zeros(N)
    for i in range(N):
        cnt = 0
        x = i
        while x:
            cnt += x & 1
            x >>= 1
        tm[i] = 1.0 if cnt % 2 == 0 else -1.0
    candidates.append(tm)
    
    # 4. Another random with different seed
    rng2 = np.random.default_rng(999)
    candidates.append(rng2.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates by peak sidelobe
    best_f = None
    best_peak = np.inf
    for cand in candidates:
        peak = _peak_sidelobe(cand)
        if peak < best_peak:
            best_peak = peak
            best_f = cand.copy()
    
    # Phase 1: Fast greedy coordinate descent
    greedy_time = start_time + time_limit * 0.4
    best_f, best_peak = _fast_greedy_with_analytic_update(best_f, greedy_time, rng)
    
    # Phase 2: Simulated annealing from the greedy result
    sa_time = start_time + time_limit * 0.8
    best_f, best_peak = _simulated_annealing_fast(best_f, sa_time, rng)
    
    # Phase 3: Final greedy pass
    remaining_time = start_time + time_limit
    if time.time() < remaining_time - 5:
        best_f, best_peak = _fast_greedy_with_analytic_update(best_f, remaining_time, rng)
    
    # Return the sequence
    return best_f
