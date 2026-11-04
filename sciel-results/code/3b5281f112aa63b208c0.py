"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(h):
    """Compute the autocorrelation vector R where R[k] = sum h[i]h[i+k]."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # We only care about lags 1 to n-1 for the score, but we need the full vector for updates
    # acf[k] for k in 0..n-1 corresponds to positive lags
    # Note: np.fft.irfft returns real part. For autocorrelation, it should be real.
    return acf[:n]

def _score_from_acf(acf):
    """Compute max absolute autocorrelation at non-zero lags from precomputed ACF."""
    if len(acf) <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _local_search_fast(h_init, max_iters=200, time_limit=5.0):
    """
    Fast local search using incremental ACF updates.
    Flipping h[i] to -h[i] changes ACF[k] by:
    For lag k > 0:
    R_new[k] = R_old[k] - 4 * h[i] * h[i-k]  (if i-k >= 0)
              R_new[k] = R_old[k] - 4 * h[i] * h[i+k]  (if i+k < n)
    Actually, the change is simpler if we think about the contribution of h[i].
    Let's derive the update:
    R[k] = sum_{j} h[j] h[j+k]
    If we flip h[i] -> -h[i]:
    Terms involving h[i] are:
    1. j=i: h[i] h[i+k] -> -h[i] h[i+k]. Change: -2 h[i] h[i+k]
    2. j=i-k: h[i-k] h[i] -> h[i-k] (-h[i]). Change: -2 h[i-k] h[i]
    
    So for each lag k:
    delta_R[k] = -2 h[i] h[i+k] (if i+k < n) - 2 h[i-k] h[i] (if i-k >= 0)
    
    We can precompute the "interaction" vectors or just update the ACF array directly.
    Updating the entire ACF array for one flip is O(N).
    Evaluating all N flips is O(N^2).
    N=400, N^2=160,000 operations per iteration. This is very fast in Python/NumPy if vectorized.
    """
    n = len(h_init)
    h = h_init.copy()
    acf = _compute_acf(h)
    current_score = _score_from_acf(acf)
    
    start_time = time.time()
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        best_gain = 0
        best_idx = -1
        best_new_acf = None
        best_new_score = current_score
        
        # We will calculate the change for each possible flip i
        # To speed this up, we can vectorize the calculation of the delta for all i?
        # It's tricky because the delta for lag k depends on i.
        # Let's just loop i, but update ACF efficiently.
        
        for i in range(n):
            # Calculate new ACF for flipping i
            # We don't want to copy the whole ACF N times.
            # Let's compute the delta array for this specific i.
            # delta[k] = -2*h[i]*h[i+k] - 2*h[i-k]*h[i]
            
            # This is still O(N) per i. Total O(N^2) per iteration.
            # 400*400 = 160k ops. 200 iters = 32M ops. Should be < 1s.
            
            # Optimization: Precompute h shifted?
            # Let's just do it simply first.
            
            # Create a temporary view or just calculate the score change?
            # We need the max of |acf + delta|.
            # We can't just look at the max of acf and delta.
            
            # Efficient way:
            # We will update a copy of acf.
            new_acf = acf.copy()
            
            # Apply change for flip at i
            # Term 1: j=i contributes to lags k where i+k < n
            # new_acf[k] += -2 * h[i] * h[i+k]
            # This is equivalent to: new_acf[0:n-i] += -2 * h[i] * h[i:n]
            
            # Term 2: j=i-k contributes to lags k where i-k >= 0 => k <= i
            # new_acf[k] += -2 * h[i-k] * h[i]
            # This is equivalent to: new_acf[0:i+1] += -2 * h[0:i+1][::-1] * h[i]
            
            if i < n:
                new_acf[0:n-i] += -2.0 * h[i] * h[i:n]
            if i >= 0:
                new_acf[0:i+1] += -2.0 * h[0:i+1][::-1] * h[i]
            
            new_score = _score_from_acf(new_acf)
            
            if new_score < best_new_score - 1e-12:
                best_new_score = new_score
                best_gain = current_score - new_score
                best_idx = i
                best_new_acf = new_acf
        
        if best_idx == -1:
            break
            
        # Apply the best flip
        h[best_idx] *= -1.0
        acf = best_new_acf
        current_score = best_new_score
        
    return h, current_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    """
    # Safety margin: 120s limit. Use 110s.
    deadline = time.time() + 110.0
    
    best_h = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. Thue-Morse (known good structure)
    h_tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    
    # 2. Random
    np.random.seed(42)
    h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
    
    candidates = [h_tm, h_rand]
    
    # Run local search on initial candidates
    for h_init in candidates:
        if time.time() > deadline - 10:
            break
        h_opt, score = _local_search_fast(h_init, max_iters=100, time_limit=5.0)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # Iterated Local Search (ILS) with random restarts
    # Since local search is fast now, we can do more restarts.
    
    while time.time() < deadline - 5:
        # Perturb the best solution found so far
        if best_h is None:
            h_current = np.random.choice([-1, 1], size=N).astype(np.float64)
        else:
            h_current = best_h.copy()
            # Flip a random number of bits
            k = np.random.randint(10, 30)
            indices = np.random.choice(N, size=k, replace=False)
            h_current[indices] *= -1
        
        # Local search
        h_opt, score = _local_search_fast(h_current, max_iters=100, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
