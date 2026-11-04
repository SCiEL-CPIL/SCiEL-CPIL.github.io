"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = np.zeros(n, dtype=np.int8)
    seq[0] = 1
    for i in range(1, n):
        # Thue-Morse: t(n) = parity of number of 1s in binary representation
        seq[i] = 1 - 2 * (bin(i).count('1') % 2)
    return seq

def _autocorr_bound(a):
    """Compute the max absolute autocorrelation over non-zero lags (upper bound on C3)."""
    n = len(a)
    # Use FFT for efficient autocorrelation
    f = np.fft.fft(a, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # Exclude lag 0
    return np.max(np.abs(acf[1:]))

def _flip_delta(a, idx):
    """Compute change in objective if we flip a[idx]."""
    # Flipping a[idx] from v to -v changes autocorrelation at lag k by:
    # For lag k > 0: acf[k] changes by -2*a[idx]*a[idx-k] (if idx-k in range) and -2*a[idx]*a[idx+k] (if idx+k in range)
    # This is O(N) per flip, so we'll just recompute for simplicity in the inner loop
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initialize with Thue-Morse
    a = _thue_morse(N).astype(np.float64)
    current_bound = _autocorr_bound(a)
    best_a = a.copy()
    best_bound = current_bound
    
    # Greedy local search: repeatedly flip the bit that gives the largest improvement
    max_greedy_iters = 5000
    for it in range(max_greedy_iters):
        if time.time() > deadline:
            break
        best_delta = 0
        best_idx = -1
        # Sample a subset of indices for speed, or do full scan
        # Full scan: for each idx, compute new bound
        # To speed up, precompute contributions
        n = N
        # For flipping index i from a[i] to -a[i]:
        # New acf[k] = old_acf[k] - 2*a[i]*a[i-k] (for k where i-k >= 0) 
        #             - 2*a[i]*a[i+k] (for k where i+k < n)
        # We want to minimize max|acf[k]| for k>=1
        # This is complex to compute incrementally. Let's do a faster approach:
        # Precompute acf once, then for each candidate flip, estimate the new max.
        
        # Actually, let's just try random flips and keep improvements (stochastic greedy)
        # This is faster per iteration
        for trial in range(200):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            old_val = a[idx]
            a[idx] = -a[idx]
            new_bound = _autocorr_bound(a)
            if new_bound < current_bound:
                current_bound = new_bound
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_a = a.copy()
            else:
                a[idx] = old_val
                break  # No improvement in this trial batch, move on
        
        # Check for convergence
        if it > 100 and it % 100 == 0:
            # Check if we've been stuck
            pass
    
    # Simulated annealing phase
    if time.time() < deadline - 20:
        a = best_a.copy()
        current_bound = best_bound
        T = current_bound * 0.1
        T_min = current_bound * 0.001
        cooling = 0.999
        sa_iters = 0
        max_sa_iters = 200000
        
        while time.time() < deadline - 5 and sa_iters < max_sa_iters and T > T_min:
            idx = np.random.randint(N)
            old_val = a[idx]
            a[idx] = -a[idx]
            new_bound = _autocorr_bound(a)
            delta = new_bound - current_bound
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_bound = new_bound
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_a = a.copy()
            else:
                a[idx] = old_val
            T *= cooling
            sa_iters += 1
    
    # Final polish: a few more greedy passes
    if time.time() < deadline - 2:
        a = best_a.copy()
        current_bound = best_bound
        for _ in range(500):
            if time.time() > deadline:
                break
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                idx = np.random.randint(N)
                old_val = a[idx]
                a[idx] = -a[idx]
                new_bound = _autocorr_bound(a)
                if new_bound < current_bound:
                    current_bound = new_bound
                    if current_bound < best_bound:
                        best_bound = current_bound
                        best_a = a.copy()
                    improved = True
                else:
                    a[idx] = old_val
            if not improved:
                break
    
    return best_a
# EVOLVE-BLOCK-END
