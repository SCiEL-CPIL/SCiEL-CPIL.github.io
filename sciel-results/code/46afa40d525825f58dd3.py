import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    Uses direct O(N^2) computation via vectorized NumPy.
    """
    # Shift domain to [0, 1]
    # f(s) = signed_step[i] for s in [i/N, (i+1)/N)
    # Autoconvolution at t = k/N:
    # (f*f)(k/N) = (1/N) * sum_{i=0}^{N-1-k} signed_step[i] * signed_step[i+k] for k >= 0
    # By symmetry, (f*f)(-k/N) = (f*f)(k/N)
    
    # Compute all lags k = 0, 1, ..., N-1
    # For lag k, the sum is sum_{i=0}^{N-1-k} f[i] * f[i+k]
    # This is the autocorrelation at lag k
    
    # Use a vectorized approach: for each lag k, compute the dot product
    # of f[0:N-k] and f[k:N]
    
    # Build a Toeplitz-like matrix or use direct computation
    # For efficiency, compute all lags at once using a sliding window
    
    # Method: create a matrix where row k contains f[0:N-k] and f[k:N]
    # But that's O(N^2) memory. Instead, compute lag by lag but vectorize.
    
    # Actually, let's use the fact that we can compute this via FFT more efficiently
    # but let's try direct first for correctness
    
    f = signed_step.astype(np.float64)
    
    # Compute autocorrelation at all lags using np.correlate
    # np.correlate(f, f, mode='full') gives correlations at lags -(N-1) to (N-1)
    # But we want the sum, not the normalized correlation
    
    corr = np.correlate(f, f, mode='full')
    # corr[k + N - 1] = sum_{i} f[i] * f[i+k] for k >= 0
    # The peak is at k=0: corr[N-1] = sum f[i]^2 = N
    
    # The autoconvolution integral at t = k/N is (1/N) * corr[k + N - 1] for k >= 0
    # But we need to be careful: the actual autoconvolution of the step function
    # is a piecewise linear function, and its maximum occurs at the breakpoints
    # t = k/N.
    
    # For a step function on [0,1] with N intervals of width 1/N:
    # (f*f)(t) for t in [0,1] is:
    # (f*f)(t) = sum_{i=0}^{N-1} f[i] * (1/N) * (number of j such that j/N <= t - i/N < (j+1)/N)
    # This equals (1/N) * sum_{i=0}^{N-1} f[i] * f[floor(N*t - i)] (with appropriate bounds)
    
    # The maximum of |f*f| occurs at t = k/N for integer k.
    # At t = k/N: (f*f)(k/N) = (1/N) * sum_{i=max(0,k-N+1)}^{min(N-1,k)} f[i] * f[k-i]
    
    # This is exactly (1/N) * corr[k + N - 1] for k in range(-N+1, N)
    
    # So C3 = max_k |corr[k + N - 1]| / N
    
    c3 = np.max(np.abs(corr)) / N
    
    return c3

def construct():
    start_time = time.time()
    
    # Use multiple restarts with different initializations
    best_step = None
    best_c3 = float('inf')
    
    np.random.seed(42)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Random
            current = np.random.choice([-1, 1], size=N).astype(np.int8)
        elif restart == 1:
            # Alternating pattern
            current = np.ones(N, dtype=np.int8)
            current[1::2] = -1
        elif restart == 2:
            # Walsh-like: use Hadamard matrix first row
            # Simple: blocks of same sign
            block_size = max(1, N // 8)
            current = np.zeros(N, dtype=np.int8)
            for i in range(N):
                current[i] = 1 if (i // block_size) % 2 == 0 else -1
        elif restart == 3:
            # Quadratic phase
            x = np.linspace(0, 2*np.pi, N)
            current = np.sign(np.cos(x)).astype(np.int8)
            current[current == 0] = 1
        else:
            # Random with bias toward previous best
            if best_step is not None:
                current = best_step.copy()
                # Flip some random elements
                num_flips = np.random.randint(10, 50)
                flip_idx = np.random.choice(N, size=num_flips, replace=False)
                current[flip_idx] = -current[flip_idx]
            else:
                current = np.random.choice([-1, 1], size=N).astype(np.int8)
        
        current_c3 = _compute_c3(current)
        
        # Local search: steepest descent with random restarts within
        improved = True
        max_local_iters = 100
        local_iter = 0
        while improved and local_iter < max_local_iters:
            improved = False
            local_iter += 1
            
            if time.time() - start_time > DEADLINE - 2.0:
                break
            
            # Try all single flips and find the best one
            best_delta = 0
            best_idx = -1
            
            # Sample a subset of indices for speed
            sample_size = min(N, 100)
            if N <= 100:
                indices = np.arange(N)
            else:
                indices = np.random.choice(N, size=sample_size, replace=False)
            
            for idx in indices:
                old_val = current[idx]
                current[idx] = -old_val
                new_c3 = _compute_c3(current)
                delta = current_c3 - new_c3  # positive if improvement
                current[idx] = old_val
                
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
            
            if best_idx >= 0:
                current[best_idx] = -current[best_idx]
                current_c3 -= best_delta
                improved = True
        
        if current_c3 < best_c3:
            best_c3 = current_c3
            best_step = current.copy()
    
    return best_step
