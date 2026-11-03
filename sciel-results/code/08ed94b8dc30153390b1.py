import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin for 150s limit

    def _autoconv_max_abs(s):
        ac = fftconvolve(s, s, mode='full')
        return np.max(np.abs(ac))

    def _squared_integral(s):
        return (np.sum(s)) ** 2

    def _raw_value(s):
        si = _squared_integral(s)
        if si < 1e-15:
            return np.inf
        return _autoconv_max_abs(s) / si

    def _soft_max_abs(x, temp):
        # Soft max of absolute values
        ax = np.abs(x)
        m = np.max(ax)
        if m == 0:
            return 0.0
        # Use log-sum-exp trick for stability
        scaled = ax / temp
        max_scaled = np.max(scaled)
        if max_scaled > 50:
            # Avoid overflow, just return max
            return m
        log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
        return temp * log_sum

    def _objective_smooth(s, temp):
        ac = fftconvolve(s, s, mode='full')
        # Exclude the center (k=0) which is the squared integral
        center = N - 1
        ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        sm = _soft_max_abs(ac_sidelobes, temp)
        si = _squared_integral(s)
        if si < 1e-15:
            return 1e10
        return sm / si

    def _local_search_flip(s, max_iters=5000):
        # Sign flip local search with O(N) incremental update
        # For sign flip at index i: s_new = s, s[i] = -s[i]
        # Change in autoconvolution:
        # A[k] = sum_j s[j] s[j+k]
        # If we flip s[i], the change in A[k] is:
        # delta A[k] = -2 * s[i] * (s[i+k] + s[i-k])  (with appropriate boundary handling)
        # Actually, let's just recompute for simplicity if N is small enough,
        # or use the incremental update.
        
        # For N=400, recomputing fftconvolve is fast (~ms).
        # But let's try incremental.
        
        # Actually, for sign flips, the change in the autoconvolution is:
        # Let s' = s with s[i] flipped to -s[i].
        # s'[j] = s[j] for j != i, s'[i] = -s[i].
        # A'[k] = sum_j s'[j] s'[j+k]
        # = sum_{j != i, j+k != i} s[j] s[j+k] + s'[i] s'[i+k] + s'[i+k] s'[i] + (if i == i+k, i.e., k=0, s'[i]^2)
        # This is getting complicated. Let's just recompute.
        # With FFT, it's very fast.
        
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        # Precompute indices to flip
        # We'll do a simple local search: try flipping each element, accept if better.
        # To speed up, we can use the fact that flipping s[i] changes the autoconvolution
        # in a structured way. But for now, let's just recompute.
        
        # Actually, let's use a smarter approach:
        # The autoconvolution A = fftconvolve(s, s, 'full')
        # If we flip s[i], the new autoconvolution is:
        # A_new = A - 2*s[i]*(s shifted by i) - 2*s[i]*(s shifted by -i) + 2*s[i]^2 * delta (for k=0)
        # This is not quite right. Let's just recompute.
        
        # For efficiency, let's batch the flips.
        # We'll try all N flips in each iteration.
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                s_flip = s.copy()
                s_flip[i] = -s_flip[i]
                val = _raw_value(s_flip)
                if val < best_val:
                    best_val = val
                    best_s = s_flip
                    s = s_flip
                    improved = True
                    break  # Restart the loop from index 0 after a successful flip
            
            if not improved:
                break
        
        return best_s, best_val

    def _local_search_block(s, max_iters=2000):
        # Block flip local search: flip a contiguous block of elements
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try random blocks
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                start = np.random.randint(0, N)
                length = np.random.randint(1, N//4 + 1)
                end = min(start + length, N)
                
                s_flip = s.copy()
                s_flip[start:end] = -s_flip[start:end]
                val = _raw_value(s_flip)
                if val < best_val:
                    best_val = val
                    best_s = s_flip
                    s = s_flip
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_s, best_val

    # Generate initial candidates
    np.random.seed(42)
    
    candidates = []
    
    # 1. Random signs
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        s = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(s)
    
    # 2. Barker-like (if available)
    # For N=400, there's no Barker code, but we can use other structured sequences
    # 3. Random with some structure
    for seed in [11, 22, 33]:
        np.random.seed(seed)
        s = np.random.randn(N)
        # Make it more sign-like
        s = np.sign(s)
        # Ensure not all same sign
        if np.all(s > 0) or np.all(s < 0):
            s[0] = -s[0]
        candidates.append(s)
    
    # 4. Smooth random
    for seed in [7, 8, 9]:
        np.random.seed(seed)
        s = np.random.randn(N)
        # Smooth it
        kernel = np.ones(5) / 5
        s = np.convolve(s, kernel, mode='same')
        s = np.sign(s)
        if np.all(s > 0) or np.all(s < 0):
            s[0] = -s[0]
        candidates.append(s)
    
    # Optimize each candidate
    best_s = None
    best_val = np.inf
    
    for s in candidates:
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Flip local search
        s1, val1 = _local_search_flip(s, max_iters=3000)
        if val1 < best_val:
            best_val = val1
            best_s = s1.copy()
        
        # Phase 2: Block local search
        if time.time() > deadline - 10:
            break
        s2, val2 = _local_search_block(s1, max_iters=1000)
        if val2 < best_val:
            best_val = val2
            best_s = s2.copy()
        
        # Phase 3: Another flip pass
        if time.time() > deadline - 10:
            break
        s3, val3 = _local_search_flip(s2, max_iters=2000)
        if val3 < best_val:
            best_val = val3
            best_s = s3.copy()
    
    # Final polish: try to improve with more aggressive search
    if best_s is not None and time.time() < deadline - 10:
        # Try random restarts from the best solution
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            s_perturbed = best_s.copy()
            # Flip a few random elements
            for _ in range(5):
                idx = np.random.randint(0, N)
                s_perturbed[idx] = -s_perturbed[idx]
            s_new, val_new = _local_search_flip(s_perturbed, max_iters=1000)
            if val_new < best_val:
                best_val = val_new
                best_s = s_new.copy()
    
    # Ensure the result is a valid sequence
    if best_s is None:
        # Fallback: return a simple sequence
        best_s = np.ones(N)
        best_s[0] = -1.0
    
    return best_s

# EVOLVE-BLOCK-END
