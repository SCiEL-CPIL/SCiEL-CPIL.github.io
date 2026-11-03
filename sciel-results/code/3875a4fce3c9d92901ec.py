import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    def _autoconv(s):
        return fftconvolve(s, s, mode='full')

    def _squared_integral(s):
        return (np.sum(s)) ** 2

    def _raw_value(s):
        si = _squared_integral(s)
        if si < 1e-15:
            return np.inf
        ac = _autoconv(s)
        return np.max(np.abs(ac)) / si

    def _evaluate_flip(s, i):
        """Quickly evaluate the effect of flipping element i.
        Flipping s[i] changes the autoconvolution in a structured way.
        We recompute for simplicity given the FFT speed."""
        s_flip = s.copy()
        s_flip[i] = -s_flip[i]
        return _raw_value(s_flip)

    def _local_search_flip(s, max_iters=3000):
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try all single flips
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                val = _evaluate_flip(best_s, i)
                if val < best_val:
                    best_val = val
                    best_s[i] = -best_s[i]
                    improved = True
                    break  # restart after improvement
            
            if not improved:
                break
        
        return best_s, best_val

    def _local_search_two_flip(s, max_iters=1000):
        """Try flipping two elements at a time."""
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try random pairs of flips
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                if i == j:
                    continue
                
                s_trial = best_s.copy()
                s_trial[i] = -s_trial[i]
                s_trial[j] = -s_trial[j]
                val = _raw_value(s_trial)
                if val < best_val:
                    best_val = val
                    best_s = s_trial
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_s, best_val

    def _local_search_block(s, max_iters=500):
        """Try flipping contiguous blocks."""
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for _ in range(80):
                if time.time() > deadline - 5:
                    break
                start = np.random.randint(0, N)
                length = np.random.randint(2, N // 5 + 1)
                end = min(start + length, N)
                
                s_trial = best_s.copy()
                s_trial[start:end] = -s_trial[start:end]
                val = _raw_value(s_trial)
                if val < best_val:
                    best_val = val
                    best_s = s_trial
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_s, best_val

    def _random_signs(seed):
        np.random.seed(seed)
        return np.random.choice([-1.0, 1.0], size=N)

    def _random_smoothed(seed):
        np.random.seed(seed)
        s = np.random.randn(N)
        # Light smoothing to create correlated structure
        kernel = np.ones(7) / 7
        s = np.convolve(s, kernel, mode='same')
        return np.sign(s)

    def _random_with_gaps(seed, gap_prob=0.3):
        """Random signs with some zero-like gaps (represented by sign flips in clusters)."""
        np.random.seed(seed)
        s = np.random.choice([-1.0, 1.0], size=N)
        # Create a few longer runs of same sign
        for _ in range(5):
            start = np.random.randint(0, N - 20)
            length = np.random.randint(5, 25)
            s[start:start+length] = s[start]
        return s

    # Generate diverse initial candidates
    np.random.seed(42)
    candidates = []
    
    # Different initialization strategies
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        candidates.append(_random_signs(seed))
    
    for seed in [11, 22, 33, 44]:
        candidates.append(_random_smoothed(seed))
    
    for seed in [55, 66, 77]:
        candidates.append(_random_with_gaps(seed))
    
    # Optimize each candidate with multiple search phases
    best_s = None
    best_val = np.inf
    
    for s in candidates:
        if time.time() > deadline - 15:
            break
        
        # Phase 1: Single flip local search (most important)
        s1, val1 = _local_search_flip(s, max_iters=2000)
        if val1 < best_val:
            best_val = val1
            best_s = s1.copy()
        
        if time.time() > deadline - 10:
            break
        
        # Phase 2: Two-flip search
        s2, val2 = _local_search_two_flip(s1, max_iters=500)
        if val2 < best_val:
            best_val = val2
            best_s = s2.copy()
        
        if time.time() > deadline - 8:
            break
        
        # Phase 3: Block search
        s3, val3 = _local_search_block(s2, max_iters=300)
        if val3 < best_val:
            best_val = val3
            best_s = s3.copy()
        
        if time.time() > deadline - 5:
            break
        
        # Phase 4: Final polish with single flips
        s4, val4 = _local_search_flip(s3, max_iters=1000)
        if val4 < best_val:
            best_val = val4
            best_s = s4.copy()
    
    # Random restarts from best solution
    if best_s is not None and time.time() < deadline - 10:
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            s_perturbed = best_s.copy()
            # Flip 3-8 random elements
            n_flips = np.random.randint(3, 9)
            for _ in range(n_flips):
                idx = np.random.randint(0, N)
                s_perturbed[idx] = -s_perturbed[idx]
            
            s_new, val_new = _local_search_flip(s_perturbed, max_iters=800)
            if val_new < best_val:
                best_val = val_new
                best_s = s_new.copy()
    
    # Fallback
    if best_s is None:
        best_s = np.ones(N)
        best_s[0] = -1.0
    
    return best_s

# EVOLVE-BLOCK-END
