import numpy as np
import time

N = 50

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = 0.0
    best_h = np.ones(N) # Fallback
    
    # Helper to compute score
    def score(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Convolution
        c = np.convolve(h, h, mode='full')
        l_inf = np.max(c)
        return l2_sq / (l1 * l_inf)
    
    # Initial candidates
    candidates = []
    
    # 1. Single Spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    candidates.append(h_spike)
    
    # 2. Uniform Block
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    candidates.append(h_block)
    
    # 3. Uniform
    h_uniform = np.ones(N)
    candidates.append(h_uniform)
    
    # 4. Random
    for _ in range(10):
        h_rand = np.random.rand(N)
        candidates.append(h_rand)
        
    # Evaluate initial candidates
    for h in candidates:
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # Local Search
    # We normalize h to L1=1 during search to keep numbers stable
    # The score is scale invariant, so this is safe.
    
    current_h = best_h.copy()
    current_score = best_score
    
    # If we found a score close to 1, we might be at the optimum.
    # But let's try to improve.
    
    steps = 0
    max_steps = 10000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        # Normalize
        l1 = np.sum(current_h)
        if l1 == 0:
            current_h = np.ones(N)
            l1 = np.sum(current_h)
        current_h = current_h / l1
        
        # Pick a random index to perturb
        idx = np.random.randint(N)
        
        # Try increasing and decreasing
        old_val = current_h[idx]
        
        # Perturbation
        delta = np.random.uniform(-0.1, 0.1)
        new_val = max(0.0, old_val + delta)
        
        # Also try setting to 0 or 1
        if np.random.rand() < 0.1:
            new_val = 0.0
        elif np.random.rand() < 0.1:
            new_val = old_val + np.random.uniform(0.1, 0.5)
            
        current_h[idx] = new_val
        
        # Re-normalize
        l1_new = np.sum(current_h)
        if l1_new == 0:
            current_h = np.ones(N)
            l1_new = np.sum(current_h)
        current_h = current_h / l1_new
        
        new_score = score(current_h)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            # Revert
            current_h[idx] = old_val
            l1_revert = np.sum(current_h)
            if l1_revert > 0:
                current_h = current_h / l1_revert
            else:
                current_h = best_h.copy()
                
    # Final check: ensure best_h is valid and non-negative
    best_h = np.maximum(best_h, 0)
    return list(best_h)
