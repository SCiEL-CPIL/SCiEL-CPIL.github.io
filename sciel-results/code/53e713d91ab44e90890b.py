import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial guess: Uniform
    f = np.ones(N) / N
    
    best_f = f.copy()
    best_score = np.inf
    
    # Helper to compute score (Max of centered convolution)
    def get_score(f):
        c = np.convolve(f, f, mode='full')
        mean_c = np.mean(c)
        # We want to minimize the max deviation from mean
        # Or just minimize max(c) if mean is fixed?
        # Let's minimize max(c) - min(c) to flatten
        return np.max(c) - np.min(c)
    
    current_score = get_score(f)
    best_score = current_score
    
    # Coordinate Descent / Local Search
    # We try to reduce the peak by moving mass from high-contributing bins to low-contributing bins
    
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        c = np.convolve(f, f, mode='full')
        mean_c = np.mean(c)
        
        # Find the peak index
        k_max = np.argmax(c)
        # Find the valley index
        k_min = np.argmin(c)
        
        # Gradient of c_k w.r.t f_i is f_{k-i}
        # To reduce c_k_max, we should decrease f_i where f_{k_max-i} is large.
        # To increase c_k_min, we should increase f_j where f_{k_min-j} is large.
        
        # Simple heuristic:
        # Identify i that contributes most to c_k_max
        # c_k_max = sum f_i f_{k_max-i}
        # Contribution of f_i is f_i * f_{k_max-i}
        
        # Let's try a small perturbation
        # Pick a random index i to decrease, and j to increase
        # Better: Pick i with high f_i * f_{k_max-i}
        
        contrib_max = f * f[k_max::-1] if k_max < N else np.zeros(N)
        # Handle indices properly for full convolution
        # c_k = sum_{i} f_i f_{k-i}
        # Indices for f_{k-i} must be valid.
        
        # Simplified: Just do a random walk with acceptance if score improves
        # This is slow but robust.
        
        # Faster: Gradient step on max(c)
        # Approximate gradient of max(c) is 1 at k_max, 0 else.
        # grad_f = f_{k_max - i}
        
        # Let's use a small step size
        step = 0.01
        grad = np.zeros(N)
        # Gradient of c_k_max w.r.t f
        # c_k_max = sum f_i f_{k_max-i}
        # d(c_k_max)/d(f_i) = f_{k_max-i}
        
        # We want to decrease f_i where f_{k_max-i} is large.
        # And increase f_j where f_{k_min-j} is large.
        
        # Construct gradient for objective: c_k_max - c_k_min
        g_max = np.zeros(N)
        g_min = np.zeros(N)
        
        for i in range(N):
            idx = k_max - i
            if 0 <= idx < N:
                g_max[i] = f[idx]
            
            idx2 = k_min - i
            if 0 <= idx2 < N:
                g_min[i] = f[idx2]
        
        grad = g_max - g_min
        
        # Update f
        f_new = f - step * grad
        
        # Project to simplex (f >= 0, sum f = 1)
        f_new = np.maximum(f_new, 0)
        if np.sum(f_new) > 0:
            f_new = f_new / np.sum(f_new)
        else:
            f_new = f.copy()
            
        new_score = get_score(f_new)
        
        if new_score < current_score:
            f = f_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        else:
            # Reduce step size
            step *= 0.9
            
    return list(best_f)
