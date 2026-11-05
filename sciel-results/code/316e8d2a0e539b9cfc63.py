import numpy as np

N = 400

def construct():
    # New approach: Simulated Annealing / Stochastic Local Search
    # This differs from the gradient-based L-BFGS-B approach.
    
    def get_val(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))

    best_h = np.ones(N)
    best_val = get_val(best_h)
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.995 # Cooling rate
    max_iter = 5000
    
    # Try a few different starting points
    for start_iter in range(5):
        if start_iter == 0:
            h = np.ones(N)
        elif start_iter == 1:
            h = rng.normal(size=N)
        elif start_iter == 2:
            h = np.sin(np.arange(N) * 0.1)
        else:
            h = rng.random(N) * 2 - 1
            
        h = h.copy()
        current_val = get_val(h)
        
        T = T_start
        
        for i in range(max_iter):
            # Perturbation: flip a random subset of signs or add noise
            # Adding noise to a random subset
            idx = rng.integers(0, N, size=N//10)
            noise = rng.normal(size=len(idx)) * 0.1
            h_new = h.copy()
            h_new[idx] += noise
            
            new_val = get_val(h_new)
            
            if new_val < current_val:
                h = h_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Accept with probability exp(-(new_val - current_val) / T)
                delta = new_val - current_val
                if delta < 0:
                    p = 1.0
                else:
                    p = np.exp(-delta / T)
                if rng.random() < p:
                    h = h_new
                    current_val = new_val
            
            T *= alpha
            if T < T_end:
                break
                
    return best_h
