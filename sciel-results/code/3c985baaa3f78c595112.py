import numpy as np

N = 100

def construct():
    # Initialize with a random binary pattern: N/2 zeros and N/2 ones
    # This ensures sum is exactly N/2 and values are in [0,1]
    h_current = np.zeros(N)
    h_current[:N//2] = 1.0
    
    def calc_max_c(h):
        # Calculate max C_k
        # r_k = sum h[i] * h[i+k]
        # Using np.correlate for speed
        r = np.correlate(h, h, mode='full')[N-1:]
        cs = (2.0 / N) * (N / 2.0 - r)
        return np.max(cs)

    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.995
    max_iter = 20000
    
    score_current = calc_max_c(h_current)
    h_best = h_current.copy()
    score_best = score_current
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        # Generate a neighbor
        h_new = h_current.copy()
        
        # Move type: 80% swap, 20% perturb
        if rng.random() < 0.8:
            # Swap two elements
            idx = rng.choice(N, size=2, replace=False)
            h_new[idx[0]], h_new[idx[1]] = h_current[idx[1]], h_current[idx[0]]
        else:
            # Perturb two elements
            idx = rng.choice(N, size=2, replace=False)
            delta = rng.uniform(-0.1, 0.1)
            h_new[idx[0]] += delta
            h_new[idx[1]] -= delta
            
            # Check bounds
            if h_new[idx[0]] < 0 or h_new[idx[0]] > 1 or h_new[idx[1]] < 0 or h_new[idx[1]] > 1:
                # Fallback to swap if perturb invalid
                h_new = h_current.copy()
                idx = rng.choice(N, size=2, replace=False)
                h_new[idx[0]], h_new[idx[1]] = h_current[idx[1]], h_current[idx[0]]
        
        score_new = calc_max_c(h_new)
        
        # Metropolis criterion
        delta_score = score_new - score_current
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            h_current = h_new
            score_current = score_new
            
            if score_new < score_best:
                h_best = h_new.copy()
                score_best = score_new
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return h_best
