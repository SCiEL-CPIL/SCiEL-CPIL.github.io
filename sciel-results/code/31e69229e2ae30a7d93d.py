import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Initialize with a random sequence of -1 and 1
    x = np.random.choice([-1, 1], size=N)
    
    def get_score(x):
        r = np.correlate(x, x, mode='full')
        max_abs_r = np.max(np.abs(r))
        s = np.sum(x)
        s_sq = s * s
        if s_sq == 0:
            return float('inf')
        return max_abs_r / s_sq

    current_x = x.copy()
    current_score = get_score(current_x)
    best_x = current_x.copy()
    best_score = current_score

    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-6
    num_iterations = 500000
    
    # Geometric cooling
    if T_end > 0:
        alpha = (T_end / T_start) ** (1.0 / num_iterations)
    else:
        alpha = 1.0 # Should not happen
        
    for i in range(num_iterations):
        T = T_start * (alpha ** i)
        
        # Propose move: flip a random bit
        idx = np.random.randint(N)
        current_x[idx] *= -1
        
        new_score = get_score(current_x)
        
        # Acceptance probability
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if T > 0 and np.isfinite(delta):
                prob = np.exp(-delta / T)
            else:
                prob = 0.0
            
            if np.random.rand() < prob:
                current_score = new_score
            else:
                # Revert
                current_x[idx] *= -1
                
    return best_x
# EVOLVE-BLOCK-END
