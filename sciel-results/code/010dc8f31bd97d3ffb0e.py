import numpy as np
import math

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Simulated Annealing approach
    np.random.seed(123) # Fixed seed for reproducibility of the search
    x = np.random.choice([-1, 1], size=N)
    
    def objective(x):
        r = np.correlate(x, x, mode='full')
        max_abs_r = np.max(np.abs(r))
        s = np.sum(x)
        s_sq = s * s
        if s_sq == 0:
            return float('inf')
        return max_abs_r / s_sq

    best_x = x.copy()
    best_score = objective(x)
    current_x = x.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    alpha = 0.9995  # Cooling rate
    iterations = 30000 # 30k iterations should be fast enough
    
    for i in range(iterations):
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Generate a move: flip k random bits
        k = np.random.randint(1, 6)
        indices = np.random.choice(N, size=k, replace=False)
        
        # Apply move
        current_x[indices] *= -1
        new_score = objective(current_x)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            # Check if we are escaping an infinite state
            if current_score == float('inf') and new_score != float('inf'):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = current_x.copy()
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if T > 0 and delta > 0:
                    accept_prob = math.exp(-delta / T)
                    if np.random.rand() < accept_prob:
                        current_score = new_score
                    else:
                        # Reject: revert
                        current_x[indices] *= -1
                else:
                    # Reject: revert
                    current_x[indices] *= -1
                
    return best_x
# EVOLVE-BLOCK-END
