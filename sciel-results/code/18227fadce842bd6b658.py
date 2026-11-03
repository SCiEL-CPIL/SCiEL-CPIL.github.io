"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit
    
    # Start with a random sign sequence
    rng = np.random.RandomState(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Precompute initial autocorrelation and sum
    s = np.sum(x)
    c = np.convolve(x, x, mode='full')
    max_abs_c = np.max(np.abs(c))
    
    def current_score():
        if abs(s) < 1e-10:
            return float('inf')
        return max_abs_c / (s * s)
    
    best_score = current_score()
    best_x = x.copy()
    best_c = c.copy()
    best_s = s
    best_max_abs = max_abs_c
    
    # Local search with incremental updates
    improved = True
    iteration = 0
    while improved and (time.time() - start_time) < deadline:
        improved = False
        iteration += 1
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            
            old_val = x[i]
            new_val = -old_val
            delta = new_val - old_val  # -2*old_val
            
            # Update sum
            new_s = s + delta
            
            # Update autocorrelation: c[k] += delta * (x[i+k] + x[i-k]) for valid k
            # c[k] = sum_j x[j]*x[j+k], flipping x[i] affects terms where j=i or j+k=i
            # c[k] changes by delta*x[i+k] (when j=i) + x[i]*delta (when j+k=i, i.e., j=i-k)
            # But need to be careful about bounds
            
            new_c = c.copy()
            # For each lag k, the change is:
            # If we flip x[i], then for lag k:
            #   c[k] had contribution x[i]*x[i+k] (if i+k < N) and x[i-k]*x[i] (if i-k >= 0)
            #   New contribution: (-x[i])*x[i+k] and x[i-k]*(-x[i])
            #   Change: -2*x[i]*x[i+k] (if i+k<N) - 2*x[i-k]*x[i] (if i-k>=0)
            #   But wait, these might be the same term if k=0
            
            # Actually, let's just recompute the affected lags
            # Lag k ranges from -(N-1) to (N-1) in full convolution
            # c_full[k] for k in [0, 2N-2] corresponds to lag k-(N-1)
            
            # Simpler: just recompute c for the affected indices
            # Flipping x[i] affects c_full[m] for m where the product x[j]*x[j+m-N+1] includes x[i]
            # This is complex, let's just recompute if N is small enough
            
            if N <= 500:
                # Recompute autocorrelation
                new_c = np.convolve(x * (x != old_val) * new_val + x * (x == old_val) * x, 
                                     x * (x != old_val) * new_val + x * (x == old_val) * x, 
                                     mode='full')
                # That's wrong, let me just do it properly
                x[i] = new_val
                new_c = np.convolve(x, x, mode='full')
                new_max_abs = np.max(np.abs(new_c))
                
                if abs(new_s) > 1e-10:
                    new_score = new_max_abs / (new_s * new_s)
                else:
                    new_score = float('inf')
                
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
                    best_c = new_c.copy()
                    best_s = new_s
                    best_max_abs = new_max_abs
                    s = new_s
                    c = new_c
                    max_abs_c = new_max_abs
                    improved = True
                else:
                    x[i] = old_val
                    c = c  # unchanged since we reverted
            else:
                x[i] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
