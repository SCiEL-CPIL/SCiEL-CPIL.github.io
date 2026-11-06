import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 24
    k = np.arange(n)
    
    # Grid for evaluation
    # Use a moderate grid size for speed, refinement later if needed?
    # The task says "accuracy about 1e-9". 
    # A grid of 4096 might be enough if we don't refine?
    # Or we can refine the best candidate at the end.
    m = 2048
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    # powers_grid[k, i] = exp(i * k * thetas[i])
    powers_grid = np.exp(1j * np.outer(k, thetas))
    
    def get_max_norm_from_vals(vals):
        # vals: complex array of shape (m,)
        return np.max(np.abs(vals))

    def get_max_norm(c):
        # c: numpy array of ints
        # V = dot(c, powers_grid)
        V = np.dot(c, powers_grid)
        return get_max_norm_from_vals(V)

    best_c = np.random.choice([-1, 1], size=n)
    best_V = np.dot(best_c, powers_grid)
    best_score = get_max_norm_from_vals(best_V)
    
    # Vectorized Steepest Descent
    # Precompute Z_k for each k
    # Z_all shape (n, m)
    Z_all = powers_grid
    
    while time.time() < deadline:
        # Random restart
        if random.random() < 0.3 or best_score > 1.26: # If not close to opt, restart more often
            c = np.random.choice([-1, 1], size=n)
        else:
            c = best_c.copy()
            # Perturb
            for _ in range(3):
                idx = random.randint(0, n-1)
                c[idx] *= -1
        
        V = np.dot(c, powers_grid)
        current_score = get_max_norm_from_vals(V)
        
        # Steepest Descent Loop
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Compute max norm for all single-bit flips
            # V_flips[k, i] = V[i] - 2 * c[k] * Z_all[k, i]
            # Shape: (n, m)
            V_flips = V[np.newaxis, :] - 2.0 * c[:, np.newaxis] * Z_all
            
            abs_flips = np.abs(V_flips)
            max_norms = np.max(abs_flips, axis=1)
            
            # Find the flip that minimizes the max norm
            min_idx = np.argmin(max_norms)
            min_norm = max_norms[min_idx]
            
            if min_norm < current_score - 1e-12:
                # Apply flip
                c[min_idx] *= -1
                V = V_flips[min_idx] # Update V directly
                current_score = min_norm
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
            else:
                # No single-bit flip improves
                # Try double-bit flips?
                # This is more expensive but might escape local minima.
                # Let's try a random double-bit flip if stuck?
                # Or just break and restart.
                break
        
        # Optional: Local refinement of the best candidate found so far?
        # The grid max might be slightly off from the true max.
        # We can do a golden section search on the top 3 peaks of best_c.
        # But for now, let's rely on the grid.
        # To improve accuracy, we can increase m for the final check?
        # No, keep m constant for speed.
        
    # Final polish: Refine the best_c on a finer grid?
    # The task requires high accuracy.
    # Let's do a high-precision check on best_c.
    # Use a finer grid (e.g., 16384) and refine peaks.
    
    def refine_peak(c, theta_center, delta):
        # Golden section search
        def eval_p(theta):
            z = np.exp(1j * theta)
            # Vectorized eval?
            # sum c_k z^k
            # np.dot(c, z**k)
            # z**k is expensive.
            # Use Horner?
            # Or precompute powers for the fine grid?
            # Since we only do this once or twice, it's okay.
            return np.abs(np.dot(c, np.exp(1j * k * theta)))
            
        gr = 0.6180339887498949
        a = theta_center - delta
        b = theta_center + delta
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = eval_p(c1)
        f2 = eval_p(c2)
        
        for _ in range(30):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = eval_p(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = eval_p(c2)
                
        return max(f1, f2, eval_p(theta_center))

    # Find peaks in best_c
    V_best = np.dot(best_c, powers_grid)
    abs_best = np.abs(V_best)
    # Get top 5 indices
    top_indices = np.argsort(abs_best)[-5:]
    
    delta = 2 * np.pi / m
    max_refined = 0
    for idx in top_indices:
        theta_peak = thetas[idx]
        val = refine_peak(best_c, theta_peak, 2*delta)
        if val > max_refined:
            max_refined = val
            
    # The score is max_refined / sqrt(24)
    # But the evaluator does this. We just return the coeffs.
    # We don't need to return the score.
    
    return best_c.tolist()
