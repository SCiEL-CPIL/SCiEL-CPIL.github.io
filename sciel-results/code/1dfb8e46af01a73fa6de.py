import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 24
    
    # Precompute DFT matrix for fast evaluation
    # Use a fine grid to resolve peaks accurately
    M_grid = 2 ** 14  # 16384 points
    theta = 2 * np.pi * np.arange(M_grid) / M_grid
    # E[i, k] = exp(i * theta[i] * k)
    E = np.exp(1j * np.outer(theta, np.arange(N)))
    
    def get_supnorm(coeffs):
        """Compute sup norm of polynomial with given coefficients."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))
    
    # Generate initial Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    init_c = np.array(p[:N], dtype=float)
    
    # Local search: find best 1-flip and 2-flip moves
    def local_search(current_c, max_iters=100000):
        """Perform local search with 1-flip and 2-flip moves."""
        current_val = get_supnorm(current_c)
        best_c = current_c.copy()
        best_val = current_val
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try all 1-flip moves
            found_improvement = False
            best_flip_val = current_val
            best_flip_idx = -1
            
            for i in range(N):
                new_c = current_c.copy()
                new_c[i] *= -1
                new_val = get_supnorm(new_c)
                if new_val < best_flip_val - 1e-12:
                    best_flip_val = new_val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                current_c[best_flip_idx] *= -1
                current_val = best_flip_val
                found_improvement = True
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # Try 2-flip moves
            best_2flip_val = current_val
            best_2flip = None
            
            for i in range(N - 1):
                for j in range(i + 1, N):
                    new_c = current_c.copy()
                    new_c[i] *= -1
                    new_c[j] *= -1
                    new_val = get_supnorm(new_c)
                    if new_val < best_2flip_val - 1e-12:
                        best_2flip_val = new_val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                current_val = best_2flip_val
                found_improvement = True
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # No improvement found, we're at a local minimum
            break
        
        return best_c, best_val
    
    # Start with GR sequence
    current_c = init_c.copy()
    best_c, best_val = local_search(current_c)
    
    # Try additional random starts if time permits
    random_starts = 10
    for s in range(random_starts):
        if time.time() > deadline - 20:
            break
        
        # Generate random ±1 sequence
        random_c = np.random.choice([-1, 1], size=N).astype(float)
        rc, rv = local_search(random_c)
        
        if rv < best_val - 1e-12:
            best_c = rc
            best_val = rv
    
    # Final refinement with more iterations if time allows
    if time.time() < deadline - 5:
        refined_c, refined_val = local_search(best_c, max_iters=500000)
        if refined_val < best_val - 1e-12:
            best_c = refined_c
            best_val = refined_val
    
    return [int(x) for x in best_c]
