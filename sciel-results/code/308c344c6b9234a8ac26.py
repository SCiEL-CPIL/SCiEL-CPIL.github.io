import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 24
    # Grid for sup norm estimation
    # Use a dense grid. 10000 points is usually sufficient for N=24 to distinguish candidates.
    num_points = 10000
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Precompute powers of z for all k
    # Z shape: (num_points, N)
    Z = np.exp(1j * theta[:, None] * np.arange(N)[None, :])
    
    def sup_norm(coeffs):
        # coeffs: array of shape (N,) with values +/- 1
        # Evaluate P(z) at all theta
        vals = np.sum(coeffs[None, :] * Z, axis=1)
        return np.max(np.abs(vals))
    
    def refine_sup_norm(coeffs):
        # Optional: refine the max found by the grid
        # Find index of max
        vals = np.sum(coeffs[None, :] * Z, axis=1)
        abs_vals = np.abs(vals)
        max_idx = np.argmax(abs_vals)
        # The peak is near theta[max_idx]
        # Do a few iterations of golden section or fine grid around it
        # For simplicity and speed, a fine grid of 1000 points around the max is often enough
        # But since we just need to compare candidates, the coarse grid might be noisy.
        # Let's stick to the coarse grid for search, and maybe do a final check?
        # Actually, for N=24, the function is smooth. The grid error is small.
        return np.max(abs_vals)
        
    def evaluate(coeffs):
        return sup_norm(coeffs)

    # Initial candidates
    candidates = []
    
    # 1. GRS Polynomial
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    gsr = np.array(p[:N])
    candidates.append(gsr)
    
    # 2. Random candidates
    for _ in range(20):
        if time.time() > deadline:
            break
        r = np.array([1 if np.random.rand() < 0.5 else -1 for _ in range(N)])
        candidates.append(r)

    best_score = float('inf')
    best_coeffs = None
    
    # Local Search (Hill Climbing)
    # We will iterate through candidates and improve them.
    # To avoid getting stuck, we can do multiple passes or random restarts.
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        current = cand.copy()
        current_score = evaluate(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each bit
            # Randomize order to avoid bias
            indices = np.random.permutation(N)
            for k in indices:
                if time.time() > deadline:
                    break
                # Flip bit k
                old_val = current[k]
                current[k] = -old_val
                new_score = evaluate(current)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[k] = old_val
            
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current.copy()
            
        # If we have time, do a random restart from the best found so far?
        # Or just continue with the next initial candidate.
        
    # If best_coeffs is None (shouldn't happen), return GRS
    if best_coeffs is None:
        return [int(x) for x in gsr]
        
    return [int(x) for x in best_coeffs]
