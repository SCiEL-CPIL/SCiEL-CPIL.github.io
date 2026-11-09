"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    N = 24
    
    # Precompute evaluation grid for fast sup-norm estimation
    # Use a sufficiently dense grid to distinguish near-optimal candidates
    num_points = 20000
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Z[k, j] = exp(1j * theta_j * k) for k=0..N-1, j=0..num_points-1
    # Shape: (N, num_points)
    k_indices = np.arange(N)
    Z = np.exp(1j * np.outer(k_indices, theta))  # (N, num_points)
    
    def sup_norm(coeffs):
        """Compute sup norm of P(z) on the unit circle using dense grid."""
        # coeffs: shape (N,)
        # P(z_j) = sum_k coeffs[k] * z_j^k = coeffs @ Z[:, j]
        vals = Z.T @ coeffs  # shape (num_points,)
        return np.max(np.abs(vals))
    
    def refine_sup_norm(coeffs):
        """Refine the sup norm by doing a local search around the peak."""
        vals = Z.T @ coeffs
        abs_vals = np.abs(vals)
        max_idx = np.argmax(abs_vals)
        max_theta = theta[max_idx]
        
        # Golden section refinement around the peak
        # The function |P(z)| is smooth near the peak
        # Do a few iterations of fine-grained search
        best_theta = max_theta
        best_val = abs_vals[max_idx]
        
        # Fine grid around the peak
        delta_theta = 2 * np.pi / num_points
        for scale in [1.0, 0.1, 0.01, 0.001]:
            t_range = np.linspace(best_theta - scale * delta_theta, 
                                  best_theta + scale * delta_theta, 
                                  100)
            z_fine = np.exp(1j * t_range)
            # Evaluate P at these points
            # P(z) = sum_k coeffs[k] * z^k
            powers = z_fine[:, None] ** k_indices[None, :]  # (100, N)
            vals_fine = powers @ coeffs  # (100,)
            abs_fine = np.abs(vals_fine)
            local_max_idx = np.argmax(abs_fine)
            if abs_fine[local_max_idx] > best_val:
                best_val = abs_fine[local_max_idx]
                best_theta = t_range[local_max_idx]
        
        return best_val
    
    def evaluate(coeffs):
        """Evaluate the score: sup_norm / sqrt(N)."""
        return sup_norm(coeffs) / np.sqrt(N)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro polynomial
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    gsr = np.array(p[:N])
    candidates.append(gsr)
    
    # 2. The hardcoded candidate from inspiration 2 (claimed optimal)
    hardcode1 = np.array([
        1, 1, -1, 1, -1, 1, 1, -1, 
        1, -1, 1, -1, -1, 1, -1, -1, 
        1, 1, -1, -1, 1, -1, 1, 1
    ])
    candidates.append(hardcode1)
    
    # 3. The hardcoded candidate from inspiration 3
    hardcode2 = np.array([
        1, 1, 1, 1, -1, 1, -1, -1, 
        1, -1, 1, -1, -1, 1, 1, -1, 
        1, -1, -1, 1, -1, 1, -1, -1
    ])
    candidates.append(hardcode2)
    
    # 4. Random candidates
    np.random.seed(42)
    for _ in range(30):
        r = np.array([1 if np.random.rand() < 0.5 else -1 for _ in range(N)])
        candidates.append(r)
    
    best_score = float('inf')
    best_coeffs = None
    
    # Local Search (Hill Climbing) on each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
            
        current = cand.copy().astype(float)
        current_score = evaluate(current)
        
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            # Try flipping each bit in random order
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
    
    # If we have time, do more random restarts from the best found
    if best_coeffs is not None and time.time() < deadline - 20.0:
        for restart in range(20):
            if time.time() > deadline:
                break
            # Start from a perturbation of the best
            current = best_coeffs.copy().astype(float)
            # Flip 2-4 random bits
            num_flips = np.random.randint(2, 5)
            flip_indices = np.random.choice(N, num_flips, replace=False)
            for k in flip_indices:
                current[k] = -current[k]
            
            current_score = evaluate(current)
            improved = True
            max_iterations = 30
            iteration = 0
            while improved and time.time() < deadline and iteration < max_iterations:
                improved = False
                iteration += 1
                indices = np.random.permutation(N)
                for k in indices:
                    if time.time() > deadline:
                        break
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
    
    # If best_coeffs is None (shouldn't happen), return GRS
    if best_coeffs is None:
        return [int(x) for x in gsr]
        
    return [int(x) for x in best_coeffs]
# EVOLVE-BLOCK-END
