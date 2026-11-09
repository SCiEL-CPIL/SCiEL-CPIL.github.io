import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def compute_energy_and_grad(u):
    """
    u: shape (N, 3) array of unconstrained vectors.
    Returns energy and gradient with respect to u.
    """
    u_norms = np.linalg.norm(u, axis=1, keepdims=True)
    # Avoid division by zero
    u_norms = np.where(u_norms < 1e-12, 1e-12, u_norms)
    x = u / u_norms
    
    # Compute pairwise distances
    # x shape (N, 3)
    # diff shape (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Using fsum for precision as requested? The task says "pair terms summed with math.fsum".
    # However, for optimization, we can use standard sum for speed, 
    # but for the final score, the evaluator uses math.fsum.
    # The objective value returned to the optimizer doesn't need to be exactly the evaluator's value,
    # just monotonic and close. But let's be safe.
    # For speed, we compute the gradient analytically.
    
    # Gradient calculation
    # g_i = -1/||u_i||^2 * sum_{j!=i} ( (x_i . x_j) x_i - x_j ) / d_ij^2
    
    # x_dot_x: N x N matrix of dot products
    x_dot_x = x @ x.T
    
    # d_sq: N x N matrix of squared distances
    d_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero in gradient (should not happen if points are distinct)
    d_sq = np.where(d_sq < 1e-12, 1e-12, d_sq)
    
    # Term: (x_i . x_j) x_i - x_j
    # For each i, sum over j.
    # coeff_ij = x_dot_x[i, j] / d_sq[i, j]
    # grad_term_ij = coeff_ij * x_i - x_j / d_sq[i, j] ? No.
    # Formula: ( (x_i . x_j) x_i - x_j ) / d_sq[i, j]
    
    # Let's compute the sum for each i efficiently.
    # Sum_j [ (x_dot_x[i,j] * x_i - x_j) / d_sq[i,j] ]
    
    # We can rewrite this as:
    # x_i * Sum_j (x_dot_x[i,j] / d_sq[i,j]) - Sum_j (x_j / d_sq[i,j])
    
    # A[i,j] = x_dot_x[i,j] / d_sq[i,j]
    A = x_dot_x / d_sq
    np.fill_diagonal(A, 0)
    
    # B[i] = Sum_j (x_j / d_sq[i,j])
    # We can compute this using matrix multiplication if we had a matrix D_inv where D_inv[i,j] = 1/d_sq[i,j]
    # But we have A and D_inv.
    # D_inv = A / x_dot_x ? No, x_dot_x can be 0 or negative.
    # Let's just compute D_inv separately.
    D_inv = 1.0 / d_sq
    np.fill_diagonal(D_inv, 0)
    
    # Sum_j (x_j / d_sq[i,j]) = D_inv @ x
    S1 = D_inv @ x  # Shape (N, 3)
    
    # Sum_j (x_dot_x[i,j] / d_sq[i,j]) = row sums of A
    S2 = np.sum(A, axis=1)  # Shape (N,)
    
    # Gradient for u_i:
    # g_u_i = -1/||u_i||^2 * ( S2[i] * x_i - S1[i] )
    
    u_norms_sq = np.sum(u**2, axis=1)
    u_norms_sq = np.where(u_norms_sq < 1e-12, 1e-12, u_norms_sq)
    
    # grad_u shape (N, 3)
    # S2[:, np.newaxis] * x - S1
    grad_x_part = S2[:, np.newaxis] * x - S1
    
    grad_u = -grad_x_part / u_norms_sq[:, np.newaxis]
    
    # Calculate Energy for the optimizer
    # E = -sum_{i<j} log(d_ij)
    # We can compute this from d_sq
    d_ij = np.sqrt(d_sq)
    # Mask diagonal
    d_ij = np.where(mask, d_ij, 1.0) # Set diagonal to 1 so log(1)=0
    log_d = np.log(d_ij)
    # Sum over i<j
    # Sum of upper triangle
    energy = -np.sum(log_d[mask])
    
    return energy, grad_u.flatten()

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin
    
    best_energy = float('inf')
    best_points = None
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Create several diverse starting points
    starts = []
    starts.append(init_x.copy())
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        u = rng.normal(size=(N, 3))
        # Normalize to sphere
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        starts.append(u / norms)
    
    # Also add a perturbed Fibonacci
    for _ in range(5):
        u = init_x + rng.normal(scale=0.1, size=(N, 3))
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        starts.append(u / norms)
    
    # Optimization loop
    for k, start_x in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Unconstrained parameter u is just the initial x (normalized)
        u0 = start_x.flatten()
        
        try:
            res = minimize(
                compute_energy_and_grad,
                u0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                u_opt = res.x.reshape(N, 3)
                # Normalize to get final points
                norms = np.linalg.norm(u_opt, axis=1, keepdims=True)
                x_opt = u_opt / norms
                
                # Calculate final energy with high precision (math.fsum)
                # as required by the evaluator interface description
                # The evaluator calculates: sum_{i<j} log(1 / |x_i - x_j|)
                
                # Check for invalid points (too close)
                # The task says: "Points closer than 1e-9 ... make the candidate invalid"
                # We should check this.
                valid = True
                # Compute min distance
                # For 150 points, O(N^2) is fine
                min_dist = float('inf')
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(x_opt[i] - x_opt[j])
                        if d < min_dist:
                            min_dist = d
                if min_dist < 1e-9:
                    valid = False
                
                if valid:
                    # Calculate energy with math.fsum
                    terms = []
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.linalg.norm(x_opt[i] - x_opt[j])
                            terms.append(math.log(1.0 / d))
                    energy = math.fsum(terms)
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_points = x_opt.copy()
                        
        except Exception as e:
            pass # Ignore optimization errors and continue to next restart

    if best_points is None:
        # Fallback to initial Fibonacci
        return init_x
    else:
        return best_points
