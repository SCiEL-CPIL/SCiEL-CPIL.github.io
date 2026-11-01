"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150
TIME_LIMIT = 140.0  # Leave 10s safety margin for 150s limit

# EVOLVE-BLOCK-START

def _log_energy_and_hessian(x):
    """
    Compute the logarithmic energy and the Hessian matrix on the sphere.
    Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient: grad_i E = sum_{j!=i} (x_i - x_j) / d_ij^2
    Hessian (Euclidean, before projection):
    H_ii = sum_{j!=i} [ I/d_ij^2 - (x_i-x_j)(x_i-x_j)^T / d_ij^4 ]
    H_ij = (x_i-x_j)(x_i-x_j)^T / d_ij^4  (for i!=j)
    
    Note: We compute the Euclidean Hessian H. 
    The Riemannian Hessian is (I - x x^T) H (I - x x^T) + x (x^T H x) ... 
    Actually, for constrained minimization on the sphere, we can just use the 
    Euclidean Hessian in the Newton step but project the step back.
    However, a more stable approach for the sphere is to solve for the step 
    in the tangent space.
    
    Let's use the standard Euclidean Newton step: x_new = x - H^{-1} g.
    Then normalize. This works well if we are close to the manifold.
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    
    # Compute pairwise distances squared
    # d2[i,j] = ||x_i - x_j||^2
    # d2 = (x_i^2 + x_j^2 - 2 x_i . x_j)
    # Since ||x_i||=1, d2 = 2 - 2 * G, where G is Gram matrix
    G = x @ x.T
    np.fill_diagonal(G, 0.0)
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 1.0) # Avoid division by zero, value doesn't matter for i=j in sums
    
    # Energy
    # E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # log(d) = 0.5 * log(d2)
    # E = -0.5 * sum_{i!=j} 0.5 * log(d2) = -0.25 * sum_{i!=j} log(d2)
    # Wait, sum_{i<j} log(1/d) = - sum_{i<j} log(d).
    # sum_{i<j} log(d) = 0.5 * sum_{i!=j} log(d).
    # So E = -0.5 * sum_{i!=j} log(d_ij) = -0.25 * sum_{i!=j} log(d2_ij).
    
    # For gradient and Hessian, we need individual terms.
    
    # Gradient: g_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Vectorized:
    # diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # g_i = sum_j (x_i - x_j) / d2_ij
    # We can compute this as: sum_j diff[i,j] / d2[i,j]
    # Note d2[i,j] = d2[j,i]
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    
    # Hessian:
    # H_ii = sum_{j!=i} [ I/d2_ij - diff_ij diff_ij^T / d2_ij^2 ]
    # H_ij = diff_ij diff_ij^T / d2_ij^2 (for i!=j)
    
    # This is a large matrix (150*3 x 150*3). 450x450.
    # We can construct it block by block or use einsum.
    # Let's construct the full Hessian.
    # H is symmetric.
    
    # Term 1: I / d2_ij
    # Term 2: - diff diff^T / d2_ij^2
    
    # Let's build H as a 3D array of 3x3 blocks? Or just a big matrix.
    # A big matrix is easier to invert with np.linalg.solve.
    
    H = np.zeros((n*3, n*3), dtype=np.float64)
    
    # Iterate over i,j to fill H. This is O(N^2 * 9) operations. 150^2 * 9 ~ 200k. Fast.
    # But Python loops are slow. Let's try to vectorize or use efficient loops.
    # With N=150, a double loop in Python might take ~0.1-0.5s. 
    # We call this in Newton step. If we do 100 iterations, that's 10-50s.
    # We need to be faster.
    
    # Alternative: Use the fact that H_ij = u_ij u_ij^T / d2_ij^2 where u_ij = x_i - x_j.
    # And H_ii = sum_{j!=i} (I/d2_ij - u_ij u_ij^T / d2_ij^2).
    
    # Let's precompute u_ij u_ij^T / d2_ij^2.
    # u_ij = diff[i,j]
    # outer = np.einsum('ij,ik->ijk', diff, diff) -> shape (N, N, 3, 3)
    # This is 150*150*9 = 202,500 floats. Small.
    
    outer = np.einsum('ij,ik->ijk', diff, diff) # diff is (N,N,3)
    # outer[i,j,k,l] = diff[i,j,k] * diff[i,j,l]
    
    # d2 is (N,N)
    # d2_inv_sq = 1/d2^2
    d2_inv_sq = 1.0 / (d2 * d2)
    np.fill_diagonal(d2_inv_sq, 0.0)
    
    # H_ij (i!=j) = outer[i,j] * d2_inv_sq[i,j]
    # H_ii = sum_{j!=i} (I * inv_d2[i,j] - outer[i,j] * d2_inv_sq[i,j])
    
    # Let's build the Hessian matrix H_mat (N*3, N*3)
    # We can use scipy.sparse? No, dense is fine for 450x450.
    
    # Initialize H_mat with zeros
    H_mat = np.zeros((n*3, n*3))
    
    # Fill off-diagonal blocks
    # For i != j:
    # H_mat[3*i:3*i+3, 3*j:3*j+3] = outer[i,j] * d2_inv_sq[i,j]
    
    # This is hard to vectorize directly into a 2D array without loops or advanced indexing.
    # Given N=150, let's just use a loop over i,j.
    # 150*150 = 22500 iterations. In Python, this is ~0.05s. Acceptable.
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = d2_inv_sq[i, j]
            block = outer[i, j] * val
            H_mat[3*i:3*i+3, 3*j:3*j+3] = block
            
            # Symmetry: H_mat[3*j:3*j+3, 3*i:3*i+3] = block.T
            # Since outer is symmetric in i,j? No, diff[i,j] = -diff[j,i], so outer is same.
            # d2 is same. So block is same.
            # But H is symmetric, so H_ij = H_ji.
            # block is 3x3. H_ij = block. H_ji = block^T?
            # diff[i,j] = x_i - x_j. diff[j,i] = x_j - x_i = -diff[i,j].
            # outer[i,j] = diff[i,j] diff[i,j]^T.
            # outer[j,i] = diff[j,i] diff[j,i]^T = (-diff[i,j]) (-diff[i,j])^T = same.
            # So block is symmetric.
            H_mat[3*j:3*j+3, 3*i:3*i+3] = block.T # Just in case, though it's symmetric
            
    # Fill diagonal blocks
    # H_ii = sum_{j!=i} (I/d2_ij - outer[i,j] * d2_inv_sq[i,j])
    for i in range(n):
        # Sum over j != i
        # We can sum the vector of blocks
        # Sum of I/d2_ij is I * sum(1/d2_ij)
        sum_inv_d2 = np.sum(inv_d2[i, :]) # sum over j
        # Sum of outer[i,j] * d2_inv_sq[i,j]
        # outer[i, :] is (N, 3, 3). d2_inv_sq[i, :] is (N,).
        sum_outer = np.einsum('j, jkl -> kl', d2_inv_sq[i, :], outer[i, :, :, :])
        
        H_ii = sum_inv_d2 * np.eye(3) - sum_outer
        H_mat[3*i:3*i+3, 3*i:3*i+3] = H_ii
        
    return -0.25 * np.sum(np.log(d2)), grad, H_mat

def _newton_polish(x, max_iters=20, tol=1e-10):
    """
    Polish the configuration using Newton's method on the sphere.
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    for _ in range(max_iters):
        E, g, H = _log_energy_and_hessian(x)
        
        # Check convergence
        # Project gradient to tangent space? 
        # For the step, we solve H dx = -g.
        # Then x_new = x + dx. Normalize.
        
        try:
            dx = np.linalg.solve(H, -g.reshape(-1))
        except np.linalg.LinAlgError:
            # If H is singular, use pseudo-inverse or fall back to gradient descent
            dx = -0.01 * g.reshape(-1)
            
        dx = dx.reshape(n, 3)
        
        # Check step size
        step_norm = np.linalg.norm(dx)
        if step_norm < tol:
            break
            
        # Damping if step is too large
        if step_norm > 0.1:
            dx = dx * (0.1 / step_norm)
            
        x_new = x + dx
        x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
        
        # Check if energy decreased
        E_new, _, _ = _log_energy_and_hessian(x_new)
        if E_new < E:
            x = x_new
        else:
            # Line search
            alpha = 0.5
            for _ in range(10):
                x_try = x + alpha * dx
                x_try = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
                E_try, _, _ = _log_energy_and_hessian(x_try)
                if E_try < E:
                    x = x_try
                    break
                alpha *= 0.5
            else:
                # Failed to decrease, break
                break
                
    return x

def _random_sphere_points(n, seed=None):
    rng = np.random.default_rng(seed)
    # Random normal vectors
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _perturb(x, sigma=0.1, seed=None):
    rng = np.random.default_rng(seed)
    # Add Gaussian noise
    noise = rng.normal(scale=sigma, size=x.shape)
    x_pert = x + noise
    x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
    return x_pert

def construct():
    start_time = time.time()
    best_x = None
    best_E = float('inf')
    
    # Initial starts: Fibonacci sphere is a good global structure
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Random starts
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Add Fibonacci start
    candidates.append(x_fib)
    
    # Add some random starts
    for _ in range(5):
        candidates.append(_random_sphere_points(N, seed=rng.integers(10000)))
    
    # Multi-start loop
    for idx, x_start in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Polish the start
        x_polished = _newton_polish(x_start)
        E, _, _ = _log_energy_and_hessian(x_polished)
        
        if E < best_E:
            best_E = E
            best_x = x_polished.copy()
            
        # Basin hopping: Perturb the best solution found so far
        # Only do this if we have time
        if time.time() - start_time < TIME_LIMIT - 10:
            for hop in range(5):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                x_pert = _perturb(best_x, sigma=0.2 + 0.1*hop, seed=rng.integers(100000))
                x_polished_hop = _newton_polish(x_pert)
                E_hop, _, _ = _log_energy_and_hessian(x_polished_hop)
                
                if E_hop < best_E:
                    best_E = E_hop
                    best_x = x_polished_hop.copy()
                    
    # Final polish on the best
    if best_x is not None:
        if time.time() - start_time < TIME_LIMIT - 2:
            best_x = _newton_polish(best_x, max_iters=50)
            
    return best_x

# EVOLVE-BLOCK-END
