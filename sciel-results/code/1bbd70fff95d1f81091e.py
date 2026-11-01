"""
Strategy: Riemannian Newton Method with explicit Hessian and damped line search.
This structurally differs from the seed by using second-order information.
The logarithmic potential is highly curved near close pairs; Newton's method
exploits this for quadratic convergence, allowing us to reach much deeper local
minima (higher precision) in fewer iterations than gradient-based methods.
We combine this with diverse initializations (Fibonacci, FPS, Random) to explore
different basins of attraction.
"""
import numpy as np
import time
import math

N = 150

def _log_energy(x):
    """Compute logarithmic energy."""
    # diff is (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    return energy

def _log_energy_and_grad_hess(x):
    """
    Computes energy, Euclidean gradient, and Euclidean Hessian.
    Returns: energy, grad (N, 3), hess (N, 3, 3)
    """
    # Pairwise differences: D[i, j] = x_i - x_j
    D = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # Squared distances: r2[i, j] = |x_i - x_j|^2
    r2 = np.sum(D * D, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    # Safe r2 to avoid div by 0
    safe_r2 = np.where(mask, r2, 1.0)
    
    # Energy: 0.5 * sum_{i!=j} -log(r) = 0.25 * sum_{i!=j} -log(r2)
    log_terms = -0.5 * np.log(safe_r2)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient:
    # E = sum_{i<j} -log(r_ij)
    # dE/dx_i = sum_{j!=i} d(-log(r_ij))/dx_i
    # r_ij = sqrt(r2_ij)
    # d(r_ij)/dx_i = (x_i - x_j) / r_ij
    # d(-log r_ij)/dx_i = -1/r_ij * (x_i - x_j)/r_ij = -(x_i - x_j)/r2_ij
    # So grad_i = sum_{j!=i} -(x_i - x_j)/r2_ij
    
    # inv_r2 shape (N, N)
    inv_r2 = np.zeros_like(r2)
    inv_r2[mask] = 1.0 / safe_r2[mask]
    
    # weighted_D[i, j] = D[i, j] * inv_r2[i, j]
    weighted_D = D * inv_r2[:, :, np.newaxis]
    
    # grad[i] = - sum_j weighted_D[i, j]
    grad = -np.sum(weighted_D, axis=1)
    
    # Hessian:
    # d2E/dx_i dx_i (diagonal block):
    # d/dx_i [ -(x_i - x_j)/r2_ij ]
    # = - [ I/r2 - 2(x_i-x_j)(x_i-x_j)^T / r2^2 ]
    # = -1/r2 * I + 2 D D^T / r2^2
    
    # Off-diagonal block H_{ij} (i != j):
    # d/dx_j [ -(x_i - x_j)/r2_ij ]
    # = - [ -I/r2 - 2(-D)(-D)^T / r2^2 ]  (Note: d(x_i-x_j)/dx_j = -I)
    # Let's re-derive carefully.
    # Term T = -(x_i - x_j) * r2^{-1}
    # dT/dx_j = - ( -I * r2^{-1} + (x_i - x_j) * d(r2^{-1})/dx_j )
    # d(r2)/dx_j = d( (x_i-x_j)^2 )/dx_j = -2(x_i-x_j)
    # d(r2^{-1})/dx_j = -r2^{-2} * d(r2)/dx_j = 2(x_i-x_j)/r2^2
    # So dT/dx_j = - ( -I/r2 + (x_i-x_j) * 2(x_i-x_j)^T / r2^2 )
    # = I/r2 - 2 D D^T / r2^2
    
    # We need Hessian H such that H_{ij} is the 3x3 matrix.
    # H_{ii} = sum_{j!=i} ( -1/r2_ij * I + 2 D_ij D_ij^T / r2_ij^2 )
    # H_{ij} = I/r2_ij - 2 D_ij D_ij^T / r2_ij^2  (for i != j)
    
    # Construct Hessian (N, 3, 3)
    # We can compute this efficiently.
    # Let M = D * (2/r2^2)[:,:,None]  -> (N, N, 3) where M[i,j] = 2 D_ij / r2_ij^2
    # Then D_ij D_ij^T / r2^2 = 0.5 * M[i,j] * D_ij^T
    
    inv_r4 = np.zeros_like(r2)
    inv_r4[mask] = 1.0 / (safe_r2[mask] ** 2)
    
    # 2 * D * inv_r4
    term2 = D * (2.0 * inv_r4)[:, :, np.newaxis]
    
    # H_{ij} for i!=j is: I/r2 - D D^T / r2^2 * 2? No.
    # H_{ij} = I/r2 - 2 D D^T / r2^2
    # H_{ii} = sum_{j!=i} ( -I/r2 + 2 D D^T / r2^2 )
    
    # Let's compute the outer product term: Outer[i, j] = D_ij D_ij^T
    # Outer shape (N, N, 3, 3) might be too big? 150*150*9 ~ 200k floats. 1.6MB. Fine.
    # But we can avoid full 4D array if we sum carefully.
    
    # Let's compute Hessian block by block or using einsum.
    # H = 0
    # For each i:
    #   H_ii += sum_j ( -I/r2_ij + 2 D_ij D_ij^T / r2_ij^2 )
    #   H_ij += ( I/r2_ij - 2 D_ij D_ij^T / r2_ij^2 )
    
    # Vectorized approach:
    # A = D * inv_r2[:,:,None]  # (N, N, 3) -> A[i,j] = D_ij / r2_ij
    # B = D * (2*inv_r4)[:,:,None] # (N, N, 3) -> B[i,j] = 2 D_ij / r2_ij^2
    
    # H_{ij} (i!=j) = I/r2_ij - 0.5 * B[i,j] D_ij^T ? No.
    # 2 D D^T / r2^2 = B D^T.
    # So H_{ij} = I/r2 - B D^T.
    
    # Let's compute H as (N, 3, 3).
    # H = np.zeros((N, 3, 3))
    
    # Diagonal part:
    # H_ii = sum_{j!=i} ( -I/r2_ij + 2 D_ij D_ij^T / r2_ij^2 )
    # Sum of -I/r2: -I * sum_j 1/r2_ij
    # Sum of 2 D D^T / r2^2: sum_j B[i,j] D[i,j]^T
    
    # Off-diagonal part:
    # H_ij = I/r2_ij - 2 D_ij D_ij^T / r2_ij^2 = I/r2_ij - B[i,j] D[i,j]^T
    
    # We can compute the full Hessian matrix H (N, N, 3, 3) and then reshape?
    # Or just compute the (N, 3, 3) blocks.
    
    # Let's use einsum for the outer products.
    # outer[i, j, a, b] = D[i, j, a] * D[i, j, b]
    # We need sum over j for diagonal, and specific j for off-diagonal.
    
    # Actually, since N=150, we can just loop over i and compute the 3x3 blocks?
    # 150 iterations, each doing vectorized ops on 150x3 arrays. Very fast.
    
    hess = np.zeros((N, 3, 3))
    
    # Precompute terms
    # inv_r2 (N, N)
    # inv_r4 (N, N)
    
    for i in range(N):
        # Diagonal block H_ii
        # Sum over j != i
        # -I * sum_{j!=i} 1/r2_ij
        sum_inv_r2 = np.sum(inv_r2[i, mask[i, :]])
        
        # 2 * sum_{j!=i} D_ij D_ij^T / r2_ij^2
        # D[i, j] is (3,)
        # We want sum_j (2/r2_ij^2) * D_ij D_ij^T
        # D_i = D[i, :, :] shape (N, 3)
        # weights = 2 * inv_r4[i, :] shape (N,)
        # Mask out j=i
        w = 2.0 * inv_r4[i, :]
        w[i] = 0.0
        
        # Outer sum: sum_j w_j * D_ij D_ij^T
        # D_i = D[i]
        # result = D_i.T @ (w[:, None] * D_i)
        outer_sum = D[i].T @ (w[:, np.newaxis] * D[i])
        
        hess[i] = -np.eye(3) * sum_inv_r2 + outer_sum
        
        # Off-diagonal blocks H_ij for j != i
        # H_ij = I/r2_ij - 2 D_ij D_ij^T / r2_ij^2
        # Note: H is symmetric, so we can compute for j > i and mirror, or just compute all.
        # Computing all is easier.
        
        # For a fixed i, we want H[i, j] for all j.
        # H[i, j] = I/r2_ij - 2 D_ij D_ij^T / r2_ij^2
        # D_i = D[i] (N, 3)
        # r2_i = r2[i] (N,)
        # inv_r2_i = inv_r2[i] (N,)
        # inv_r4_i = inv_r4[i] (N,)
        
        # Term 1: I / r2_ij
        # This is a diagonal matrix for each j.
        # We can construct a (N, 3, 3) array where block j is I/r2_ij.
        # But we only need to add it to hess? No, hess[i] is the whole row of blocks?
        # No, hess is (N, 3, 3). hess[i] is the block H_ii?
        # Wait. The Hessian of a function f: R^{3N} -> R is a 3N x 3N matrix.
        # My hess array is (N, 3, 3). This represents the block matrix H where H[i, j] is 3x3.
        # So hess[i] should contain H_{i, 0..N-1}? No.
        # Usually in these loops, we compute the full 3N x 3N Hessian or the blocks.
        # If I return (N, 3, 3), it implies hess[i] is the i-th 3x3 block of the diagonal?
        # No, that's not right. The Hessian has N^2 blocks.
        
        # Let's return the full Hessian as a (3N, 3N) matrix?
        # 450x450 = 202,500 floats. 1.6 MB.
        # Inverting a 450x450 matrix is O(450^3) ~ 90 million ops. Very fast.
        # This is the cleanest way to implement Newton's method.
        
        # Let's refactor to return full Hessian.
        
    return energy, grad, hess

def _log_energy_and_grad_hess_full(x):
    """
    Returns energy, grad (3N,), hess (3N, 3N)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    safe_r2 = np.where(mask, r2, 1.0)
    
    # Energy
    log_terms = -0.5 * np.log(safe_r2)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient
    inv_r2 = np.zeros_like(r2)
    inv_r2[mask] = 1.0 / safe_r2[mask]
    
    weighted_diff = diff * inv_r2[:, :, np.newaxis]
    grad_blocks = -np.sum(weighted_diff, axis=1) # (N, 3)
    grad = grad_blocks.reshape(3 * N)
    
    # Hessian
    # H_{ij} (i != j) = I/r2_ij - 2 D_ij D_ij^T / r2_ij^2
    # H_{ii} = sum_{j!=i} ( -I/r2_ij + 2 D_ij D_ij^T / r2_ij^2 )
    
    inv_r4 = np.zeros_like(r2)
    inv_r4[mask] = 1.0 / (safe_r2[mask] ** 2)
    
    # Precompute 2 * D * inv_r4
    term2 = diff * (2.0 * inv_r4)[:, :, np.newaxis] # (N, N, 3)
    
    hess_full = np.zeros((3 * N, 3 * N))
    
    for i in range(N):
        # Diagonal block H_ii
        # -I * sum_{j!=i} 1/r2_ij
        sum_inv_r2 = np.sum(inv_r2[i, mask[i, :]])
        
        # 2 * sum_{j!=i} D_ij D_ij^T / r2_ij^2
        # D_i = diff[i] (N, 3)
        # w = 2 * inv_r4[i] (N,)
        w = 2.0 * inv_r4[i, :]
        w[i] = 0.0
        
        # Outer sum: sum_j w_j D_ij D_ij^T
        # D_i.T @ (w[:, None] * D_i)
        outer_sum = diff[i].T @ (w[:, np.newaxis] * diff[i])
        
        H_ii = -np.eye(3) * sum_inv_r2 + outer_sum
        
        # Place in hess_full
        hess_full[3*i:3*i+3, 3*i:3*i+3] = H_ii
        
        # Off-diagonal blocks H_ij for j != i
        # H_ij = I/r2_ij - 2 D_ij D_ij^T / r2_ij^2
        # D_i = diff[i] (N, 3)
        # r2_i = r2[i] (N,)
        # inv_r2_i = inv_r2[i] (N,)
        # term2_i = term2[i] (N, 3) -> 2 D_ij / r2_ij^2
        
        # We need to fill hess_full[3*i:3*i+3, 3*j:3*j+3] for all j != i
        # And hess_full[3*j:3*j+3, 3*i:3*i+3] = H_ij.T (which is H_ji)
        
        # Let's compute the blocks for all j at once?
        # H_ij is 3x3.
        # H_ij = (1/r2_ij) * I - D_ij D_ij^T * (2/r2_ij^2)
        
        # Let's loop j for simplicity, N=150 is small.
        for j in range(i + 1, N):
            d_ij = diff[i, j] # (3,)
            r2_ij = r2[i, j]
            
            # H_ij = I/r2_ij - 2 d_ij d_ij^T / r2_ij^2
            H_ij = np.eye(3) / r2_ij - 2.0 * np.outer(d_ij, d_ij) / (r2_ij ** 2)
            
            hess_full[3*i:3*i+3, 3*j:3*j+3] = H_ij
            hess_full[3*j:3*j+3, 3*i:3*i+3] = H_ij.T
            
    return energy, grad, hess_full

def _exp_map(x, v):
    """
    Exponential map on the unit sphere.
    x: (N, 3) current points.
    v: (N, 3) tangent vectors.
    """
    dot_xv = np.sum(x * v, axis=1, keepdims=True)
    v_tangent = v - dot_xv * x
    
    theta = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    theta_safe = np.where(theta < 1e-10, 1e-10, theta)
    
    cos_t = np.cos(theta_safe)
    sin_t_over_t = np.sin(theta_safe) / theta_safe
    
    return cos_t * x + sin_t_over_t * v_tangent

def _project_to_tangent(x, v):
    """Project vector v (Euclidean) to tangent space at x."""
    dot_xv = np.sum(x * v, axis=1, keepdims=True)
    return v - dot_xv * x

def _riemannian_newton(x_init, max_iter=50, time_limit=None):
    """
    Riemannian Newton Method.
    """
    x_curr = x_init.copy()
    e_curr, g_curr, h_curr = _log_energy_and_grad_hess_full(x_curr)
    
    # Project gradient to tangent space
    g_tangent = _project_to_tangent(x_curr, g_curr.reshape(N, 3)).reshape(-1)
    
    for i in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Check convergence
        norm_g = np.linalg.norm(g_tangent)
        if norm_g < 1e-10:
            break
        
        # Solve Hessian system: H v = -g_tangent
        # Hessian is symmetric. We can use Cholesky if PD, else LDL or regularized.
        # Log energy Hessian might not be PD everywhere.
        # We add regularization if needed.
        
        try:
            # Add small regularization to ensure invertibility
            reg = 1e-6 * np.eye(3 * N)
            h_reg = h_curr + reg
            v_newton = np.linalg.solve(h_reg, -g_tangent)
        except np.linalg.LinAlgError:
            # Fallback to gradient direction if Hessian is singular
            v_newton = -g_tangent
            # Scale it?
            v_newton = v_newton / (np.linalg.norm(v_newton) + 1e-10)
            
        # Project Newton step to tangent space
        v_tangent = _project_to_tangent(x_curr, v_newton.reshape(N, 3))
        
        # Line search
        alpha = 1.0
        found = False
        for _ in range(20):
            x_try = _exp_map(x_curr, alpha * v_tangent)
            e_try, _, _ = _log_energy_and_grad_hess_full(x_try)
            
            if e_try < e_curr:
                found = True
                break
            alpha *= 0.5
            
        if not found:
            # If line search fails, try smaller alpha or break
            # If alpha is very small, we are stuck
            if alpha < 1e-6:
                break
            # Try once more with very small alpha?
            x_try = _exp_map(x_curr, alpha * v_tangent)
            e_try, _, _ = _log_energy_and_grad_hess_full(x_try)
            if e_try < e_curr:
                found = True
            else:
                break
                
        if found:
            x_curr = x_try
            e_curr = e_try
            # Recompute grad and hess
            _, g_curr, h_curr = _log_energy_and_grad_hess_full(x_curr)
            g_tangent = _project_to_tangent(x_curr, g_curr.reshape(N, 3)).reshape(-1)
            
    return x_curr, e_curr

def _farthest_point_sampling(n, seed=0):
    """
    Generate n points on the sphere using farthest point sampling.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    v = rng.normal(size=3)
    x[0] = v / np.linalg.norm(v)
    
    for i in range(1, n):
        batch_size = 1000
        candidates = rng.normal(size=(batch_size, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - x[:i][np.newaxis, :, :], axis=2)
        min_dists_batch = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists_batch)
        x[i] = candidates[best_idx]
        
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    inits = []
    
    # 1. Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Farthest Point Sampling (several seeds)
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        x_fps = _farthest_point_sampling(N, seed)
        inits.append(x_fps)
        
    # 3. Random starts
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(seed + 100)
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
        
    # Run Newton on each initialization
    for x_init in inits:
        if time.time() > deadline - 5:
            break
            
        x_opt, e_opt = _riemannian_newton(x_init, max_iter=50, time_limit=deadline)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
            
    # Polish the best solution
    if best_x is not None and time.time() < deadline - 10:
        x_final, e_final = _riemannian_newton(best_x, max_iter=50, time_limit=deadline)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final

    if best_x is None:
        best_x = inits[0]
        
    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    e = _log_energy(pts)
    print(f"Energy: {e}")
