"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _spectral_energy_and_grad(pts):
    """
    Compute the spectral energy (sum of squared pairwise dot products) and its gradient.
    Energy: E = sum_{i < j} (p_i . p_j)^2 = 0.5 * (Tr(G^2) - N)
    Gradient w.r.t p_i: g_i = (G p_i) - (p_i . G p_i) p_i
    """
    n = len(pts)
    # Normalize points to ensure they are on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_n = pts / norms
    
    # Compute Gram matrix G = pts_n @ pts_n.T
    G = pts_n @ pts_n.T
    
    # Energy calculation
    # Tr(G^2) = sum(G * G)
    tr_g2 = np.sum(G * G)
    # Diagonal elements are 1, so sum of diagonal is N.
    # Sum_{i != j} (G_ij)^2 = Tr(G^2) - N
    # E = 0.5 * (Tr(G^2) - N)
    energy = 0.5 * (tr_g2 - n)
    
    # Gradient calculation
    # grad_i = 2 * sum_j (G_ij) * p_j  (unprojected)
    # Actually, d/dp_i [ sum_j (p_i . p_j)^2 ] = 2 * sum_j (p_i . p_j) p_j = 2 * G @ p_i
    # We minimize E, so gradient is proportional to G @ p_i.
    # Let's compute G @ pts_n
    Gp = G @ pts_n
    
    # Project gradient to tangent space
    # proj_grad_i = Gp_i - (p_i . Gp_i) * p_i
    # Note: p_i . Gp_i = sum_j (p_i . p_j)^2 = (G^2)_ii
    diag_g2 = np.diag(G * G) # This is (G^2)_ii
    # Wait, (G^2)_ii = sum_k G_ik G_ki = sum_k G_ik^2. Yes.
    
    grad = Gp.copy()
    # Subtract radial component
    # grad_i -= (p_i . Gp_i) * p_i
    # p_i . Gp_i is the i-th element of diag_g2? 
    # Let's verify: p_i . (G p_i) = p_i . sum_j (p_i . p_j) p_j = sum_j (p_i . p_j)^2. Yes.
    for i in range(n):
        grad[i] -= diag_g2[i] * pts_n[i]
        
    return energy, grad


def _covering_radius_fast(pts, n_test=100000):
    """
    Fast estimate of covering radius using a Fibonacci grid of test points.
    """
    M = n_test
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute max dot product for each test point
    dots = test_pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    return np.max(angles)


def _optimize_single(pts_init, max_time=90.0):
    """
    Optimize a single configuration using L-BFGS-B with analytical gradient of spectral energy.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        pts_reshaped = x.reshape(N, 3)
        e, g = _spectral_energy_and_grad(pts_reshaped)
        return e, g.flatten()
    
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B with analytical gradient
    result = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    elapsed = time.time() - start_time
    if elapsed > max_time:
        return pts_init
    
    x_opt = result.x
    pts_opt = x_opt.reshape(N, 3)
    
    # Normalize
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_opt = pts_opt / norms
    
    return pts_opt


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Uses spectral energy minimization with multiple restarts.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Standard Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with different offset
    initial_configs.append(_fibonacci_spiral(N, offset=0.3))
    
    # 3. Random configurations
    rng = np.random.RandomState(42)
    for _ in range(4):
        random_pts = rng.randn(N, 3)
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / norms
        initial_configs.append(random_pts)
        
    # 4. Perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for _ in range(2):
        perturbed = base.copy()
        # Add small noise
        noise = rng.randn(N, 3) * 0.1
        perturbed = perturbed + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        initial_configs.append(perturbed)

    # Optimize each configuration and keep the best
    best_pts = initial_configs[0]
    best_cr = _covering_radius_fast(best_pts)
    
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Optimize this configuration
        remaining_time = time_budget - (time.time() - start_time) - 5
        if remaining_time < 2:
            break
        optimized = _optimize_single(config, max_time=remaining_time)
        
        # Evaluate
        cr = _covering_radius_fast(optimized)
        if cr < best_cr:
            best_cr = cr
            best_pts = optimized
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
