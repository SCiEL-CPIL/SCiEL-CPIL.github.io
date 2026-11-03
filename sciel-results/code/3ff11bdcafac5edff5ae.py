"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _gram_variance_energy(pts):
    """
    Compute energy based on variance of pairwise angular distances via Gram matrix.
    For uniform points on sphere, off-diagonal Gram entries should be ~ -1/(N-1).
    We minimize deviation from this target, plus a spread term.
    """
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_n = pts / norms
    
    # Gram matrix
    G = pts_n @ pts_n.T
    np.fill_diagonal(G, 0.0)
    
    # Target: for uniform distribution, E[dot] = -1/(N-1)
    target = -1.0 / (n - 1)
    
    # Variance of off-diagonal entries
    off_diag = G[np.triu_indices(n, k=1)]
    mean_val = np.mean(off_diag)
    var_val = np.var(off_diag)
    
    # Energy: penalize variance + deviation of mean from target
    # Also add a term to ensure points are well-spread (not clustered)
    energy = var_val + 10.0 * (mean_val - target) ** 2
    
    # Gradient computation
    # d/dP_ij [sum_{a<b} (G_ab - target)^2 + lambda*(mean-target)^2]
    # G_ab = sum_k P_ak * P_bk
    # dG_ab/dP_ij = delta_ia * P_bj + delta_ib * P_aj
    
    grad = np.zeros_like(pts_n)
    
    # For each pair (a,b), a < b:
    # contribution to grad[a] += (G_ab - target) * P_b
    # contribution to grad[b] += (G_ab - target) * P_a
    # Plus mean term: d(mean)/dP_ij = (1/(n*(n-1)/2)) * (P_bj * delta_ia + P_aj * delta_ib) summed over pairs
    
    # Simplified: compute gradient analytically
    # Let's use a more direct approach with matrix operations
    
    # Energy = sum_{a<b} (G_ab - target)^2
    # grad_i = 2 * sum_{j != i} (G_ij - target) * P_j
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            grad[i] += 2.0 * (G[i, j] - target) * pts_n[j]
    
    # Add gradient for mean term: lambda * 2 * (mean - target) * d(mean)/dP
    # d(mean)/dP_i = (1/M) * sum_{j != i} P_j where M = n*(n-1)/2
    M = n * (n - 1) / 2
    mean_grad = (1.0 / M) * np.sum(pts_n, axis=0)  # This is actually sum over all, need to be careful
    # Actually for mean of off-diagonal: d(mean)/dP_ik = (1/M) * sum_{j != i} P_jk
    for i in range(n):
        # Sum of all other points
        other_sum = np.sum(pts_n, axis=0) - pts_n[i]
        grad[i] += 10.0 * 2.0 * (mean_val - target) * other_sum / M
    
    # Project to tangent space
    for i in range(n):
        dot_val = np.dot(grad[i], pts_n[i])
        grad[i] = grad[i] - dot_val * pts_n[i]
    
    return energy, grad


def _covering_radius_fast(pts, n_test=50000):
    """
    Fast estimate of covering radius using a grid of test points.
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
    Optimize a single configuration using L-BFGS-B with analytical gradient.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        pts_reshaped = x.reshape(N, 3)
        e, g = _gram_variance_energy(pts_reshaped)
        return e, g.flatten()
    
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B with analytical gradient
    result = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-12, 'gtol': 1e-10}
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


def _icosahedron_based_init(n):
    """Generate initial points based on icosahedron symmetry with perturbations."""
    # Start with Fibonacci spiral and add structured perturbations
    base = _fibonacci_spiral(n)
    rng = np.random.RandomState(123)
    
    # Add small random rotations to break symmetry
    for _ in range(5):
        # Random rotation
        v = rng.randn(3)
        v /= np.linalg.norm(v)
        angle = 0.01 * rng.randn()
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        base = base @ R.T
    
    return base


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Uses Gram matrix variance minimization with multiple restarts.
    """
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Standard Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with different offset
    initial_configs.append(_fibonacci_spiral(N, offset=0.3))
    initial_configs.append(_fibonacci_spiral(N, offset=0.7))
    
    # 3. Icosahedron-based initialization
    initial_configs.append(_icosahedron_based_init(N))
    
    # 4. Random configurations
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        rng.seed(seed)
        random_pts = rng.randn(N, 3)
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / norms
        initial_configs.append(random_pts)
    
    # Optimize each configuration and keep the best
    best_pts = initial_configs[0]
    best_cr = _covering_radius_fast(best_pts)
    
    for i, config in enumerate(initial_configs):
        elapsed_so_far = time.time() - start_time
        remaining = time_budget - elapsed_so_far - 5
        if remaining < 2.0:
            break
        
        # Optimize this configuration
        max_opt_time = min(12.0, remaining)
        optimized = _optimize_single(config, max_time=max_opt_time)
        
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
