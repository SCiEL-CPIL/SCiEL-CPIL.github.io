"""Combined approach: L-BFGS-B on sphere with momentum gradient descent polish and multi-start exploration."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists)


def _energy_and_grad(x_flat):
    """Energy and gradient for L-BFGS-B. x_flat is shape (N*3,)."""
    n = N
    points = x_flat.reshape(n, 3)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy = -sum_{i<j} log(d_ij)
    log_dists_sq = np.log(dists_sq)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_dists_sq[iu])
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    grads = -(points * sum_weights[:, np.newaxis] - weighted_sum)
    
    # Project to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    grad_flat = grads_tangent.flatten()
    
    return energy, grad_flat


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _optimize_lbfgsb(points, maxiter=2000, tol=1e-10):
    """Optimize using L-BFGS-B on the sphere."""
    x_flat = points.flatten()
    
    result = minimize(
        _energy_and_grad,
        x_flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': tol, 'maxfun': 100000}
    )
    
    opt_points = _project_to_sphere(result.x.reshape(N, 3))
    energy = _log_energy(opt_points)
    return opt_points, energy


def _momentum_gradient_descent(pts0, max_time, seed=42):
    """Momentum gradient descent with adaptive step size on the sphere."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts.flatten())
    g = g.reshape(N, 3)
    
    best_pts = pts.copy()
    best_e = e
    
    velocity = np.zeros_like(pts)
    momentum = 0.9
    step_size = 0.005
    
    for iteration in range(50000):
        if time.time() - start_time >= max_time:
            break
        
        # Project gradient to tangent space
        g_dot_n = np.sum(g * pts, axis=1, keepdims=True)
        g_tan = g - g_dot_n * pts
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        # Momentum update
        velocity = momentum * velocity - step_size * g_tan
        
        # Trial step
        new_pts = pts + velocity
        new_pts = _project_to_sphere(new_pts)
        
        # Compute new energy
        new_e, new_g = _energy_and_grad(new_pts.flatten())
        new_g = new_g.reshape(N, 3)
        
        # Backtracking if energy increased
        if new_e > e + 1e-10:
            for _ in range(15):
                step_size *= 0.5
                velocity = momentum * velocity - step_size * g_tan
                new_pts = pts + velocity
                new_pts = _project_to_sphere(new_pts)
                new_e, new_g = _energy_and_grad(new_pts.flatten())
                new_g = new_g.reshape(N, 3)
                if new_e <= e + 1e-10:
                    break
            else:
                step_size *= 0.1
                continue
        
        # Accept step
        pts = new_pts
        e = new_e
        g = new_g
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Gradually increase step size if making progress
        if iteration > 200 and step_size < 0.02:
            step_size *= 1.03
    
    return best_pts, best_e


def _basin_hopping(points, max_time, seed=42):
    """Basin hopping: random perturbations followed by local minimization."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    T = 0.04
    
    while time.time() - start_time < max_time and T > 1e-6:
        # Random perturbation of a subset of points
        n_perturb = rng.randint(1, 8)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        perturbation = rng.randn(N, 3) * T
        perturbation[idx] *= 1.5
        
        new_points = _project_to_sphere(current + perturbation)
        
        # Local minimize from perturbed point
        opt_points, opt_energy = _optimize_lbfgsb(new_points, maxiter=500, tol=1e-8)
        
        delta = opt_energy - current_energy
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-15)):
            current = opt_points
            current_energy = opt_energy
            if opt_energy < best_energy:
                best_energy = opt_energy
                best = opt_points.copy()
        
        T *= 0.93
    
    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_energy = float('inf')
    
    def try_candidate(pts, label=""):
        nonlocal best_points, best_energy
        if pts is None:
            return
        if not np.all(np.isfinite(pts)):
            return
        e = _log_energy(pts)
        if e < best_energy:
            best_energy = e
            best_points = pts.copy()
    
    # Strategy 1: Fibonacci sphere + L-BFGS-B
    if time.time() < deadline - 60:
        points = _fibonacci_sphere(N)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=3000, tol=1e-10)
        try_candidate(opt_points, "fib+lbfgs")
    
    # Strategy 2: Multiple random starts with L-BFGS-B
    for seed in range(8):
        if time.time() > deadline - 50:
            break
        points = _random_uniform_sphere(N, seed=seed * 100 + 42)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=1500, tol=1e-9)
        try_candidate(opt_points, f"rand{seed}+lbfgs")
    
    # Strategy 3: Momentum gradient descent polish from best
    if best_points is not None and time.time() < deadline - 30:
        mgd_pts, mgd_e = _momentum_gradient_descent(best_points, max_time=20.0, seed=123)
        try_candidate(mgd_pts, "mgd_polish")
        
        # L-BFGS-B after MGD
        if time.time() < deadline - 15:
            opt_points, energy = _optimize_lbfgsb(mgd_pts, maxiter=2000, tol=1e-11)
            try_candidate(opt_points, "mgd+lbfgs")
    
    # Strategy 4: Basin hopping from best
    if best_points is not None and time.time() < deadline - 25:
        bh_pts, bh_e = _basin_hopping(best_points, max_time=15.0, seed=456)
        try_candidate(bh_pts, "basin_hopping")
        
        # Polish after basin hopping
        if time.time() < deadline - 10:
            opt_points, energy = _optimize_lbfgsb(bh_pts, maxiter=3000, tol=1e-11)
            try_candidate(opt_points, "bh+lbfgs")
    
    # Strategy 5: Second basin hopping with different seed
    if best_points is not None and time.time() < deadline - 12:
        bh2_pts, bh2_e = _basin_hopping(best_points, max_time=8.0, seed=789)
        try_candidate(bh2_pts, "bh2")
        
        if time.time() < deadline - 5:
            opt_points, energy = _optimize_lbfgsb(bh2_pts, maxiter=2000, tol=1e-11)
            try_candidate(opt_points, "bh2+lbfgs")
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        opt_points, energy = _optimize_lbfgsb(best_points, maxiter=5000, tol=1e-12)
        try_candidate(opt_points, "final_polish")
        
        # One more MGD pass
        if time.time() < deadline - 1:
            mgd_pts2, mgd_e2 = _momentum_gradient_descent(best_points, max_time=2.0, seed=999)
            try_candidate(mgd_pts2, "final_mgd")
    
    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
