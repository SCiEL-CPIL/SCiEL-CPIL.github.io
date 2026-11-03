"""Rewritten program: optimize 100 points on sphere to minimize log energy."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _energy_and_grad(pts):
    """Compute log energy and gradient for N points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i (in R^3): sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Project gradient to tangent space: grad_tan_i = grad_i - (grad_i . x_i) x_i
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)  # (n, n)
    
    # Energy: sum over i<j of log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dist_sq[i,j]
    # Note: diff[i,j] = x_i - x_j
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)  # (n, 1)
    grad_tan = grad - grad_dot_x * pts  # (n, 3)
    
    return energy, grad_tan


def _energy_only(pts):
    """Fast energy computation for comparison."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial configuration."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_from_start(pts0, time_budget):
    """Run optimization from a given starting point, respecting time budget."""
    start_time = time.time()
    
    # Flatten to 1D for scipy
    x0 = pts0.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    best_pts = pts0.copy()
    best_energy = objective(x0)
    
    # Use L-BFGS-B with projected gradient approach
    # We'll do manual gradient descent with backtracking since we need to stay on sphere
    lr = 0.01
    momentum = 0.9
    vel = np.zeros_like(x0)
    
    max_iter = 2000
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        pts = _project_to_sphere(x0.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        # Update velocity (momentum)
        vel = momentum * vel + lr * g.flatten()
        
        # Try step
        x_new = x0 - vel
        pts_new = _project_to_sphere(x_new.reshape(N, 3))
        e_new = _energy_only(pts_new)
        
        # Backtracking if energy increased
        lr_eff = lr
        while e_new > e and lr_eff > 1e-8:
            lr_eff *= 0.5
            x_new = x0 - vel * (lr_eff / lr)
            pts_new = _project_to_sphere(x_new.reshape(N, 3))
            e_new = _energy_only(pts_new)
        
        if e_new <= e:
            x0 = x_new
            lr = lr_eff
        else:
            lr *= 0.5
            vel *= 0.5
        
        # Check convergence
        if it > 100 and np.linalg.norm(g) < 1e-8:
            break
    
    return best_pts, best_energy


def _refine_with_scipy(pts, time_budget):
    """Use scipy minimize for final polish."""
    start_time = time.time()
    x0 = pts.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        return g.flatten()
    
    try:
        result = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                         options={'maxiter': 500, 'gtol': 1e-10,
                                  'maxfun': 2000})
        if result.success or result.fun < objective(x0):
            pts_new = _project_to_sphere(result.x.reshape(N, 3))
            e_new = _energy_only(pts_new)
            e_old = _energy_only(pts)
            if e_new < e_old:
                return pts_new
    except Exception:
        pass
    
    return pts


def construct():
    global N
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Phase 1: Gradient descent with momentum
    remaining = deadline - time.time() - 5.0  # leave 5s for refinement
    if remaining > 10:
        pts, e = _optimize_from_start(pts, remaining)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 2: Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    remaining = deadline - time.time() - 10.0
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        # Perturbed Fibonacci start
        pts_init = _fibonacci_sphere(N)
        noise = rng.normal(0, 0.1, size=(N, 3))
        pts_init = _project_to_sphere(pts_init + noise)
        
        t_rem = min(deadline - time.time() - 8.0, 15.0)
        if t_rem < 5:
            break
        pts, e = _optimize_from_start(pts_init, t_rem)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 3: Final refinement with scipy
    remaining = deadline - time.time() - 3.0
    if remaining > 5:
        best_pts = _refine_with_scipy(best_pts, remaining)
    
    # Final check
    final_energy = _energy_only(best_pts)
    if final_energy < best_energy:
        best_energy = final_energy
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
