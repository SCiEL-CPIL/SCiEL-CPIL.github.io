"""Combined program for 100-point logarithmic energy minimization on the unit sphere.
Uses structured Hamiltonian-loop initialization, Riemannian gradient flow with
conjugate gradient, multiple restarts, perturbation-based escape, and a final
high-precision L-BFGS-B polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 140.0  # Safety margin: 150s limit, 10s buffer


def _time_left():
    return max(0.0, DEADLINE - time.time())


def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_vec(points):
    """Vectorized energy and tangent-space gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    return energy, g_tangent


def _project(points):
    """Project to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_flow(points, time_budget, initial_lr=0.05, patience=3000):
    """Riemannian gradient descent with backtracking line search."""
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-11:
            break
        
        alpha = lr
        improved = False
        for _ in range(25):
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            new_energy, _ = _energy_grad_vec(new_points)
            if new_energy < energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.95
            if no_improve > patience:
                break
    
    return best_points


def _riemannian_cg(points, time_budget, max_iter=5000):
    """Riemannian conjugate gradient with adaptive step size."""
    start_time = time.time()
    x = points.copy()
    e, g = _energy_grad_vec(x)
    p = -g
    best_points = x.copy()
    best_energy = e
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Adaptive step size via backtracking
        alpha = 1.0
        for _ in range(30):
            x_new = _project(x + alpha * p)
            e_new, g_new = _energy_grad_vec(x_new)
            if e_new < e - 1e-13:
                break
            alpha *= 0.5
        else:
            break
        
        # Fletcher-Reeves beta with damping
        beta = np.dot(g_new, g_new) / max(np.dot(g, g), 1e-30)
        beta = min(beta, 0.9)
        p = -g_new + beta * p
        # Project p to tangent space
        p = p - (p * x_new).sum(axis=1, keepdims=True) * x_new
        
        # Check convergence
        if np.linalg.norm(g_new) < 1e-10:
            x = x_new
            e = e_new
            break
        
        x = x_new
        e = e_new
        g = g_new
        
        if e < best_energy:
            best_energy = e
            best_points = x.copy()
    
    return best_points


def _polish(points, time_budget):
    """High-precision polish using L-BFGS-B."""
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 200})
        if hasattr(res, 'x'):
            return _project(res.x.reshape(-1, 3))
    except Exception:
        pass
    return points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)


def _hamiltonian_loop_init(n):
    """Build a space-filling loop using greedy nearest-neighbor on a dense grid.
    This gives a globally coherent ordering that helps the optimizer."""
    # Start with a dense Fibonacci grid, then build a Hamiltonian path
    grid_n = 200  # denser grid for better ordering
    grid = _fibonacci_spiral(grid_n)
    
    # Greedy nearest-neighbor to build a path through all grid points
    used = np.zeros(grid_n, dtype=bool)
    path = []
    current = 0
    used[0] = True
    path.append(0)
    
    for _ in range(grid_n - 1):
        # Find nearest unused point
        d = np.linalg.norm(grid - grid[current], axis=1)
        d[used] = np.inf
        nxt = np.argmin(d)
        used[nxt] = True
        path.append(nxt)
        current = nxt
    
    # Subsample to n points evenly along the path
    indices = np.linspace(0, grid_n - 1, n, dtype=int)
    points = grid[path[indices]]
    return _project(points)


def _perturb(points, scale, seed):
    """Add small random perturbation to escape local minima."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*points.shape) * scale
    return _project(points + noise)


def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    best_points = None
    best_energy = np.inf
    
    # Phase 1: Multiple structured initializations with Riemannian CG
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_hamiltonian_loop_init(N))
    
    # Add a few random inits
    for seed in [42, 123, 456]:
        inits.append(_random_points(N, seed))
    
    for init in inits:
        if _time_left() < 15.0:
            break
        
        budget = min(_time_left() * 0.15, 12.0)
        if budget < 2.0:
            break
        
        # Riemannian CG
        refined = _riemannian_cg(init, budget)
        e, _ = _energy_grad_vec(refined)
        
        # Follow up with gradient flow
        if _time_left() > 5.0:
            budget2 = min(_time_left() * 0.05, 3.0)
            refined2 = _riemannian_flow(refined, budget2, initial_lr=0.02)
            e2, _ = _energy_grad_vec(refined2)
            if e2 < e:
                refined = refined2
                e = e2
        
        if e < best_energy:
            best_energy = e
            best_points = refined.copy()
    
    # Phase 2: Perturbation-based escape from current best
    if best_points is not None and _time_left() > 10.0:
        for pert_seed in range(20):
            if _time_left() < 5.0:
                break
            for scale in [0.01, 0.005, 0.001]:
                perturbed = _perturb(best_points, scale, pert_seed * 100 + int(scale * 1000))
                budget = min(_time_left() * 0.05, 4.0)
                if budget < 1.0:
                    break
                refined = _riemannian_flow(perturbed, budget, initial_lr=0.01)
                e, _ = _energy_grad_vec(refined)
                if e < best_energy:
                    best_energy = e
                    best_points = refined.copy()
    
    # Phase 3: Final high-precision polish
    if best_points is not None and _time_left() > 5.0:
        polish_budget = min(_time_left() * 0.7, 25.0)
        polished = _polish(best_points, polish_budget)
        e, _ = _energy_grad_vec(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Final validation
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    best_points = _project(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                noise = np.random.randn(3) * 1e-5
                best_points[i] = _project(best_points[i] + noise)
    
    return best_points
