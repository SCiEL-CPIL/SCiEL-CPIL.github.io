import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 145.0

def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _energy_grad_vec(points):
    """Vectorized energy and gradient on the sphere."""
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
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-7, patience=2000):
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        alpha = lr
        improved = False
        for _ in range(20):
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-10:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
    
    return best_points

def _polish_bfgs(points, time_budget):
    """Polish using BFGS which is often better for N=100 than L-BFGS-B."""
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='BFGS', jac=True,
                       options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 50})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    return points

def _polish_lbfgs(points, time_budget):
    """Polish using L-BFGS-B with tight tolerances."""
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    return points

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)

def construct():
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    def _time_left():
        return max(0.0, DEADLINE - time.time())
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Fibonacci Spiral
    points_fib = _fibonacci_spiral(N)
    if _time_left() > 5.0:
        refined = _riemannian_flow(points_fib, min(_time_left() * 0.2, 10.0))
        polished = _polish_bfgs(refined, min(_time_left() * 0.1, 5.0))
        polished = _polish_lbfgs(polished, min(_time_left() * 0.1, 5.0))
        e, _ = _energy_grad_vec(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Phase 2: Random Restarts
    # We allocate 60% of remaining time to random restarts
    restart_time_limit = _time_left() * 0.6
    restart_start = time.time()
    
    for seed in range(1, 500):
        if time.time() - restart_start > restart_time_limit:
            break
        if _time_left() < 15.0:
            break
        
        random_points = _random_points(N, seed)
        
        # Quick flow
        budget_flow = 2.0
        refined = _riemannian_flow(random_points, budget_flow, initial_lr=0.1, patience=500)
        
        # Quick polish
        polished = _polish_bfgs(refined, 2.0)
        e, _ = _energy_grad_vec(polished)
        
        if e < best_energy:
            best_energy = e
            best_points = polished.copy()
            
    # Phase 3: Final Deep Polish on the best point
    if best_points is not None and _time_left() > 5.0:
        # Use remaining time for a very thorough polish
        budget_polish = _time_left() - 5.0
        
        # Split between BFGS and L-BFGS-B
        p1 = _polish_bfgs(best_points, budget_polish * 0.5)
        e1, _ = _energy_grad_vec(p1)
        
        p2 = _polish_lbfgs(p1, budget_polish * 0.5)
        e2, _ = _energy_grad_vec(p2)
        
        if e2 < best_energy:
            best_energy = e2
            best_points = p2
        elif e1 < best_energy:
            best_energy = e1
            best_points = p1

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
