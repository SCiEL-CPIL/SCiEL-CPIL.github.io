"""
Optimized algorithm for 100-point logarithmic energy minimization on the unit sphere.
Uses Riemannian Conjugate Gradient for fast convergence, followed by L-BFGS-B polishing.
Includes multiple diverse initializations and systematic perturbation to escape local minima.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s kill


def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_vec(points):
    """Vectorized energy and tangent gradient on the sphere."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy (fast, for comparison during optimization)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent


def _project(points):
    """Project to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_cg(points, time_budget, initial_lr=0.05, min_lr=1e-7, patience=500):
    """Riemannian Conjugate Gradient method."""
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    
    x = best_points
    energy, g = _energy_grad_vec(x)
    p = -g
    beta = 0.0
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        grad_norm = np.linalg.norm(g)
        if grad_norm < 1e-10:
            break
        
        # Line search
        alpha = lr
        improved = False
        for _ in range(20):
            # Riemannian step
            new_x = x + alpha * p
            new_x = _project(new_x)
            
            new_energy, new_g = _energy_grad_vec(new_x)
            
            if new_energy < energy - 1e-10:
                x = new_x
                energy = new_energy
                g = new_g
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            if energy < best_energy:
                best_energy = energy
                best_points = x.copy()
            no_improve = 0
            
            # Update direction (Fletcher-Reeves)
            new_grad_norm_sq = np.sum(g * g)
            beta = new_grad_norm_sq / (np.sum(g * g) + 1e-15)
            # Approximate parallel transport by projecting old p to new tangent space
            p_transported = p - (p * x).sum(axis=1, keepdims=True) * x
            p = -g + beta * p_transported
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
    
    return best_points


def _polish_lbfgs(points, time_budget):
    """High-precision polish using L-BFGS-B."""
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _fibonacci_spiral(n, offset=0.5):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)


def _kick_and_optimize(points, time_budget, kick_scale=0.1):
    """Perturb points and optimize."""
    rng = np.random.RandomState(int(time.time() * 1000) % 100000)
    noise = rng.randn(N, 3) * kick_scale
    kicked_points = _project(points + noise)
    
    budget_kick = time_budget * 0.4
    kicked_opt = _riemannian_cg(kicked_points, budget_kick, initial_lr=0.05, patience=500)
    
    e_kicked = _energy_fsum(kicked_opt)
    e_orig = _energy_fsum(points)
    
    if e_kicked < e_orig:
        return kicked_opt, e_kicked
    else:
        return points, e_orig


def construct():
    start_time = time.time()
    
    def _time_left():
        return max(0.0, TIME_LIMIT - (time.time() - start_time))
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Riemannian CG from multiple diverse initializations
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        if _time_left() < 5.0:
            break
        points_init = _fibonacci_spiral(N, offset)
        budget = min(_time_left() * 0.1, 10.0)
        refined = _riemannian_cg(points_init, budget, initial_lr=0.05, patience=1000)
        e = _energy_fsum(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: Multiple random restarts
    restart_time_limit = _time_left() * 0.4
    restart_start = time.time()
    
    for seed in range(1, 500):
        if time.time() - restart_start > restart_time_limit:
            break
        if _time_left() < 10.0:
            break
        
        random_points = _random_points(N, seed)
        
        budget_per = min(2.0, _time_left() * 0.05)
        if budget_per < 0.5:
            break
        
        refined = _riemannian_cg(random_points, budget_per, initial_lr=0.1, patience=500)
        e = _energy_fsum(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 3: Polish the best with L-BFGS-B
    if _time_left() > 5.0:
        polish_budget = min(_time_left() * 0.5, 30.0)
        polished = _polish_lbfgs(best_points, polish_budget)
        e = _energy_fsum(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Phase 4: Kick and re-optimize
    if _time_left() > 5.0:
        for kick_scale in [0.03, 0.05, 0.1, 0.15]:
            if _time_left() < 3.0:
                break
            remaining = _time_left() * 0.3
            if remaining < 2.0:
                break
            kicked, e_kicked = _kick_and_optimize(best_points, remaining, kick_scale)
            if e_kicked < best_energy:
                best_energy = e_kicked
                best_points = kicked
                if _time_left() > 3.0:
                    rem = _time_left() * 0.3
                    if rem > 2.0:
                        new_opt = _polish_lbfgs(best_points, rem)
                        e_new = _energy_fsum(new_opt)
                        if e_new < best_energy:
                            best_energy = e_new
                            best_points = new_opt
    
    # Phase 5: Final high-precision polish
    if _time_left() > 3.0:
        final_budget = _time_left() * 0.5
        if final_budget > 2.0:
            final_opt = _polish_lbfgs(best_points, final_budget)
            e_final = _energy_fsum(final_opt)
            if e_final < best_energy:
                best_energy = e_final
                best_points = final_opt
    
    # Final projection and validation
    best_points = _project(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                perturb = rng.randn(3) * 1e-5
                best_points[i] = _project(best_points[i] + perturb)
    
    # Final energy check
    final_energy = _energy_fsum(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N, 0.5)
    
    return best_points
