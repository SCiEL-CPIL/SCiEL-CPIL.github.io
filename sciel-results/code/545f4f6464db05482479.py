"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d) and its gradient.
    
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    # True gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j)/d^2
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = math.fsum(np.log(1.0 / d).tolist())

    return energy, grad


def _energy_only(pts):
    """Compute log energy only (faster, no gradient)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(grad, pts):
    """Project gradient to tangent space of sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0, tight=False):
    """Run L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    if tight:
        options = {
            'maxiter': 50000,
            'maxfun': 100000,
            'ftol': 1e-18,
            'gtol': 1e-16,
            'maxls': 500
        }
    else:
        options = {
            'maxiter': 10000,
            'maxfun': 30000,
            'ftol': 1e-15,
            'gtol': 1e-13,
            'maxls': 100
        }

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options=options
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _barzilai_borwein_polish(pts, time_limit=10.0):
    """
    Barzilai-Borwein step size adaptation for gradient descent on the sphere.
    This is often faster than L-BFGS in the final stages for large N because
    it uses only O(1) memory and adapts the step size based on the recent
    change in gradient, which helps with the ill-conditioning near the minimum.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    # Initial step size
    alpha = 0.01
    prev_pts = pts.copy()
    prev_grad = None
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    
    max_iter = 100000
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Project gradient to tangent space
        grad_t = _tangent_project(grad, pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-14:
            break
            
        if prev_grad is not None:
            # Barzilai-Borwein step size calculation
            # s_k = x_k - x_{k-1}
            # y_k = g_k - g_{k-1}
            # alpha_BB = <s_k, s_k> / <s_k, y_k> (BB1)
            # alpha_BB = <s_k, y_k> / <y_k, y_k> (BB2)
            
            s = pts - prev_pts
            y = grad_t - prev_grad
            
            sy = np.sum(s * y)
            yy = np.sum(y * y)
            ss = np.sum(s * s)
            
            if yy > 1e-16:
                alpha_bb2 = sy / yy
                if ss > 1e-16:
                    alpha_bb1 = ss / sy if sy > 1e-16 else alpha_bb2
                else:
                    alpha_bb1 = alpha_bb2
                
                # Use BB2 or average, with damping
                # Damping to ensure global convergence
                alpha_new = alpha_bb2
                if alpha_new < 1e-6:
                    alpha_new = 1e-6
                if alpha_new > 1.0:
                    alpha_new = 1.0
                alpha = alpha_new
            else:
                alpha *= 0.5
                
        # Update
        step = alpha * grad_t
        new_pts = pts - step
        new_pts = _project_to_sphere(new_pts)
        
        # If step was too large and energy increased significantly, damp it
        new_energy = _energy_only(new_pts)
        if new_energy > energy + 0.1:
             alpha *= 0.5
             step = alpha * grad_t
             new_pts = pts - step
             new_pts = _project_to_sphere(new_pts)
             new_energy = _energy_only(new_pts)
             
        prev_pts = pts
        prev_grad = grad_t
        pts = new_pts
        
    return best_pts, best_energy


def _final_polish(pts, time_limit=5.0):
    """High-precision final polish combining L-BFGS and BB."""
    start_time = time.time()
    
    # First, a tight L-BFGS run
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        pts, e1 = _lbfgs_optimize(pts, time_limit=remaining * 0.6, tight=True)
    else:
        e1 = _energy_only(pts)
        
    # Then, a BB polish to catch any remaining drift
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        pts, e2 = _barzilai_borwein_polish(pts, time_limit=remaining)
        if e2 < e1:
            return pts, e2
    return pts, e1


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)

    best_pts = None
    best_energy = np.inf
    
    # Store top candidates to polish later
    candidates = []

    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_fibonacci_sphere(N, seed=1))
    starts.append(_fibonacci_sphere(N, seed=7))
    starts.append(_fibonacci_sphere(N, seed=13))
    starts.append(_fibonacci_sphere(N, seed=42))
    starts.append(_fibonacci_sphere(N, seed=123))
    for s in range(2, 8):
        starts.append(_random_sphere(N, seed=s * 100))

    # Phase 1: L-BFGS-B from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 40:
            break
        time_limit = min(4.0, deadline - elapsed - 35)
        if time_limit < 2.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()
            
        # Keep top 5 candidates
        candidates.append((candidate_energy, candidate_pts))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 5:
            candidates.pop()

    # Phase 2: Intensive Polish of the best candidates
    # We have spent ~30s on Phase 1. We have ~110s left.
    # Let's spend the remaining time polishing the top 3 candidates.
    
    if candidates:
        # Sort candidates by energy
        candidates.sort(key=lambda x: x[0])
        
        # Determine time budget per polish
        # We want to leave 5s for final safety check
        time_for_polish = deadline - (time.time() - start_time) - 5.0
        if time_for_polish > 10.0:
            # Polish top 3
            n_polish = 3
            time_per_polish = time_for_polish / n_polish
            
            for i in range(min(n_polish, len(candidates))):
                elapsed = time.time() - start_time
                if elapsed > deadline - 6:
                    break
                
                # Allocate time based on rank
                # Best candidate gets more time
                if i == 0:
                    t_limit = time_per_polish * 1.5
                else:
                    t_limit = time_per_polish * 0.75
                    
                t_limit = min(t_limit, deadline - elapsed - 5.0)
                if t_limit < 2.0:
                    continue
                    
                c_energy, c_pts = candidates[i]
                polished_pts, polished_energy = _final_polish(c_pts, time_limit=t_limit)
                
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts.copy()

    # Final safety check and projection
    if best_pts is not None:
        best_pts = _project_to_sphere(best_pts)
        
        # Validate: check for very close points
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)

        if min_dist < 1e-9:
            # Points too close, fallback to best Fibonacci
            return _fibonacci_sphere(N)

    else:
        # Should not happen, but fallback
        best_pts = _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
