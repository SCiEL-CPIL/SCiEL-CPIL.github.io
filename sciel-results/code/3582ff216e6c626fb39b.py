import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())

def _energy_and_gradient(pts):
    """Compute energy and gradient of log energy w.r.t. points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = -math.fsum(np.log(dist[iu]).tolist())
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _perturbed_fibonacci(n, scale, rng):
    """Fibonacci points with random perturbation."""
    fib = _fibonacci_points(n)
    perturbed = fib + scale * rng.standard_normal((n, 3))
    return _project_to_sphere(perturbed)

def _antipodal_pairs(n, rng):
    """Start with pairs of near-antipodal points, then perturb."""
    base = rng.standard_normal((n // 2 + n % 2, 3))
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    base = base / norms

    if n % 2 == 0:
        pts = np.vstack([base, -base])
    else:
        extra = rng.standard_normal((1, 3))
        extra = extra / np.linalg.norm(extra)
        pts = np.vstack([base, -base, extra])

    # Small perturbation to break symmetry
    pts = pts + 0.01 * rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _optimize_candidate(pts, deadline):
    """Optimize a single candidate using L-BFGS-B and Newton polish."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()

    # Pass 1: Coarse L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 40}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        pts = _project_to_sphere(res.x.reshape(N, 3))
    except Exception:
        pass

    # Pass 2: Fine L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(
                objective_and_grad,
                pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            if time.time() < deadline:
                pts = _project_to_sphere(res2.x.reshape(N, 3))
        except Exception:
            pass

    # Pass 3: Newton Polish
    if time.time() < deadline - 2.0:
        current_energy = _log_energy(pts)
        best_pts = pts.copy()
        best_polish_energy = current_energy
        
        for it in range(50):
            if time.time() > deadline:
                break
            e, grad = _energy_and_gradient(pts)
            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            grad_norm = np.linalg.norm(grad_tangent)
            
            if grad_norm < 1e-14:
                break

            step = 1.0 / (1.0 + grad_norm * 50)
            alpha = 1.0
            new_pts = None
            new_energy = None
            
            for _ in range(20):
                candidate = pts - alpha * step * grad_tangent
                candidate = _project_to_sphere(candidate)
                cand_energy = _log_energy(candidate)
                
                if cand_energy < current_energy - 1e-4 * alpha * grad_norm * grad_norm:
                    new_pts = candidate
                    new_energy = cand_energy
                    break
                alpha *= 0.5
                if alpha < 1e-8:
                    break

            if new_pts is not None and new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                if current_energy < best_polish_energy:
                    best_polish_energy = current_energy
                    best_pts = pts.copy()
            else:
                break
        
        pts = best_pts
        current_energy = best_polish_energy

    return pts, _log_energy(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci base
    initializations.append(_fibonacci_points(N))
    
    # 2. Random points with different seeds
    for seed in [1, 7, 13, 42, 99]:
        r = np.random.default_rng(seed)
        initializations.append(_random_points(N, r))
        
    # 3. Perturbed Fibonacci
    for scale in [0.05, 0.1, 0.2]:
        initializations.append(_perturbed_fibonacci(N, scale, rng))
        
    # 4. Antipodal pairs
    initializations.append(_antipodal_pairs(N, rng))
    
    # Optimize each candidate
    for init_pts in initializations:
        if time.time() > deadline:
            break
            
        # Check if we have enough time for a full optimization cycle
        # Estimate cost: ~2-5 seconds per candidate
        if time.time() > deadline - 5.0:
            break
            
        try:
            opt_pts, opt_energy = _optimize_candidate(init_pts, deadline)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            continue
            
    # If no candidate was successfully optimized (shouldn't happen), fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_points(N)
        best_energy = _log_energy(best_pts)
        
    return best_pts
