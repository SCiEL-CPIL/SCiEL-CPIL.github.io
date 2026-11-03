"""Rewritten program: Riesz/Logarithmic energy minimisation via L-BFGS-B on sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_and_grad(pts):
    """Return energy and Riemannian gradient (tangent-space) for log energy on sphere."""
    pts = _project_to_sphere(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    weight = 1.0 / dist_sq
    np.fill_diagonal(weight, 0.0)
    
    grad = -np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= grad_dot * pts
    
    return energy, grad


def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.normal(size=(n, 3))
    return _project_to_sphere(pts)


def _optimize_lbfgsb(pts0, maxiter=2000, ftol=1e-16, gtol=1e-12):
    """Run L-BFGS-B optimization from given starting points."""
    try:
        result = minimize(
            lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
            pts0.flatten(),
            jac=lambda x: _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'disp': False, 'maxls': 50}
        )
        if result.fun < 1e10:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < 1e10:
                return candidate, cand_energy
    except Exception:
        pass
    return None, np.inf


def _optimize_trustconstr(pts0, maxiter=500):
    """Run trust-constr optimization with sphere constraints."""
    try:
        from scipy.optimize import NonlinearConstraint
        
        def sphere_constraint(x):
            return np.sum(x.reshape(N, 3) ** 2, axis=1)
        
        constraint = NonlinearConstraint(sphere_constraint, 1.0, 1.0, jac=lambda x: 2.0 * x.reshape(N, 3))
        
        result = minimize(
            lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
            pts0.flatten(),
            jac=lambda x: _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
            method='trust-constr',
            constraints=[constraint],
            options={'maxiter': maxiter, 'gtol': 1e-10, 'xtol': 1e-14}
        )
        if result.fun < 1e10:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < 1e10:
                return candidate, cand_energy
    except Exception:
        pass
    return None, np.inf


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    best_pts = None
    best_energy = np.inf
    
    starts = []
    
    # Fibonacci spiral variants
    for offset in [0.0, 0.3, 0.7, 1.1, 1.5, 2.0]:
        starts.append(_fibonacci_spiral(N, offset))
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        rng = np.random.default_rng(seed)
        starts.append(_random_points(N, rng))
    
    # Optimize from each start using L-BFGS-B
    for start in starts:
        pts, energy = _optimize_lbfgsb(start, maxiter=1500, ftol=1e-14, gtol=1e-10)
        if pts is not None and energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Refinement: run trust-constr on best result
    if best_pts is not None:
        pts_tc, energy_tc = _optimize_trustconstr(best_pts, maxiter=300)
        if pts_tc is not None and energy_tc < best_energy:
            best_energy = energy_tc
            best_pts = pts_tc
    
    # Additional L-BFGS-B refinement passes
    if best_pts is not None:
        for _ in range(5):
            pts, energy = _optimize_lbfgsb(best_pts, maxiter=1000, ftol=1e-16, gtol=1e-14)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_pts = pts
            else:
                break
    
    # Final projection and validation
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any points too close together
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        rng = np.random.default_rng(42)
        best_pts = best_pts + rng.normal(0, 1e-6, best_pts.shape)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
