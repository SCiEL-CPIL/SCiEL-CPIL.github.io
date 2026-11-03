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
    
    # Gradient of -sum log(d_ij) wrt pts[i] = sum_j (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    # = sum_j diff[i,j] / dist_sq[i,j]
    weight = 1.0 / dist_sq
    np.fill_diagonal(weight, 0.0)
    
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space: grad_t = grad - (grad · x) x
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


def _tetrahedral_starts(n):
    """Generate diverse structured starts based on tetrahedral symmetries."""
    starts = []
    # Golden ratio based spiral with different parameters
    for offset in np.linspace(0, 2*np.pi, 8, endpoint=False):
        starts.append(_fibonacci_spiral(n, offset))
    
    # Random with different seeds
    for seed in [1, 7, 13, 23, 42, 99, 123, 456, 789, 1024, 2024, 3141, 2718, 1618]:
        rng = np.random.default_rng(seed)
        starts.append(_random_points(n, rng))
    
    # Spherical designs based on roots of Chebyshev polynomials
    for k in [2, 3, 5, 7]:
        pts = np.zeros((n, 3))
        for i in range(n):
            # Use different angular distributions
            theta = np.arccos(1 - 2*(i+0.5)/n)
            phi = 2*np.pi*i/k + np.pi*i/(k+1)
            pts[i] = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
        starts.append(_project_to_sphere(pts))
    
    return starts


def _optimize_lbfgsb(pts0, maxiter=2000, ftol=1e-16, gtol=1e-12):
    """Run L-BFGS-B optimization from given starting points."""
    try:
        def obj(x):
            return _log_energy(_project_to_sphere(x.reshape(N, 3)))
        
        def jac(x):
            return _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten()
        
        result = minimize(
            obj,
            pts0.flatten(),
            jac=jac,
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


def _optimize_nelder_mead(pts0, maxiter=5000, tol=1e-10):
    """Run Nelder-Mead as a local refinement."""
    try:
        def obj(x):
            return _log_energy(_project_to_sphere(x.reshape(N, 3)))
        
        result = minimize(
            obj,
            pts0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < 1e10:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < 1e10:
                return candidate, cand_energy
    except Exception:
        pass
    return None, np.inf


def _optimize_bfgs(pts0, maxiter=1000, gtol=1e-12):
    """Run BFGS as an alternative local optimizer."""
    try:
        def obj(x):
            return _log_energy(_project_to_sphere(x.reshape(N, 3)))
        
        def jac(x):
            return _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten()
        
        result = minimize(
            obj,
            pts0.flatten(),
            jac=jac,
            method='BFGS',
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        if result.fun < 1e10:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < 1e10:
                return candidate, cand_energy
    except Exception:
        pass
    return None, np.inf


def _refine_with_perturbation(pts, rng, n_perturbs=10, scale=0.01):
    """Try small random perturbations and re-optimize to escape local minima."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    for _ in range(n_perturbs):
        perturbed = pts + rng.normal(0, scale, pts.shape)
        perturbed = _project_to_sphere(perturbed)
        
        # Quick L-BFGS-B from perturbed start
        cand_pts, cand_energy = _optimize_lbfgsb(perturbed, maxiter=500, ftol=1e-14, gtol=1e-10)
        if cand_pts is not None and cand_energy < best_energy:
            best_energy = cand_energy
            best_pts = cand_pts
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    best_pts = None
    best_energy = np.inf
    
    # Generate diverse starting points
    starts = _tetrahedral_starts(N)
    
    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        pts, energy = _optimize_lbfgsb(start, maxiter=2000, ftol=1e-15, gtol=1e-11)
        if pts is not None and energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Phase 2: BFGS refinement on best result
    if best_pts is not None:
        pts_b, energy_b = _optimize_bfgs(best_pts, maxiter=500, gtol=1e-13)
        if pts_b is not None and energy_b < best_energy:
            best_energy = energy_b
            best_pts = pts_b
    
    # Phase 3: Nelder-Mead local refinement
    if best_pts is not None:
        pts_nm, energy_nm = _optimize_nelder_mead(best_pts, maxiter=2000, tol=1e-11)
        if pts_nm is not None and energy_nm < best_energy:
            best_energy = energy_nm
            best_pts = pts_nm
    
    # Phase 4: Perturbation-based escape from local minima
    if best_pts is not None:
        rng = np.random.default_rng(42)
        best_pts, best_energy = _refine_with_perturbation(best_pts, rng, n_perturbs=8, scale=0.02)
        
        # Additional L-BFGS-B passes with tighter tolerances
        for _ in range(3):
            pts, energy = _optimize_lbfgsb(best_pts, maxiter=1500, ftol=1e-16, gtol=1e-14)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_pts = pts
            else:
                break
    
    # Phase 5: Final Nelder-Mead polish
    if best_pts is not None:
        pts_final, energy_final = _optimize_nelder_mead(best_pts, maxiter=3000, tol=1e-12)
        if pts_final is not None and energy_final < best_energy:
            best_energy = energy_final
            best_pts = pts_final
    
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
