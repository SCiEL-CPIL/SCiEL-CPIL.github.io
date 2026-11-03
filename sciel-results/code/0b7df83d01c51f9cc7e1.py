"""Minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _log_energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    return -math.fsum(np.log(d).tolist())


def _log_energy_grad(x):
    """Compute gradient of logarithmic energy with respect to x."""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    coeff = 1.0 / dist_sq
    np.fill_diagonal(coeff, 0.0)
    
    coeff_sum = coeff.sum(axis=1, keepdims=True)
    grad = coeff @ x - coeff_sum * x
    return grad


def _project_to_sphere(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _golden_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _tangent_project(grad, x):
    """Project gradient to tangent space of sphere at each point."""
    return grad - np.sum(grad * x, axis=1, keepdims=True) * x


def _optimize_lbfgs(x0, time_budget, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        e = _log_energy(x)
        g = _log_energy_grad(x)
        g_tangent = _tangent_project(g, x)
        return e, g_tangent.flatten()
    
    x_flat = x0.flatten()
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    try:
        res = minimize(
            lambda xf: objective_and_grad(xf),
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': gtol}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _log_energy(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.5:
        try:
            res = minimize(
                lambda xf: objective_and_grad(xf),
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol * 0.01, 'gtol': gtol * 0.01}
            )
            x_opt = _project_to_sphere(res.x.reshape(-1, 3))
            e_opt = _log_energy(x_opt)
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x, best_e


def _kick_and_polish(x0, time_budget, rng, n_kicks=5):
    """
    Random kick-and-polish strategy: perturb the current best configuration
    with random perturbations of various magnitudes, then polish each with L-BFGS.
    This provides global exploration more efficiently than simulated annealing.
    """
    start_time = time.time()
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    n = x0.shape[0]
    
    # Try several kick magnitudes
    kick_scales = [0.3, 0.15, 0.08, 0.04, 0.02]
    
    for scale in kick_scales:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        # Generate a random perturbation
        perturb = rng.standard_normal((n, 3)) * scale
        # Project to tangent space
        perturb = perturb - np.sum(perturb * x0, axis=1, keepdims=True) * x0
        
        x_kicked = x0 + perturb
        x_kicked = _project_to_sphere(x_kicked)
        
        # Polish with L-BFGS
        kick_budget = max(2.0, remaining * 0.4)
        x_pol, e_pol = _optimize_lbfgs(x_kicked, kick_budget, maxiter=1500, ftol=1e-14, gtol=1e-11)
        
        if e_pol < best_e:
            best_e = e_pol
            best_x = x_pol.copy()
    
    return best_x, best_e


def _kick_and_polish(x0, time_budget, rng, n_kicks=5):
    """
    Random kick-and-polish strategy: perturb the current best configuration
    with random perturbations of various magnitudes, then polish each with L-BFGS.
    This provides global exploration more efficiently than simulated annealing.
    """
    start_time = time.time()
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    n = x0.shape[0]
    
    # Try several kick magnitudes
    kick_scales = [0.3, 0.15, 0.08, 0.04, 0.02]
    
    for scale in kick_scales:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        # Generate a random perturbation
        perturb = rng.standard_normal((n, 3)) * scale
        # Project to tangent space
        perturb = perturb - np.sum(perturb * x0, axis=1, keepdims=True) * x0
        
        x_kicked = x0 + perturb
        x_kicked = _project_to_sphere(x_kicked)
        
        # Polish with L-BFGS
        kick_budget = max(2.0, remaining * 0.4)
        x_pol, e_pol = _optimize_lbfgs(x_kicked, kick_budget, maxiter=1500, ftol=1e-14, gtol=1e-11)
        
        if e_pol < best_e:
            best_e = e_pol
            best_x = x_pol.copy()
    
    return best_x, best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Golden spiral
    inits.append(_golden_spiral(N))
    
    # Perturbed golden spiral
    for scale in [0.1, 0.05, 0.02, 0.01]:
        perturbed = _golden_spiral(N) + scale * rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(perturbed))
    
    # Random points
    for _ in range(6):
        inits.append(_random_sphere_points(N, rng))
    
    # Different golden spiral offsets
    for offset in [1.0, 2.0, 0.5, 3.0, 1.5, 4.0, 0.3]:
        inits.append(_golden_spiral(N, seed_offset=offset))
    
    # More random with different seeds
    for seed in [123, 456, 789]:
        r = np.random.default_rng(seed)
        inits.append(_random_sphere_points(N, r))
    
    best_x = None
    best_e = float('inf')
    
    num_inits = len(inits)
    
    for i, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < 12:
            break
        
        # Allocate time budget for this optimization
        time_budget = max(3.0, remaining / max(1, num_inits - i))
        
        # First try L-BFGS-B with gradient
        x_opt, e_opt = _optimize_lbfgs(x0, time_budget * 0.5, maxiter=1500, ftol=1e-14, gtol=1e-11)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # Check if we're running low on time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 25:
            break
        
        # If we have time, try kick-and-polish from the result
        remaining = TIME_LIMIT - elapsed
        if remaining > 15:
            kp_budget = min(12.0, remaining * 0.35)
            x_kp, e_kp = _kick_and_polish(x_opt, kp_budget, rng)
            
            if e_kp < best_e:
                best_e = e_kp
                best_x = x_kp.copy()
    
    # Final high-precision polish with tight tolerances
    remaining = TIME_LIMIT - time.time()
    if remaining > 5 and best_x is not None:
        # First, a medium polish
        x_final, e_final = _optimize_lbfgs(best_x, min(8.0, remaining * 0.5), maxiter=3000, ftol=1e-15, gtol=1e-13)
        if e_final < best_e:
            best_e = e_final
            best_x = x_final.copy()
        
        # Then, a very tight polish if time permits
        remaining = TIME_LIMIT - time.time()
        if remaining > 5:
            x_final2, e_final2 = _optimize_lbfgs(best_x, min(6.0, remaining * 0.7), maxiter=5000, ftol=1e-16, gtol=1e-14)
            if e_final2 < best_e:
                best_e = e_final2
                best_x = x_final2.copy()
    
    if best_x is None:
        best_x = _golden_spiral(N)
    
    return best_x
