"""Minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
TIME_LIMIT = 135.0  # Safety margin before 150s kill


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


def _log_energy_fast(x):
    """Fast energy computation for optimization (uses numpy sum, less precise)."""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    return -np.sum(np.log(d))


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
        e = _log_energy_fast(x)
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
    if elapsed < time_budget * 0.4:
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
    
    # Final high-precision polish with exact energy
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2.0:
        def precise_objective_and_grad(x_flat):
            x = x_flat.reshape(-1, 3)
            x = _project_to_sphere(x)
            e = _log_energy(x)
            g = _log_energy_grad(x)
            g_tangent = _tangent_project(g, x)
            return e, g_tangent.flatten()
        
        try:
            res = minimize(
                precise_objective_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            x_opt = _project_to_sphere(res.x.reshape(-1, 3))
            e_opt = _log_energy(x_opt)
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x, best_e


def _simulated_annealing(x0, time_budget, rng):
    """
    Simulated annealing on the sphere using Metropolis-Hastings.
    The proposal is a Gaussian perturbation in tangent space, projected back.
    This provides global exploration to escape local minima.
    """
    start_time = time.time()
    x = x0.copy()
    e = _log_energy_fast(x)
    best_x = x.copy()
    best_e = _log_energy(x)
    
    n = x.shape[0]
    T_start = 0.5
    T_end = 1e-8
    
    # Adaptive: run until time budget exhausted
    step = 0
    while time.time() - start_time < time_budget:
        # Geometric cooling with adaptive step count
        progress = min(1.0, (time.time() - start_time) / time_budget)
        T = T_start * (T_end / T_start) ** progress
        
        # Adaptive step size: larger early, smaller late
        step_size = 0.15 * T + 0.0005
        
        # Generate random tangent vectors
        perturb = rng.standard_normal((n, 3)) * step_size
        # Project to tangent space
        perturb = perturb - np.sum(perturb * x, axis=1, keepdims=True) * x
        
        x_prop = x + perturb
        x_prop = _project_to_sphere(x_prop)
        
        e_prop = _log_energy_fast(x_prop)
        
        # Metropolis criterion
        delta_e = e_prop - e
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            x = x_prop
            e = e_prop
            
            if e < best_e - 1e-10:
                best_e = e
                best_x = x.copy()
        
        step += 1
        # Check every 100 steps to avoid excessive time checks
        if step % 100 == 0 and time.time() - start_time > time_budget:
            break
    
    # Final precise evaluation
    best_e = _log_energy(best_x)
    
    return best_x, best_e


def _cooperative_gradient_descent(x0, time_budget, rng):
    """
    Cooperative multipoint gradient descent with adaptive steps.
    Multiple points are updated simultaneously with momentum.
    """
    start_time = time.time()
    x = x0.copy()
    e = _log_energy_fast(x)
    best_x = x.copy()
    best_e = _log_energy(x)
    
    n = x.shape[0]
    momentum = np.zeros_like(x)
    beta = 0.9
    lr = 0.005
    
    step = 0
    while time.time() - start_time < time_budget:
        grad = _log_energy_grad(x)
        grad_tangent = _tangent_project(grad, x)
        
        # Adaptive learning rate with decay
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm > 1e-12:
            lr = 0.005 / (1.0 + step * 0.0005)
        
        momentum = beta * momentum + (1 - beta) * grad_tangent
        x = x - lr * momentum
        x = _project_to_sphere(x)
        
        e = _log_energy_fast(x)
        if e < best_e - 1e-10:
            best_e = e
            best_x = x.copy()
        
        step += 1
        if step % 200 == 0 and time.time() - start_time > time_budget:
            break
    
    # Final precise evaluation
    best_e = _log_energy(best_x)
    
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
        if remaining < 15:
            break
        
        # Allocate time budget: prioritize early initializations more
        # Use a decaying budget allocation
        frac_remaining = remaining / (TIME_LIMIT - start_time)
        base_budget = remaining / max(1, num_inits - i)
        time_budget = max(4.0, min(20.0, base_budget * (1.0 + 0.5 * frac_remaining)))
        
        # First try L-BFGS-B with gradient
        lbfgs_budget = time_budget * 0.4
        x_opt, e_opt = _optimize_lbfgs(x0, lbfgs_budget, maxiter=1500, ftol=1e-14, gtol=1e-11)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # Check if we're running low on time
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < 20:
            break
        
        # If we have time, try simulated annealing from the result
        if remaining > 20:
            sa_budget = min(8.0, remaining * 0.25)
            x_sa, e_sa = _simulated_annealing(x_opt, sa_budget, rng)
            
            if e_sa < best_e:
                best_e = e_sa
                best_x = x_sa.copy()
            
            # Polish the SA result with L-BFGS
            remaining = TIME_LIMIT - time.time()
            if remaining > 12:
                x_pol, e_pol = _optimize_lbfgs(x_sa, min(6.0, remaining * 0.4), maxiter=1000, ftol=1e-15, gtol=1e-12)
                if e_pol < best_e:
                    best_e = e_pol
                    best_x = x_pol.copy()
        
        # If we have time, try cooperative gradient descent
        remaining = TIME_LIMIT - time.time()
        if remaining > 20 and i % 3 == 0:
            cg_budget = min(6.0, remaining * 0.2)
            x_cg, e_cg = _cooperative_gradient_descent(x_opt, cg_budget, rng)
            
            if e_cg < best_e:
                best_e = e_cg
                best_x = x_cg.copy()
            
            # Polish
            remaining = TIME_LIMIT - time.time()
            if remaining > 12:
                x_pol, e_pol = _optimize_lbfgs(x_cg, min(6.0, remaining * 0.4), maxiter=1000, ftol=1e-15, gtol=1e-12)
                if e_pol < best_e:
                    best_e = e_pol
                    best_x = x_pol.copy()
    
    # Final polish with tight tolerances
    remaining = TIME_LIMIT - time.time()
    if remaining > 8 and best_x is not None:
        # Multiple polish passes with decreasing tolerance
        for pass_i in range(3):
            remaining = TIME_LIMIT - time.time()
            if remaining < 5:
                break
            ftol = 1e-16 / (10 ** pass_i)
            gtol = 1e-14 / (10 ** pass_i)
            x_final, e_final = _optimize_lbfgs(best_x, min(8.0, remaining * 0.6), maxiter=2000, ftol=ftol, gtol=gtol)
            if e_final < best_e:
                best_e = e_final
                best_x = x_final.copy()
    
    if best_x is None:
        best_x = _golden_spiral(N)
    
    return best_x
