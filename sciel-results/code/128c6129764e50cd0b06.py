"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    grad_ambient = W @ pts - W.sum(axis=1, keepdims=True) * pts
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Compute energy quickly for initialization sorting."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_sobol(n):
    """Generate quasi-random points on sphere using Sobol sequence."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=123)
    raw = sampler.random(n)
    # Map [0,1]^3 to sphere: use z from first dim, angle from second
    z = 2.0 * raw[:, 0] - 1.0
    theta = 2.0 * np.pi * raw[:, 1]
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)

def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """
    Perform gradient flow with annealing to explore configuration space.
    Uses tangent-space gradient descent with adaptive learning rate.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _precise_energy(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Linearly anneal learning rate
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _energy_and_grad(current)
        
        # Add small noise for exploration (annealed)
        noise_scale = 0.01 * (1.0 - progress)
        noise = rng.standard_normal((n, 3)) * noise_scale
        
        # Update in ambient space, then project back to sphere
        current = current - lr * g + noise
        current = _normalize(current)
        
        e_new = _fast_energy(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy

def _simulated_annealing(pts, rng, deadline, max_iter=2000):
    """
    Simulated annealing to escape local minima.
    Accepts worse solutions with probability exp(-delta_E / T).
    """
    n = pts.shape[0]
    current = pts.copy()
    current_e = _precise_energy(current)
    best = current.copy()
    best_e = current_e
    
    T_start = 0.5
    T_end = 0.001
    T = T_start
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Anneal temperature
        progress = it / max(max_iter - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Propose a random perturbation
        scale = 0.05 * (1.0 - 0.5 * progress)
        perturbed = current + rng.standard_normal((n, 3)) * scale
        perturbed = _normalize(perturbed)
        
        new_e = _fast_energy(perturbed)
        
        # Accept if better or with probability exp(-delta/T)
        if new_e < current_e:
            current = perturbed
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        else:
            delta = new_e - current_e
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = perturbed
                current_e = new_e
    
    # Final polish from best found
    if best_e < _precise_energy(pts):
        polished, polished_e = _optimize_single(best, deadline)
        if polished_e < best_e:
            best = polished
            best_e = polished_e
    
    return best, best_e

def _optimize_single(pts, deadline):
    """Run L-BFGS-B optimization from a given start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    # Phase 1: Standard L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-8,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Phase 2: Polish with tighter tolerances
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'maxfun': 20000,
                'ftol': 1e-15,
                'gtol': 1e-10,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    return best_pts, best_energy

def _final_polish(pts, deadline):
    """
    Final ultra-precise polish pass with very tight tolerances.
    Uses multiple L-BFGS restarts from slightly perturbed versions.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Try several tiny perturbations and polish each
    rng = np.random.default_rng(12345)
    perturbations = [0.0] + [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    
    for scale in perturbations:
        if time.time() > deadline:
            break
        
        if scale == 0.0:
            start = best_pts
        else:
            start = _normalize(best_pts + rng.standard_normal((n, 3)) * scale)
        
        try:
            result = minimize(
                objective_and_grad,
                start.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 15000,
                    'maxfun': 75000,
                    'ftol': 1e-18,
                    'gtol': 1e-14,
                }
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_pts = new_pts
                    best_energy = e_precise
        except Exception:
            pass
    
    # Additional multi-start polish with moderate perturbations
    for scale in [0.01, 0.005, 0.001]:
        if time.time() > deadline:
            break
        
        for _ in range(3):
            if time.time() > deadline:
                break
            start = _normalize(best_pts + rng.standard_normal((n, 3)) * scale)
            try:
                result = minimize(
                    objective_and_grad,
                    start.reshape(-1),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 8000,
                        'maxfun': 40000,
                        'ftol': 1e-16,
                        'gtol': 1e-13,
                    }
                )
                if result.fun < best_energy:
                    new_pts = _normalize(result.x.reshape(n, 3))
                    e_precise = _precise_energy(new_pts)
                    if e_precise < best_energy:
                        best_pts = new_pts
                        best_energy = e_precise
            except Exception:
                pass
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_generate_fibonacci(N))
    
    # 2. Sobol sequences (better space-filling than random)
    for seed in range(5):
        initializations.append(_generate_sobol(N))
    
    # 3. Random Points
    for i in range(10):
        initializations.append(_generate_random(N, rng))
        
    # 4. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.01, 0.05, 0.1, 0.2]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # 5. More structured initializations
    # Golden ratio based
    for offset in [0.0, 0.1, 0.2, 0.3]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Sort initializations by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Gradient flow exploration on top few initializations
    flow_deadline = start_time + 30.0
    for e_quick, init_pts in scored_inits[:6]:
        if time.time() > flow_deadline:
            break
        
        flowed_pts, e_precise = _gradient_flow(init_pts, rng, 400, 0.05, 0.001, flow_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = flowed_pts
    
    # Phase B: Simulated annealing from best found so far
    sa_deadline = start_time + 55.0
    if best_pts is not None:
        sa_pts, sa_e = _simulated_annealing(best_pts, rng, sa_deadline)
        if sa_e < best_energy:
            best_energy = sa_e
            best_pts = sa_pts
    else:
        best_pts = scored_inits[0][1]
        best_energy = scored_inits[0][0]
    
    # Phase C: L-BFGS optimization on top initializations (including SA result)
    if best_pts is not None:
        scored_inits.insert(0, (best_energy, best_pts))
    
    lbfgs_deadline = deadline - 30.0  # Reserve 30s for final polish
    for e_quick, init_pts in scored_inits[:12]:
        if time.time() > lbfgs_deadline:
            break
            
        optimized_pts, e_precise = _optimize_single(init_pts, lbfgs_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _generate_fibonacci(N)
    
    # Phase D: Final ultra-precise polish
    polish_deadline = deadline - 2.0  # 2s safety margin
    best_pts, best_energy = _final_polish(best_pts, polish_deadline)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
