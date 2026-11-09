"""Rewritten program using L-BFGS-B with manifold gradient handling and multi-stage polish."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _TimeOutCallback():
    """Callback class to enforce per-optimization time limits."""
    def __init__(self, deadline):
        self.deadline = deadline
    def __call__(self, xk):
        if time.time() >= self.deadline:
            return True
        return False

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere (or normalized inside).
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    # Ensure points are normalized for calculation
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    u = pts / norms
    
    # Pairwise differences
    diff = u[:, None, :] - u[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Clip to avoid log(0) and division by 0
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy calculation
    # E = sum_{i<j} log(1/d) = -0.25 * sum_{i!=j} log(d_sq_ij)
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient calculation
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    w_sum = np.sum(inv_dist_sq, axis=1)
    
    term1 = inv_dist_sq @ u
    term2 = u * w_sum[:, None]
    
    grad_raw = term1 - term2
    
    # Project to tangent space
    dot = np.sum(grad_raw * u, axis=1)
    grad_tan = grad_raw - dot[:, None] * u
    
    return energy, grad_tan, u

def _objective_and_grad(x_flat):
    """
    Objective function and gradient for scipy.minimize.
    x_flat: (3N,) vector of unnormalized coordinates.
    """
    x = x_flat.reshape(N, 3)
    
    energy, grad_tan, u = _energy_and_tangent_grad(x)
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    grad_x = grad_tan / norms
    
    return energy, grad_x.flatten()

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_antipodal():
    """Generate antipodal pairs on the sphere for diverse initialization."""
    rng = np.random.default_rng(123)
    pts = []
    for _ in range(N // 2):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        pts.append(v)
        pts.append(-v)
    return np.array(pts)

def _optimize_lbfgs(start_pts, deadline, tight=False):
    """
    Optimize using L-BFGS-B with a per-call timeout guard.
    If tight=True, use tighter tolerances for a final polish pass.
    """
    x0 = start_pts.flatten()
    cb = _TimeOutCallback(deadline)
    
    if tight:
        options = {
            'maxiter': 5000, 
            'maxfun': 20000, 
            'ftol': 1e-14, 
            'gtol': 1e-12, 
            'iprint': 0
        }
    else:
        options = {
            'maxiter': 2000, 
            'maxfun': 10000, 
            'ftol': 1e-10, 
            'gtol': 1e-8, 
            'iprint': 0
        }
    
    res = minimize(
        _objective_and_grad, 
        x0, 
        jac=True, 
        method='L-BFGS-B', 
        options=options,
        callback=cb
    )
    
    return res.x.reshape(N, 3)

def construct():
    """
    Main entry point with multi-stage optimization and diverse initializations.
    Uses 150s wall-clock deadline with safety margin.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for final validation

    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Fibonacci spiral
    fib = _initial_fibonacci()
    inits.append(fib)
    
    # 2. Antipodal pairs
    antipodal = _initial_antipodal()
    inits.append(antipodal)
    
    # 3. Random points
    for _ in range(4):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    # 4. Jittered Fibonacci
    for _ in range(4):
        pts = fib + rng.normal(size=(N, 3)) * 0.15
        inits.append(_project_to_sphere(pts))
    
    # 5. Jittered Antipodal
    for _ in range(2):
        pts = antipodal + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
        
    # 6. Additional diverse inits: jittered random with varying scales
    for scale in [0.05, 0.2, 0.3]:
        pts = fib + rng.normal(size=(N, 3)) * scale
        inits.append(_project_to_sphere(pts))
    
    best_pts = inits[0].copy()
    best_energy, _, _ = _energy_and_tangent_grad(best_pts)
    
    # Stage 1: Coarse optimization from each initialization
    # Reserve time for polish stages: at least 15s for final polish
    polish_reserve = 15.0
    
    for i, init_pts in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < polish_reserve + 5.0:
            break
            
        # Per-restart time budget: use at most 60% of remaining, capped at 30s
        per_restart_budget = min(30.0, 0.6 * (remaining - polish_reserve))
        restart_deadline = time.time() + per_restart_budget
        
        try:
            optimized_pts = _optimize_lbfgs(init_pts, restart_deadline, tight=False)
            optimized_pts = _project_to_sphere(optimized_pts)
            
            e_opt, _, _ = _energy_and_tangent_grad(optimized_pts)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = optimized_pts.copy()
                
        except Exception:
            continue
        
        # If we are getting close to deadline, stop adding new restarts
        if time.time() > deadline - polish_reserve - 5.0:
            break

    # Stage 2: Fine polish on the best found
    if time.time() < deadline - 8.0:
        try:
            polish_deadline = deadline - 3.0
            final_pts = _optimize_lbfgs(best_pts, polish_deadline, tight=True)
            final_pts = _project_to_sphere(final_pts)
            e_final, _, _ = _energy_and_tangent_grad(final_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
        except Exception:
            pass

    # Stage 3: Ultra-tight polish if time permits
    if time.time() < deadline - 3.0:
        try:
            ultra_options = {
                'maxiter': 10000, 
                'maxfun': 50000, 
                'ftol': 1e-15, 
                'gtol': 1e-13, 
                'iprint': 0
            }
            x0 = best_pts.flatten()
            ultra_cb = _TimeOutCallback(deadline - 0.5)
            res = minimize(
                _objective_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                options=ultra_options,
                callback=ultra_cb
            )
            ultra_pts = res.x.reshape(N, 3)
            ultra_pts = _project_to_sphere(ultra_pts)
            e_ultra, _, _ = _energy_and_tangent_grad(ultra_pts)
            if e_ultra < best_energy:
                best_energy = e_ultra
                best_pts = ultra_pts.copy()
        except Exception:
            pass

    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms
# EVOLVE-BLOCK-END
