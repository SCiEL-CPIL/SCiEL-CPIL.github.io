"""Minimal logarithmic energy of 150 points on the sphere - Basin Hopping with L-BFGS-B."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
MARGIN = 8.0

def _energy_grad(pts):
    """
    Vectorized logarithmic energy and gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient (R3) = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Returns (energy, grad) where grad is (N,3) in R3.
    """
    n = N
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    
    d = np.sqrt(d2)
    
    iu = np.triu_indices(n, 1)
    d_upper = d[iu]
    if np.any(d_upper < 1e-14):
        return float('inf'), np.zeros((n, 3))
    
    terms = np.log(1.0 / d_upper)
    energy = math.fsum(terms.tolist())
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    col_sum = np.sum(inv_d2, axis=1)
    weighted = inv_d2 @ pts
    grad = pts * col_sum[:, None] - weighted
    
    return energy, grad

def _project(pts):
    """Project to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_sphere(n):
    """Fibonacci spiral initialization."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(pts)

def _halton_sphere(n):
    """Halton sequence initialization."""
    def halton(idx, base):
        result = 0.0
        f = 1.0
        i = idx
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    pts = []
    for i in range(1, n + 1):
        x, y, z = halton(i, 2), halton(i, 3), halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        pts.append(p / norm if norm > 1e-15 else np.array([1.0, 0.0, 0.0]))
    return np.array(pts)

def _sobol_sphere(n, seed=0):
    """Sobol-like quasi-random initialization using structured jitter."""
    rng = np.random.default_rng(seed)
    z_vals = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0/n
    z_vals = (z_vals - 1.0) * 2.0 - 1.0
    z_vals = np.clip(z_vals, -1.0 + 1e-6, 1.0 - 1e-6)
    z_vals = z_vals + rng.uniform(-0.3, 0.3, n)
    z_vals = np.clip(z_vals, -1.0 + 1e-6, 1.0 - 1e-6)
    
    phi_vals = 2.0 * np.pi * (np.arange(n) * (1.0 + np.sqrt(5.0)) / 2.0) % (2.0 * np.pi)
    phi_vals = phi_vals + rng.uniform(-0.1, 0.1, n)
    
    r = np.sqrt(np.maximum(0.0, 1.0 - z_vals * z_vals))
    pts = np.column_stack([r * np.cos(phi_vals), r * np.sin(phi_vals), z_vals])
    return _project(pts)

def _scipy_lbfgs(pts0, time_limit, maxiter=2000):
    """
    Use scipy.optimize.minimize with L-BFGS-B on flattened coordinates.
    Normalizes points internally to stay on sphere.
    """
    start = time.time()
    x0 = pts0.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        energy, grad_r3 = _energy_grad(p)
        if energy == float('inf'):
            return 1e10, np.zeros_like(x)
        # Project gradient to tangent
        dot = np.sum(p * grad_r3, axis=1, keepdims=True)
        grad_t = grad_r3 - dot * p
        return energy, grad_t.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 5000, 'maxls': 30})
        x_opt = res.x.reshape(N, 3)
        x_opt = _project(x_opt)
        energy, _ = _energy_grad(x_opt)
        return x_opt, energy
    except Exception:
        return pts0, _energy_grad(pts0)[0]

def _high_precision_polish(pts0, time_limit, maxiter=500):
    """
    High-precision final polish using very small steps and tight tolerances.
    """
    start = time.time()
    pts = pts0.copy()
    energy, grad_r3 = _energy_grad(pts)
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    
    if np.linalg.norm(grad_t) < 1e-15:
        return pts, energy
    
    step_size = 1e-4
    min_step = 1e-16
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        gnorm = np.linalg.norm(grad_t)
        if gnorm < 1e-15:
            break
        
        # Gradient descent step
        direction = -grad_t / (gnorm + 1e-15)
        
        alpha = step_size
        accepted = False
        
        for _ in range(40):
            new_pts = _project(pts + alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy == float('inf'):
                alpha *= 0.5
                continue
            
            if new_energy < energy:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break
            continue
        
        dot_new = np.sum(new_pts * new_grad_r3, axis=1, keepdims=True)
        new_grad_t = new_grad_r3 - dot_new * new_pts
        
        pts = new_pts
        energy = new_energy
        grad_t = new_grad_t
        step_size = min(alpha * 1.3, 0.1)
    
    return pts, energy

def _optimize_single(pts, time_limit):
    """
    Optimize a single configuration using L-BFGS-B followed by high-precision polish.
    """
    start = time.time()
    best_pts = pts.copy()
    best_e, _ = _energy_grad(pts)
    
    # Stage 1: L-BFGS-B for fast convergence
    t1 = time_limit * 0.8
    pts, e = _scipy_lbfgs(pts, t1, maxiter=1000)
    if e < best_e:
        best_e, best_pts = e, pts.copy()
    
    # Stage 2: High-precision polish
    remaining = time_limit - (time.time() - start)
    if remaining > 1.0:
        pts, e = _high_precision_polish(pts, remaining - 0.5, maxiter=200)
        if e < best_e:
            best_e, best_pts = e, pts.copy()
    
    return best_pts, best_e

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - MARGIN
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_sobol_sphere(N, seed=42))
    
    # Random initializations
    for seed in [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024]:
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((N, 3))
        inits.append(_project(pts))
    
    # Perturbed Fibonacci
    for seed, strength in [(3, 0.03), (5, 0.08), (11, 0.15), (23, 0.25)]:
        rng = np.random.default_rng(seed)
        base = _fibonacci_sphere(N)
        pert = base + rng.standard_normal((N, 3)) * strength
        inits.append(_project(pert))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Basin Hopping / Multi-start optimization
    # We allocate time to each initialization. If we finish early, we can do more restarts.
    # Start with a generous budget per init, reducing as we go.
    
    n_inits = len(inits)
    # Reserve 20% of time for final polish of the best solution found
    search_time = (DEADLINE - MARGIN) * 0.8
    final_polish_time = (DEADLINE - MARGIN) * 0.2
    
    # We will try each initialization, and then potentially restart with perturbations of the best found
    current_best_pts = None
    current_best_e = float('inf')
    
    # First pass: Optimize all initializations
    # Estimate time per init: search_time / n_inits
    # But we want to be adaptive. Let's just iterate and check time.
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - final_polish_time:
            break
        
        # Allocate remaining search time / remaining inits
        remaining_search_time = deadline - final_polish_time - time.time()
        remaining_inits = n_inits - idx
        t_alloc = max(1.0, remaining_search_time / remaining_inits)
        
        # Cap t_alloc to avoid wasting time on one bad init if we have many left
        # But also ensure we give enough time for L-BFGS to converge
        t_alloc = min(t_alloc, 10.0) # Cap at 10s per init in early stages
        
        pts, energy = _optimize_single(init_pts, t_alloc)
        
        if energy < current_best_e:
            current_best_e = energy
            current_best_pts = pts.copy()
    
    # Phase 2: Basin Hopping restarts
    # If we have time left, perturb the current best and re-optimize
    if current_best_pts is not None:
        remaining = deadline - final_polish_time - time.time()
        if remaining > 5.0:
            rng = np.random.default_rng(12345)
            # Number of restarts we can afford
            # Each restart takes roughly 5-10s
            n_restarts = int(remaining / 8.0)
            n_restarts = max(1, min(n_restarts, 10))
            
            for i in range(n_restarts):
                if time.time() > deadline - final_polish_time:
                    break
                
                # Perturb the current best
                # Perturbation strength can be adaptive
                strength = 0.1 * (1.0 + 0.1 * i)
                perturbed = current_best_pts + rng.standard_normal((N, 3)) * strength
                perturbed = _project(perturbed)
                
                # Allocate remaining time / remaining restarts
                remaining_restarts = n_restarts - i
                t_alloc = max(1.0, (deadline - final_polish_time - time.time()) / remaining_restarts)
                t_alloc = min(t_alloc, 10.0)
                
                pts, energy = _optimize_single(perturbed, t_alloc)
                
                if energy < current_best_e:
                    current_best_e = energy
                    current_best_pts = pts.copy()
    
    # Phase 3: Final high-precision polish
    if current_best_pts is not None:
        remaining = deadline - time.time()
        if remaining > 1.0:
            pts, energy = _high_precision_polish(current_best_pts, remaining - 0.5, maxiter=500)
            if energy < current_best_e:
                current_best_e = energy
                current_best_pts = pts.copy()
    
    best_pts = current_best_pts
    best_energy = current_best_e
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    else:
        best_pts = _project(best_pts)
    
    # Validate
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        iu = np.triu_indices(N, 1)
        min_d = np.sqrt(d2[iu]).min()
        if min_d < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
