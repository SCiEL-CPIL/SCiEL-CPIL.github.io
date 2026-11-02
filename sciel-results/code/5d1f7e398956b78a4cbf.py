"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE_SECONDS = 140.0
EPS = 1e-14

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the Euclidean gradient w.r.t. pts.
    Uses dot-product formulation for speed.
    """
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, EPS)
    dist = np.sqrt(d2)
    mask = ~np.eye(N, dtype=bool)
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    inv_d2 = np.where(mask, 1.0 / d2, 0.0)
    weight_sums = np.sum(inv_d2, axis=1, keepdims=True)
    weighted_pts = inv_d2 @ pts
    grad = weighted_pts - weight_sums * pts
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=5000, tol=1e-13, time_limit=None):
    """
    Riemannian Conjugate Gradient with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.02
    start_time = time.time()
    
    d = -g_t.copy()
    g_t_prev = g_t.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        g_t_norm = np.linalg.norm(g_t)
        if g_t_norm < tol:
            break
        
        if np.sum(d * g_t) > 0:
            d = -g_t.copy()
        
        alpha = lr
        improved = False
        g_t_norm_sq = np.sum(g_t * g_t)
        for _ in range(40):
            new_pts = _retract(pts, alpha * d)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * d):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-14:
                break
            d = -g_t.copy()
            g_t_prev = g_t.copy()
            continue
        
        _, grad = _energy_and_grad(pts)
        g_t_new = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        g_diff = g_t_new - g_t_prev
        pr_coeff = np.sum(g_t_new * g_diff) / max(np.sum(g_t_prev * g_t_prev), 1e-30)
        if pr_coeff < 0:
            pr_coeff = 0.0
        d = -g_t_new + pr_coeff * d
        g_t_prev = g_t
        g_t = g_t_new
        
        lr = max(0.001, lr * 0.9995)
        
    return best_pts, best_energy

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _repulsion_refine(pts, n_sweeps=5, time_limit=None):
    """
    Repulsion-based Gauss-Seidel refinement.
    For each point, move it to reduce local energy contribution.
    This can escape local minima that gradient flow gets stuck in.
    """
    pts = pts.copy()
    start_time = time.time()
    
    for sweep in range(n_sweeps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        for i in range(N):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            # Compute gradient for point i only
            diff = pts[i] - np.delete(pts, i, axis=0)  # (N-1, 3)
            d2 = np.sum(diff * diff, axis=1)
            d2 = np.maximum(d2, EPS)
            inv_d2 = 1.0 / d2
            # Gradient for point i: sum_j (pts[i] - pts[j]) / d^2
            # = pts[i] * sum(inv_d2) - sum(pts[j] / d^2)
            gi = pts[i] * np.sum(inv_d2) - np.sum(np.delete(pts, i, axis=0) * inv_d2[:, None], axis=0)
            # Project to tangent
            radial = np.dot(gi, pts[i])
            gi_tan = gi - radial * pts[i]
            # Small step
            step = 0.005
            new_pt = pts[i] - step * gi_tan
            new_pt = new_pt / np.linalg.norm(new_pt)
            # Accept if energy improves
            old_e, _ = _energy_and_grad(pts)
            pts[i] = new_pt
            new_e, _ = _energy_and_grad(pts)
            if new_e > old_e:
                pts[i] = pts[i] + step * gi_tan  # revert
                pts[i] = pts[i] / np.linalg.norm(pts[i])
    
    return pts

def _polish_lbfgs(pts, max_iter=500, gtol=1e-16):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-18, 'maxls': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _polish_bfgs(pts, max_iter=200, gtol=1e-16):
    """Polish using BFGS with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def construct():
    start_time = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(('fib', _fibonacci_spiral(N)))
    
    # 2. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 3. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.05, 10), (0.1, 11), (0.15, 12), (0.2, 13), (0.02, 14), (0.08, 15)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 20:
            break
        
        # Riemannian Conjugate Gradient
        remaining = DEADLINE_SECONDS - 15 - (time.time() - start_time)
        if remaining > 2:
            pts_opt, energy_opt = _optimize_sphere(init_pts, max_iters=2000, time_limit=remaining - 2)
        else:
            pts_opt, energy_opt = init_pts, _energy_and_grad(init_pts)[0]
        
        # L-BFGS-B Polish
        if time.time() - start_time < DEADLINE_SECONDS - 15:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=300, gtol=1e-15)
        
        # BFGS Polish
        if time.time() - start_time < DEADLINE_SECONDS - 10:
            pts_opt2, energy_opt2 = _polish_bfgs(pts_opt, max_iter=150, gtol=1e-15)
            if energy_opt2 < energy_opt:
                pts_opt, energy_opt = pts_opt2, energy_opt2
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Repulsion refinement on the best result to escape local minima
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 15:
        rep_pts = _repulsion_refine(best_pts, n_sweeps=3, time_limit=DEADLINE_SECONDS - 12)
        rep_e = _energy_and_grad(rep_pts)[0]
        if rep_e < best_energy:
            best_pts = rep_pts
            best_energy = rep_e
        # Polish the repulsion result
        if time.time() - start_time < DEADLINE_SECONDS - 10:
            polished, pe = _polish_lbfgs(rep_pts, max_iter=300, gtol=1e-15)
            if pe < best_energy:
                best_pts = polished
                best_energy = pe
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 10:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=800, gtol=1e-17)
        # Second polish pass with BFGS
        if time.time() - start_time < DEADLINE_SECONDS - 5:
            best_pts2, best_energy2 = _polish_bfgs(best_pts, max_iter=300, gtol=1e-17)
            if best_energy2 < best_energy:
                best_pts, best_energy = best_pts2, best_energy2
        # Third polish: another L-BFGS pass with very tight tolerance
        if time.time() - start_time < DEADLINE_SECONDS - 3:
            best_pts3, best_energy3 = _polish_lbfgs(best_pts, max_iter=500, gtol=1e-18)
            if best_energy3 < best_energy:
                best_pts, best_energy = best_pts3, best_energy2 if best_energy3 < best_energy else best_energy
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] = best_pts[i] + 1e-8 * best_pts[j]
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
