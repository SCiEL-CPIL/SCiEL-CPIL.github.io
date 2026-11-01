import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diffs, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-(pts[i,k] - pts[j,k]) / dists_sq[i,j])
    # = -pts[i,k] * sum_j(1/d2) + sum_j(pts[j,k] / d2)
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _energy_and_grad(pts):
    """Energy and gradient in one call, for speed."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms_safe)
    sin_norms_over_norms = np.sin(norms_safe) / norms_safe
    # For zero vectors, sin(0)/0 should be 0, but we handle via mask
    zero_mask = (norms < 1e-15)
    if np.any(zero_mask):
        sin_norms_over_norms[zero_mask, 0] = 0.0
        cos_norms[zero_mask, 0] = 1.0
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _lbfgs_sphere(pts0, max_iter=2000, m=10, time_limit=None, start_time=None, tol=1e-12):
    """L-BFGS optimization on the sphere with projected gradients."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    s_list = []
    y_list = []
    
    step_size = 0.5
    min_step = 1e-14
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            direction = -grad
        else:
            q = grad.copy()
            rhos = []
            alphas = []
            
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rho = 0.0
                else:
                    rho = 1.0 / si_yi
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            if len(s_list) > 0:
                last_s = s_list[-1]
                last_y = y_list[-1]
                sy = np.sum(last_s * last_y)
                if abs(sy) > 1e-16:
                    gamma = np.sum(last_s * last_s) / sy
                    if gamma < 1e-10 or gamma > 1e10:
                        gamma = 1.0
                else:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search with backtracking
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = _normalize(pts + alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _normalize(pts + alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy

def _riemannian_cg(pts0, max_iter=2000, time_limit=None, start_time=None, tol=1e-12):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # CG state
    p = -grad.copy()
    p = p - np.sum(p * pts, axis=1, keepdims=True) * pts  # project to tangent
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search along p using exponential map
        alpha = 0.5
        accepted = False
        for _ in range(50):
            new_pts = _exponential_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * p):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller step
            alpha = 0.01
            new_pts = _exponential_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # Update CG direction
        g_old = grad
        grad = new_grad
        pts = new_pts
        energy = new_energy
        
        # Polak-Ribiere beta
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        p_new = -grad + beta * p
        # Project to tangent space
        p_new = p_new - np.sum(p_new * pts, axis=1, keepdims=True) * pts
        
        # Check for restart condition
        if np.sum(grad * p) > 0:
            p = -grad
            p = p - np.sum(p * pts, axis=1, keepdims=True) * pts
        else:
            p = p_new
        
        prev_grad = grad.copy()
    
    return pts, energy

def _simulated_annealing(pts0, time_limit=None, start_time=None, T0=0.1, T1=1e-6, max_iter=5000):
    """Simulated annealing to escape local minima."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Temperature schedule
        T = T0 * (T1 / T0) ** (it / max_iter)
        
        # Generate random perturbation
        rng = np.random.default_rng(hash((it, it*7)) % 2**32)
        noise = rng.standard_normal((N, 3)) * 0.05
        # Project to tangent space
        noise = noise - np.sum(noise * pts, axis=1, keepdims=True) * pts
        
        # Propose new point
        new_pts = _exponential_map(pts, noise)
        new_energy, _ = _energy_and_grad(new_pts)
        
        # Accept or reject
        delta_E = new_energy - energy
        if delta_E < 0 or np.random.default_rng(hash((it*13, it*17)) % 2**32).random() < math.exp(-delta_E / T):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
    
    return best_pts, best_energy

def _polish(pts0, time_limit=None, start_time=None, tol=1e-14):
    """High-precision polish with very small steps."""
    if start_time is None:
        start_time = time.time()
    
    current = pts0.copy()
    energy, grad = _energy_and_grad(current)
    step_size = 1e-4
    
    for it in range(10000):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            grad = new_grad
            step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_energy = float('inf')
    best_points = None
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Random uniform on sphere
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 4. Slightly different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        # Phase 1: L-BFGS for coarse optimization
        remaining = deadline - time.time()
        pts_lbfgs, energy_lbfgs = _lbfgs_sphere(init_pts, max_iter=800, time_limit=remaining * 0.5, start_time=start_time)
        if energy_lbfgs < best_energy:
            best_energy = energy_lbfgs
            best_points = pts_lbfgs.copy()
        
        # Phase 2: Riemannian CG for fine optimization
        if time.time() < deadline - 15:
            remaining = deadline - time.time()
            pts_cg, energy_cg = _riemannian_cg(pts_lbfgs, max_iter=500, time_limit=remaining * 0.4, start_time=start_time)
            if energy_cg < best_energy:
                best_energy = energy_cg
                best_points = pts_cg.copy()
        
        # Phase 3: Simulated annealing to escape local minima
        if time.time() < deadline - 10:
            remaining = deadline - time.time()
            pts_sa, energy_sa = _simulated_annealing(best_points, time_limit=remaining * 0.3, start_time=start_time)
            if energy_sa < best_energy:
                best_energy = energy_sa
                best_points = pts_sa.copy()
        
        # Phase 4: Final L-BFGS polish
        if time.time() < deadline - 5:
            remaining = deadline - time.time()
            pts_polish, energy_polish = _lbfgs_sphere(best_points, max_iter=300, time_limit=remaining * 0.5, start_time=start_time, tol=1e-13)
            if energy_polish < best_energy:
                best_energy = energy_polish
                best_points = pts_polish.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        remaining = deadline - time.time()
        pts_final, energy_final = _polish(best_points, time_limit=remaining * 0.8, start_time=start_time, tol=1e-15)
        if energy_final < best_energy:
            best_energy = energy_final
            best_points = pts_final.copy()
    
    if best_points is None:
        return fib_pts  # Fallback to initial
    
    return best_points
