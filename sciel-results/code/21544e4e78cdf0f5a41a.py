"""Minimal logarithmic energy of 150 points on the sphere - Riemannian CG + L-BFGS + Basin Hopping."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _energy_and_grad(pts):
    """Vectorized energy and gradient computation on the sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Check for too-close points
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    if np.any(d_upper < 1e-12):
        return float('inf'), np.zeros_like(pts)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    log_dists = np.log(dists)
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient: grad_i E = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = weighted_sum - pts * sum_inv[:, np.newaxis]
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map(pts, v):
    """Exponential map on sphere."""
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    return pts * cos_t + v * (sin_t / safe_norms)


def _log_map(pts, v):
    """Logarithmic map on sphere (inverse of exp_map)."""
    # v is the tangent vector at pts
    # Returns the point on the sphere
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    # The logarithmic map from a point p with tangent vector v is:
    # log_p(v) = (theta / ||v||) * v where theta = arccos(<p, q>)
    # But we want the inverse: given p and tangent v, find q
    # Actually, exp_map(p, v) gives q. The log map is the inverse.
    # For our purposes, we just need exp_map.
    return pts * cos_t + v * (sin_t / safe_norms)


def _riemannian_cg(pts0, max_iter=500, time_limit=None, start_time=None, tol=1e-12):
    """Riemannian conjugate gradient optimization on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # Initial direction
    d = -grad
    
    # Project to tangent space
    dot_d = np.sum(d * pts, axis=1)
    d -= dot_d[:, np.newaxis] * pts
    
    beta = 0.0
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search
        alpha = 1.0
        accepted = False
        for _ in range(40):
            new_pts = _exp_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * d):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller steps
            for trial_alpha in [0.01, 0.001, 0.0001]:
                new_pts = _exp_map(pts, trial_alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    accepted = True
                    alpha = trial_alpha
                    break
        
        if not accepted:
            break
        
        # Update beta (Polak-Ribiere)
        grad_dot = np.sum(grad * new_grad)
        old_grad_norm_sq = np.sum(grad * grad)
        beta = max(0.0, grad_dot / old_grad_norm_sq) if old_grad_norm_sq > 1e-16 else 0.0
        
        # Update direction
        d = -new_grad + beta * d
        
        # Project to tangent space
        dot_d = np.sum(d * pts, axis=1)
        d -= dot_d[:, np.newaxis] * pts
        
        # Check if direction is a descent direction
        if np.sum(d * new_grad) > 0:
            d = -new_grad
            dot_d = np.sum(d * pts, axis=1)
            d -= dot_d[:, np.newaxis] * pts
            beta = 0.0
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
    
    return pts, energy


def _lbfgs_sphere(pts0, max_iter=300, m=20, time_limit=None, start_time=None, tol=1e-14):
    """L-BFGS optimization on the sphere."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
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
                rho = 1.0 / si_yi if abs(si_yi) > 1e-16 else 0.0
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            last_s = s_list[-1]
            last_y = y_list[-1]
            sy = np.sum(last_s * last_y)
            gamma = np.sum(last_s * last_s) / sy if abs(sy) > 1e-16 else 1.0
            if gamma < 1e-10 or gamma > 1e10:
                gamma = 1.0
            
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[i]
                alpha_i = alphas[i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Project to tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = _exp_map(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy + 1e-4 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(step_size * 0.01, min_step)
            new_pts = _exp_map(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
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


def _basin_hopping(pts0, max_iter=50, time_limit=None, start_time=None, seed=42, 
                   perturb_scale=0.1, cg_max_iter=100):
    """Basin hopping: perturb, then optimize with CG."""
    if start_time is None:
        start_time = time.time()
    
    rng = np.random.default_rng(seed)
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Random perturbation
        noise = rng.standard_normal((N, 3)) * perturb_scale
        # Project noise to tangent space
        dot_noise = np.sum(noise * pts, axis=1)
        noise -= dot_noise[:, np.newaxis] * pts
        
        perturbed_pts = _exp_map(pts, noise)
        
        # Optimize with CG
        optimized_pts, optimized_energy = _riemannian_cg(
            perturbed_pts,
            max_iter=cg_max_iter,
            time_limit=2.0,
            start_time=time.time()
        )
        
        if np.isfinite(optimized_energy) and optimized_energy < best_energy:
            best_energy = optimized_energy
            best_pts = optimized_pts.copy()
            pts = optimized_pts.copy()
            energy = optimized_energy
    
    return best_pts, best_energy


def _high_precision_polish(pts0, max_iter=100, time_limit=None, start_time=None, tol=1e-16):
    """High-precision polish using L-BFGS with very tight tolerance."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # Use L-BFGS with very tight tolerance
    result = _lbfgs_sphere(
        pts,
        max_iter=max_iter,
        m=25,
        time_limit=time_limit,
        start_time=start_time,
        tol=tol
    )
    
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Riemannian CG from multiple initial configurations
    phase1_deadline = start_time + (deadline - start_time) * 0.3
    
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
    ]
    
    # Add random inits
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9876]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # Perturbed Fibonacci
    for seed, strength in [(7, 0.05), (99, 0.1), (1234, 0.03), (5678, 0.07)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * strength
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    for idx, init_pts in enumerate(inits):
        if time.time() > phase1_deadline:
            break
        
        remaining = phase1_deadline - time.time()
        time_alloc = min(remaining / max(1, len(inits) - idx), 8.0)
        time_alloc = max(time_alloc, 2.0)
        
        pts, energy = _riemannian_cg(
            init_pts,
            max_iter=300,
            time_limit=time_alloc,
            start_time=time.time(),
            tol=1e-12
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Basin hopping to escape local minima
    if best_pts is not None and time.time() < deadline - 40:
        remaining = deadline - time.time() - 30
        if remaining > 10:
            best_pts, best_energy = _basin_hopping(
                best_pts,
                max_iter=30,
                time_limit=remaining,
                start_time=time.time(),
                seed=42,
                perturb_scale=0.1,
                cg_max_iter=80
            )
    
    # Phase 3: L-BFGS polish on best result
    if best_pts is not None and time.time() < deadline - 20:
        remaining = deadline - time.time() - 15
        if remaining > 5:
            best_pts, best_energy = _lbfgs_sphere(
                best_pts,
                max_iter=200,
                m=20,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-14
            )
    
    # Phase 4: High-precision polish for final convergence
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            best_pts, best_energy = _high_precision_polish(
                best_pts,
                max_iter=50,
                time_limit=remaining,
                start_time=time.time(),
                tol=1e-16
            )
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Verify validity
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
