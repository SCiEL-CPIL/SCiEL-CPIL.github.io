"""Minimal logarithmic energy of 150 points on the sphere - Riemannian L-BFGS optimizer."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
MARGIN = 6.0  # safety margin in seconds

def _energy_grad(pts):
    """
    Vectorized logarithmic energy and gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient (R3) = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Returns (energy, grad) where grad is (N,3) in R3 (caller projects to tangent).
    """
    n = N
    # Pairwise differences: (n, n, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)  # (n, n)
    np.fill_diagonal(d2, 1.0)
    
    d = np.sqrt(d2)
    
    # Energy using fsum
    iu = np.triu_indices(n, 1)
    d_upper = d[iu]
    if np.any(d_upper < 1e-14):
        return float('inf'), np.zeros((n, 3))
    
    terms = np.log(1.0 / d_upper)
    energy = math.fsum(terms.tolist())
    
    # Gradient: grad_i = sum_j (x_i - x_j) / d_ij^2
    # = x_i * sum_j 1/d_ij^2 - sum_j x_j / d_ij^2
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = x_i * col_sum_i - (inv_d2 @ pts)_i
    col_sum = np.sum(inv_d2, axis=1)  # sum over j of 1/d_ij^2
    weighted = inv_d2 @ pts  # sum_j inv_d2[i,j] * x_j
    grad = pts * col_sum[:, None] - weighted
    
    return energy, grad

def _project(pts):
    """Project to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _exp_map(pts, v):
    """Exponential map: move along geodesic by tangent vector v."""
    # Project v to tangent space
    dot = np.sum(pts * v, axis=1, keepdims=True)
    v_t = v - dot * pts
    
    vnorm = np.linalg.norm(v_t, axis=1, keepdims=True)
    vnorm_safe = np.where(vnorm < 1e-15, 1.0, vnorm)
    
    cos_t = np.cos(vnorm_safe)
    sin_t = np.sin(vnorm_safe) / vnorm_safe
    sin_t = np.where(vnorm < 1e-15, 0.0, sin_t)
    
    new_pts = pts * cos_t + v_t * sin_t
    return _project(new_pts)

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

def _riemannian_lbfgs(pts, time_limit, maxiter=1000, tol=1e-12):
    """
    Riemannian L-BFGS on the sphere using exponential map.
    Uses scipy's L-BFGS-B in a retraction-based framework.
    """
    start = time.time()
    n = N
    
    energy, grad_r3 = _energy_grad(pts)
    # Project gradient to tangent space
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    
    if np.linalg.norm(grad_t) < tol:
        return pts, energy
    
    # Use scipy minimize with a custom objective that operates on flat R3N
    # but we use retraction to stay on manifold
    def obj_grad(flat):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(flat)
        p = _project(flat.reshape(n, 3))
        e, g = _energy_grad(p)
        if e == float('inf'):
            return e, np.zeros_like(flat)
        # Project gradient to tangent, then back to R3 for scipy
        d = np.sum(p * g, axis=1, keepdims=True)
        g_t = g - d * p
        return e, g_t.flatten()
    
    res = minimize(
        lambda x: obj_grad(x)[0],
        pts.flatten(),
        jac=lambda x: obj_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxfun': 5000, 'disp': False}
    )
    
    final_pts = _project(res.x.reshape(n, 3))
    final_energy, _ = _energy_grad(final_pts)
    return final_pts, final_energy

def _riemannian_cg(pts, time_limit, maxiter=2000, tol=1e-12):
    """Riemannian conjugate gradient on sphere."""
    start = time.time()
    n = N
    
    energy, grad_r3 = _energy_grad(pts)
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    gnorm = np.linalg.norm(grad_t)
    
    if gnorm < tol:
        return pts, energy
    
    d = -grad_t / gnorm
    alpha = 0.5
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        # Backtracking line search
        best_e = energy
        best_p = pts
        best_g = grad_t
        a = alpha
        improved = False
        
        for _ls in range(30):
            new_pts = _exp_map(pts, a * d)
            new_e, new_g_r3 = _energy_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_p = new_pts
                ddot = np.sum(new_pts * new_g_r3, axis=1, keepdims=True)
                best_g = new_g_r3 - ddot * new_pts
                improved = True
                break
            a *= 0.5
        
        if not improved:
            alpha *= 0.1
            if alpha < 1e-14:
                break
            continue
        
        # Update
        old_gnorm = gnorm
        new_gnorm = np.linalg.norm(best_g)
        
        if new_gnorm < tol:
            pts, energy = best_p, best_e
            break
        
        # Polak-Ribiere
        beta = np.sum(best_g * (best_g - grad_t)) / (old_gnorm * old_gnorm + 1e-15)
        beta = max(0.0, beta)
        
        d = -best_g + beta * d * (old_gnorm / (new_gnorm + 1e-15))
        
        pts, energy, grad_t = best_p, best_e, best_g
        gnorm = new_gnorm
        alpha = min(0.5, alpha * 1.2)
    
    return pts, energy

def _optimize(pts, time_limit):
    """
    Multi-stage optimization: CG for fast initial descent, then L-BFGS for polish.
    """
    start = time.time()
    best_pts = pts.copy()
    best_e, _ = _energy_grad(pts)
    
    # Stage 1: CG for fast descent (use 40% of time)
    t1 = time_limit * 0.4
    pts, e = _riemannian_cg(pts, t1, maxiter=3000, tol=1e-12)
    if e < best_e:
        best_e, best_pts = e, pts.copy()
    
    # Stage 2: L-BFGS for polish (use remaining time minus margin)
    remaining = time_limit - (time.time() - start)
    if remaining > 3.0:
        pts, e = _riemannian_lbfgs(pts, remaining - 1.0, maxiter=500, tol=1e-13)
        if e < best_e:
            best_e, best_pts = e, pts.copy()
    
    # Stage 3: Final CG micro-polish
    remaining = time_limit - (time.time() - start)
    if remaining > 2.0:
        pts, e = _riemannian_cg(pts, remaining - 0.5, maxiter=500, tol=1e-14)
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
    
    # Random initializations with different seeds
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
    
    # Spherical designs / structured
    # Ring-based: distribute points in latitude rings
    n_rings = 12
    pts_ring = []
    for k in range(n_rings):
        z = 1.0 - 2.0 * (k + 0.5) / n_rings
        r = np.sqrt(max(0.0, 1.0 - z * z))
        n_per_ring = N // n_rings + (1 if k < N % n_rings else 0)
        angles = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
        for a in angles:
            pts_ring.append([r * np.cos(a), r * np.sin(a), z])
    if len(pts_ring) == N:
        inits.append(_project(np.array(pts_ring)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Coarse optimization on all inits
    # Allocate time: give each init a budget, but cap total
    n_inits = len(inits)
    # Reserve 50% of time for deep polish on best
    coarse_budget = (DEADLINE - MARGIN) * 0.5
    per_init_time = coarse_budget / n_inits
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - (DEADLINE - MARGIN) * 0.5:
            break
        
        remaining_for_coarse = deadline - (DEADLINE - MARGIN) * 0.5 - time.time()
        if remaining_for_coarse < 2.0:
            break
        
        t_alloc = min(per_init_time, remaining_for_coarse)
        if t_alloc < 1.0:
            break
        
        pts, energy = _optimize(init_pts, t_alloc)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Deep polish on best result with remaining time
    if best_pts is not None:
        remaining = deadline - time.time()
        if remaining > 5.0:
            best_pts, best_energy = _optimize(best_pts, remaining - 1.0)
        
        # Phase 3: Ultra-fine polish
        remaining = deadline - time.time()
        if remaining > 3.0:
            # Multiple short CG passes for micro-improvements
            for _ in range(3):
                rem = deadline - time.time()
                if rem < 1.5:
                    break
                pts, e = _riemannian_cg(best_pts, rem - 0.5, maxiter=200, tol=1e-15)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts.copy()
    
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
        # Check minimum distance
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        iu = np.triu_indices(N, 1)
        min_d = np.sqrt(d2[iu]).min()
        if min_d < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
