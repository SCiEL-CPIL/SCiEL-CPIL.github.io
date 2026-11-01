"""
Dual-representation optimization for minimal logarithmic energy of 150 points on sphere.
Uses Riemannian L-BFGS and enhanced basin hopping.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return np.sum(np.log(1.0 / d))

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _riemannian_lbfgs(pts, time_limit=20.0, m=10):
    """
    Riemannian L-BFGS optimization on the sphere.
    """
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS memory
    s_hist = [] # steps in tangent space
    y_hist = [] # gradient differences in tangent space
    rho_hist = []
    
    # Initial direction
    d = -grad.copy()
    
    for it in range(200):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Line search (Armijo)
        alpha = 1.0
        c1 = 1e-4
        slope = np.sum(grad * d)
        
        if slope > 0:
            # Restart if not a descent direction
            d = -grad.copy()
            slope = -np.sum(grad * grad)
            
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad.copy()
        best_alpha = 0.0
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
                
            # Armijo condition
            if new_energy <= energy + c1 * alpha * slope:
                break
                
            alpha *= 0.5
            
        if best_alpha == 0.0:
            break
            
        # Update history
        s_k = best_alpha * d
        y_k = best_grad - grad
        
        # Ensure y_k and s_k are in tangent space (they should be, but numerical errors can creep in)
        # Project y_k
        dot_y = np.sum(y_k * best_pts, axis=1)
        y_k_proj = y_k - dot_y[:, np.newaxis] * best_pts
        
        rho_k = 1.0 / np.sum(s_k * y_k_proj)
        if np.sum(s_k * y_k_proj) < 1e-12:
            rho_k = 0.0
            
        s_hist.append(s_k)
        y_hist.append(y_k_proj)
        rho_hist.append(rho_k)
        
        if len(s_hist) > m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
            
        # Update current
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break
            
        # Compute new direction using L-BFGS two-loop recursion
        # We work in the tangent space at the new point
        # Note: s_k and y_k are in tangent space at old point (approx)
        # For simplicity, we use the standard L-BFGS update assuming small steps
        # A more rigorous approach would use parallel transport, but for small steps, 
        # direct use in the new tangent space is often sufficient.
        
        # Identify tangent space at current
        # d_new is in tangent space at current
        q = -grad.copy()
        
        # Two-loop recursion
        alpha_coeffs = [0.0] * len(s_hist)
        for i in range(len(s_hist) - 1, -1, -1):
            alpha_coeffs[i] = rho_hist[i] * np.sum(s_hist[i] * q)
            q -= alpha_coeffs[i] * y_hist[i]
            
        r = q
        # Initial Hessian approximation H0 = I (in tangent space)
        # We don't have an explicit H0, so we just use q as the base
        
        for i in range(len(s_hist)):
            beta = rho_hist[i] * np.sum(y_hist[i] * r)
            r += s_hist[i] * (alpha_coeffs[i] - beta)
            
        d = r
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
    return current, energy

def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy > energy:
                break
            
            alpha *= 0.8
        
        if best_alpha == 0.0:
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy

def _polish(pts, time_limit=5.0):
    """High-precision polish using Riemannian gradient steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-3
    
    for it in range(10000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(20):
            new_pts = _exponential_map(current, -alpha * grad)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-15:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.05, 0.05)
        else:
            step_size *= 0.5
            if step_size < 1e-16:
                break
    
    return current, energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    initializations = []
    
    # 1. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())
    
    # 2. Halton sequence
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())
    
    # 3. Random initializations
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
        
    # 4. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
        
    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        time_left = deadline - time.time() - 4
        if time_left < 8:
            break
        
        try:
            # Phase 1: Riemannian L-BFGS
            current, energy = _riemannian_lbfgs(init_pts, time_limit=min(time_left * 0.4, 20.0))
            
            # Phase 2: Riemannian CG for fine tuning
            time_left = deadline - time.time() - 4
            if time_left > 5:
                current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.4, 10.0))
                
            # Phase 3: Basin Hopping
            time_left = deadline - time.time() - 4
            if time_left > 3:
                rng = np.random.default_rng(42 + idx)
                for bh_iter in range(5):
                    if time.time() - start_time > deadline - 2:
                        break
                    
                    _, grad = _energy_and_grad(current)
                    grad_norm = np.linalg.norm(grad)
                    scale = 0.05 * max(0.1, min(1.0, 1.0 / (1.0 + grad_norm * 10)))
                    
                    perturbation = rng.standard_normal((N, 3))
                    dot_prod = np.sum(perturbation * current, axis=1, keepdims=True)
                    perturbation -= dot_prod * current
                    perturbation *= scale
                    
                    new_pts = _exponential_map(current, perturbation)
                    new_energy = _log_energy_fast(new_pts)
                    
                    # Metropolis acceptance
                    if new_energy < energy or np.exp(-(new_energy - energy) / 0.1) > rng.random():
                        current = new_pts
                        energy = new_energy
                        
                    # Quick local opt
                    time_left = deadline - time.time() - 2
                    if time_left > 2:
                        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.3, 3.0))
                        
            # Phase 4: Final Polish
            time_left = deadline - time.time() - 1
            if time_left > 1:
                current, energy = _polish(current, time_limit=min(time_left, 5.0))
                
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
                
        except Exception:
            pass

    best_pts = _normalize(best_pts)
    return best_pts
