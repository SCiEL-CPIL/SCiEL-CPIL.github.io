"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_d2 = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _halton_sphere(n):
    """Generate Halton sequence points projected to unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        idx = i + 1
        pts[i, 0] = halton(idx, 2)
        pts[i, 1] = halton(idx, 3)
        pts[i, 2] = halton(idx, 5)
    
    # Scale to fit in unit ball, then project to sphere
    # Map [0,1]^3 to sphere via stereographic-like mapping
    # First center and scale
    pts = pts - 0.5
    # Project to sphere: normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_cg(pts, time_limit=10.0):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    e, g = _energy_and_grad(current)
    
    # Initial direction: negative gradient
    d = -g
    # Project d to tangent space at current
    dot_cd = np.sum(current * d, axis=1)
    d = d - dot_cd[:, np.newaxis] * current
    
    best_energy = e
    best_pts = current.copy()
    
    for it in range(200):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Line search along geodesic: exp_{current}(alpha * d)
        # For small alpha, exp_p(v) ≈ p + v + 0.5*(||v||^2*p - 2*<p,v>*v)/2 ... 
        # Simpler: normalize current + alpha*d
        alpha = 1.0
        accepted = False
        
        for ls in range(30):
            new_pts = current + alpha * d
            new_pts = _normalize(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            if new_e < e - 1e-14:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try smaller initial step
            alpha = 0.01
            for ls in range(30):
                new_pts = current + alpha * d
                new_pts = _normalize(new_pts)
                new_e = _log_energy_fast(new_pts)
                if new_e < e - 1e-14:
                    accepted = True
                    break
                alpha *= 0.5
        
        if not accepted:
            break
        
        e_new = new_e
        g_new = _energy_and_grad(new_pts)[1]
        
        # Project g_new to tangent space
        dot_ng = np.sum(new_pts * g_new, axis=1)
        g_new = g_new - dot_ng[:, np.newaxis] * new_pts
        
        # Conjugate gradient update (Polak-Ribiere)
        beta = np.sum(g_new * (g_new - g)) / max(np.sum(g * g), 1e-30)
        beta = max(beta, 0.0)
        
        d_new = -g_new + beta * d
        # Project d_new to tangent space at new_pts
        dot_nd = np.sum(new_pts * d_new, axis=1)
        d_new = d_new - dot_nd[:, np.newaxis] * new_pts
        
        # Check convergence
        g_norm = np.linalg.norm(g_new)
        if g_norm < 1e-14:
            current = new_pts
            e = e_new
            break
        
        current = new_pts
        e = e_new
        g = g_new
        d = d_new
        
        if e < best_energy:
            best_energy = e
            best_pts = current.copy()
    
    return best_pts, best_energy


def _dual_optimize(pts, time_limit=15.0):
    """Alternate between Cartesian projected gradient and Riemannian CG."""
    start = time.time()
    current = pts.copy()
    e = _log_energy_fast(current)
    best_e = e
    best_pts = current.copy()
    
    # Phase 1: Cartesian projected gradient descent
    step_size = 0.05
    for it in range(500):
        if time.time() - start > time_limit - 5.0:
            break
        
        e, grad = _energy_and_grad(current)
        g_norm = np.linalg.norm(grad)
        
        if g_norm < 1e-13:
            break
        
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            if new_e < e - 1e-13:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            e = new_e
            if alpha == step_size:
                step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-16:
                break
        
        if e < best_e:
            best_e = e
            best_pts = current.copy()
    
    # Phase 2: Riemannian conjugate gradient
    time_left = time_limit - (time.time() - start)
    if time_left > 2.0:
        cg_pts, cg_e = _riemannian_cg(current, time_limit=time_left)
        if cg_e < best_e:
            best_e = cg_e
            best_pts = cg_pts.copy()
    
    # Phase 3: More Cartesian steps if time remains
    time_left = time_limit - (time.time() - start)
    if time_left > 1.0:
        current = best_pts.copy()
        e = best_e
        step_size = 0.01
        for it in range(300):
            if time.time() - start > time_limit - 0.5:
                break
            
            e, grad = _energy_and_grad(current)
            g_norm = np.linalg.norm(grad)
            
            if g_norm < 1e-14:
                break
            
            alpha = step_size
            accepted = False
            for _ in range(25):
                new_pts = current - alpha * grad
                new_pts = _normalize(new_pts)
                new_e = _log_energy_fast(new_pts)
                
                if new_e < e - 1e-13:
                    accepted = True
                    break
                alpha *= 0.5
            
            if accepted:
                current = new_pts
                e = new_e
            else:
                step_size *= 0.5
                if step_size < 1e-17:
                    break
            
            if e < best_e:
                best_e = e
                best_pts = current.copy()
    
    return best_pts, best_e


def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    # Flatten points for L-BFGS-B
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        # Project gradient back to flat space
        return energy, grad.flatten()
    
    # Use L-BFGS-B with bounds to keep points reasonable
    # No explicit bounds needed since we normalize after each step
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 10000, 'maxls': 50})
    
    # Check time
    if time.time() - start > time_limit:
        return pts, _log_energy_fast(pts)
    
    # Extract optimized points
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-4
    
    for it in range(10000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        # Backtracking line search with very tight tolerance
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy


def _grad_descent(pts, time_limit=10.0):
    """Gradient descent with adaptive step size."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 0.01
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Backtracking line search
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-12:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            # Increase step size if we took a full step
            if alpha == step_size:
                step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-16:
                break
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    
    # Halton sequence initialization
    pts_halton = _halton_sphere(N)
    
    best_pts = pts_fib.copy()
    best_energy = _log_energy_fast(pts_fib)
    
    # Also track Halton energy
    halton_energy = _log_energy_fast(pts_halton)
    if halton_energy < best_energy:
        best_energy = halton_energy
        best_pts = pts_halton.copy()
    
    # Generate diverse initializations
    restarts = [
        ("fibonacci", pts_fib.copy()),
        ("halton", pts_halton.copy()),
    ]
    
    # Random initializations with different seeds
    for seed in range(8):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        restarts.append((f"random_{seed}", pts_rand))
    
    # Perturbed Fibonacci
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        noise_level = 0.02 * (seed + 1)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * noise_level
        pts_pert = _normalize(pts_pert)
        restarts.append((f"perturbed_{seed}", pts_pert))
    
    # Perturbed Halton
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 789 + 23)
        noise_level = 0.03 * (seed + 1)
        pts_pert_h = pts_halton + rng_local.standard_normal((N, 3)) * noise_level
        pts_pert_h = _normalize(pts_pert_h)
        restarts.append((f"halton_pert_{seed}", pts_pert_h))
    
    # Optimize each restart using dual-representation approach
    for name, init_pts in restarts:
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time() - 5
        if time_left < 5:
            break
        
        # Dual-representation optimization (Cartesian + Riemannian CG)
        opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=min(time_left, 15.0))
        
        # L-BFGS-B polish if time remains
        if time.time() < deadline - 5:
            lbfgs_time = min(deadline - time.time() - 3, 10.0)
            if lbfgs_time > 2.0:
                lbfgs_pts, lbfgs_energy = _lbfgs_optimize(opt_pts, time_limit=lbfgs_time)
                if lbfgs_energy < opt_energy:
                    opt_pts = lbfgs_pts
                    opt_energy = lbfgs_energy
        
        # Final polish
        if time.time() < deadline - 2:
            polish_time = min(deadline - time.time() - 1, 3.0)
            if polish_time > 0.3:
                polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                if polished_energy < opt_energy:
                    opt_pts = polished_pts
                    opt_energy = polished_energy
        
        # Update best
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 0.5, 5.0)
        if final_time > 0.3:
            final_pts, final_energy = _polish(best_pts, time_limit=final_time)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
