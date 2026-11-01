"""Dual-representation optimization for minimal logarithmic energy of 150 points on sphere.

Combines:
- Riemannian CG with exponential map (fast manifold convergence)
- L-BFGS-B with projected gradient (precise local optimization)
- Riemannian L-BFGS (Stiefel-style, uses Hessian memory for faster convergence)
- Newton-based polish with Hessian approximation
- Multiple diverse initializations including farthest-point sampling
- Adaptive time allocation and final high-precision polish
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy_exact(pts):
    """Compute logarithmic energy using math.fsum for exact scoring."""
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
    """Vectorized log energy and tangent-space gradient."""
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
    
    # Energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


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
    return _normalize(pts)


def _farthest_point_init(n, seed=0):
    """Farthest-point sampling initialization for better spread."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, n):
        candidates = rng.standard_normal((1000, 3))
        candidates = _normalize(candidates)
        # Compute minimum distance from each candidate to existing points
        min_dists = np.full(len(candidates), np.inf)
        for j in range(i):
            dists = np.sum((candidates - pts[j]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
        best_idx = np.argmax(min_dists)
        pts[i] = candidates[best_idx]
    
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


def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search with Armijo condition
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            alpha = 1e-3
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
            if beta > 1.0:
                beta = 0.0
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _riemannian_lbfgs(pts, time_limit=25.0, m=10):
    """Riemannian L-BFGS optimization using exponential map and Hessian memory."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    S = []  # step vectors
    Y = []  # gradient difference vectors
    
    d = -grad.copy()
    
    for it in range(3000):
        if time.time() - start > time_limit - 0.5:
            break
        
        # L-BFGS two-loop recursion
        r = grad.copy()
        alphas = []
        rhos = []
        
        for k in range(len(S) - 1, -1, -1):
            s_k = S[k].flatten()
            y_k = Y[k].flatten()
            ys = np.sum(y_k * s_k)
            rho_k = 1.0 / ys if ys > 1e-12 else 0.0
            alpha_k = rho_k * np.sum(s_k * r)
            alphas.append(alpha_k)
            r = r - alpha_k * y_k
            rhos.append(rho_k)
        
        alphas.reverse()
        rhos.reverse()
        
        if len(S) > 0:
            s_last = S[-1].flatten()
            y_last = Y[-1].flatten()
            ys = np.sum(y_last * s_last)
            gamma = np.sum(s_last * y_last) / ys if ys > 1e-12 else 1.0
        else:
            gamma = 1.0
        
        q = gamma * r
        
        for k in range(len(S)):
            s_k = S[k].flatten()
            y_k = Y[k].flatten()
            beta_k = rhos[k] * np.sum(y_k * q)
            q = q + (alphas[k] - beta_k) * s_k
        
        d = -q.reshape(N, 3)
        
        # Project to tangent space
        dot_prod = np.sum(d * current, axis=1)
        d -= dot_prod[:, np.newaxis] * current
        
        # Backtracking line search
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad.copy()
        best_alpha = 0.0
        
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            alpha = 1e-3
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha
                    break
                alpha *= 0.5
            if best_alpha == 0.0:
                break
        
        # Update L-BFGS history
        s_vec = (best_pts - current).flatten()
        y_vec = (best_grad - grad).flatten()
        S.append(s_vec)
        Y.append(y_vec)
        if len(S) > m:
            S.pop(0)
            Y.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14,
                            'maxfun': 20000, 'maxls': 100})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish using Newton-like steps with Hessian approximation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        n = len(current)
        eps = 1e-5
        H_approx = np.zeros((n * 3, n * 3))
        
        rng_local = np.random.default_rng(42 + it)
        for _ in range(3):
            v = rng_local.standard_normal((n, 3))
            dot = np.sum(v * current, axis=1)
            v -= dot[:, np.newaxis] * current
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-15:
                v = v / v_norm
                vp = _normalize(current + eps * v)
                e_plus, _ = _energy_and_grad(vp)
                vm = _normalize(current - eps * v)
                e_minus, _ = _energy_and_grad(vm)
                hessian_v = (e_plus - 2 * e + e_minus) / (eps * eps)
                H_approx += hessian_v * np.outer(v.flatten(), v.flatten())
        
        try:
            reg = 1e-6
            H_reg = H_approx + reg * np.eye(n * 3)
            step = np.linalg.solve(H_reg, -grad.flatten()).reshape(n, 3)
            
            dot = np.sum(step * current, axis=1)
            step -= dot[:, np.newaxis] * current
            
            step_norm = np.linalg.norm(step)
            if step_norm > 0.1:
                step = step / step_norm * 0.1
        except:
            step = -grad / (grad_norm + 1e-15)
            step_norm = np.linalg.norm(step)
            if step_norm > 0.01:
                step = step / step_norm * 0.01
        
        alpha = 1.0
        accepted = False
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * step)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-14 * alpha * grad_norm * np.linalg.norm(step):
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
        else:
            alpha = 0.01
            for ls in range(20):
                new_pts = _exponential_map(current, alpha * (-grad / (grad_norm + 1e-15)))
                new_energy = _log_energy_fast(new_pts)
                if new_energy < e:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
    
    return current, energy


def _dual_optimize(pts, time_limit=40.0):
    """Alternate between Riemannian L-BFGS, CG, and Newton polish."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Phase 1: Riemannian L-BFGS for coarse optimization (uses Hessian memory)
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _riemannian_lbfgs(current, time_limit=min(time_left * 0.4, 18.0))
    
    # Phase 2: Riemannian CG for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.4, 12.0))
    
    # Phase 3: Newton-based polish
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        current, energy = _polish(current, time_limit=min(time_left * 0.5, 8.0))
    
    # Phase 4: Final Riemannian L-BFGS micro-polish
    time_left = time_limit - (time.time() - start)
    if time_left > 2:
        current, energy = _riemannian_lbfgs(current, time_limit=min(time_left, 6.0))
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib.copy())
    
    # 2. Halton sequence
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton.copy())
    
    # 3. Farthest-point sampling (better spread)
    for seed in range(2):
        pts_fps = _farthest_point_init(N, seed=seed)
        initializations.append(pts_fps)
    
    # 4. Random initializations
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 5. Uniform random on sphere (Marsaglia's method)
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 321 + 99)
        pts_uniform = []
        for _ in range(N):
            while True:
                u1 = rng_local.uniform(-1, 1)
                u2 = rng_local.uniform(-1, 1)
                s = u1 * u1 + u2 * u2
                if s < 1.0 and s > 1e-15:
                    break
            scale = 2.0 * np.sqrt(1.0 - s)
            pts_uniform.append([u1 * scale, u2 * scale, 1.0 - s])
        pts_uniform = np.array(pts_uniform)
        pts_uniform = _normalize(pts_uniform)
        initializations.append(pts_uniform)
    
    # 6. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 7. Perturbed Halton
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = pts_halton + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 8. Different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)
    
    # Optimize each initialization with dual-representation approach
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time() - 3
        if time_left < 8:
            break
        
        remaining = len(initializations) - idx
        base_time = min(time_left, max(12.0, 50.0 / remaining))
        
        try:
            opt_pts, opt_energy = _dual_optimize(init_pts, time_limit=base_time)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            try:
                time_left = deadline - time.time() - 3
                if time_left > 5:
                    opt_pts, opt_energy = _riemannian_lbfgs(init_pts, time_limit=min(time_left, 15.0))
                    if opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 1, 12.0)
        if final_time > 1.0:
            try:
                if final_time > 5:
                    final_pts, final_energy = _riemannian_cg(best_pts, time_limit=min(final_time * 0.5, 8.0))
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
                        final_time -= min(final_time * 0.5, 8.0)
                
                if final_time > 3:
                    final_pts, final_energy = _polish(best_pts, time_limit=min(final_time * 0.6, 6.0))
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
                        final_time -= min(final_time * 0.6, 6.0)
                
                if final_time > 2:
                    final_pts, final_energy = _riemannian_lbfgs(best_pts, time_limit=min(final_time, 5.0))
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
            except Exception:
                pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                mid = (best_pts[i] + best_pts[j]) / 2
                perp = best_pts[i] - best_pts[j]
                perp = perp / (np.linalg.norm(perp) + 1e-15)
                cross = np.cross(best_pts[i], perp)
                cross = cross / (np.linalg.norm(cross) + 1e-15)
                best_pts[i] = _normalize(best_pts[i] + 1e-8 * cross)
                best_pts[j] = _normalize(best_pts[j] - 1e-8 * cross)
    
    return best_pts
