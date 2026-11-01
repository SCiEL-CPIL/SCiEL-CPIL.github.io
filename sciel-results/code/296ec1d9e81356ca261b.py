"""Optimized logarithmic energy of 150 points on the unit sphere.

Strategy:
- Vectorized energy/gradient with tangent-space projection.
- Riemannian L-BFGS (via exponential map) as the primary optimizer.
- Multiple diverse initializations (Fibonacci, Halton, random, farthest-point).
- Adaptive time allocation and final high-precision polish.
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
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient: grad_i = -sum_{j!=i} (1/|x_i - x_j|^2) * (x_i - x_j)
    #             = -sum_{j!=i} inv_d2[i,j] * (x_i - x_j)
    #             = -pts[i] * sum_j inv_d2[i,j] + sum_j inv_d2[i,j] * pts[j]
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Energy (upper triangle)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diff, axis=2) if False else np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Exponential map on unit sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _fibonacci_sphere(n):
    """Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _halton_sphere(n):
    """Halton sequence points on unit sphere."""
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
        u1 = halton(i, 2)
        u2 = halton(i, 3)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    pts = np.array(pts)
    return _normalize(pts)


def _farthest_point_init(n, seed=0):
    """Farthest-point sampling initialization."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    # Start with a random point
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, n):
        # Find farthest point from current set
        dists = np.min([np.sum((pts[:i] - p) ** 2, axis=1) for p in [rng.standard_normal(3)]], axis=0)
        # Actually compute properly
        # Generate many candidates and pick the one farthest from all existing points
        candidates = rng.standard_normal((500, 3))
        candidates = _normalize(candidates)
        min_dists = np.min([np.sum((candidates - pts[j]) ** 2, axis=1) for j in range(i)], axis=0)
        best_idx = np.argmax(min_dists)
        pts[i] = candidates[best_idx]
    
    return pts


def _riemannian_lbfgs(pts, time_limit=30.0):
    """Riemannian L-BFGS optimization using exponential map."""
    start = time.time()
    
    # Convert to flat vector
    x0 = pts.flatten()
    
    # L-BFGS history
    m = 10  # number of correction pairs to store
    S = []  # step vectors
    Y = []  # gradient difference vectors
    
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initial direction = negative gradient
    d = -grad.copy()
    
    for it in range(2000):
        if time.time() - start > time_limit - 0.5:
            break
        
        # L-BFGS two-loop recursion to compute new direction
        # r = grad
        r = grad.copy()
        alphas = []
        rhos = []
        
        # Backward loop
        for k in range(len(S) - 1, -1, -1):
            s_k = S[k].flatten()
            y_k = Y[k].flatten()
            ys = np.sum(y_k * s_k)
            if ys > 1e-12:
                rho_k = 1.0 / ys
            else:
                rho_k = 0.0
            alpha_k = rho_k * np.sum(s_k * r)
            alphas.append(alpha_k)
            r = r - alpha_k * y_k
            rhos.append(rho_k)
        
        alphas.reverse()
        rhos.reverse()
        
        # Initial Hessian approximation
        if len(S) > 0:
            s_last = S[-1].flatten()
            y_last = Y[-1].flatten()
            ys = np.sum(y_last * s_last)
            if ys > 1e-12:
                gamma = np.sum(s_last * y_last) / ys
            else:
                gamma = 1.0
        else:
            gamma = 1.0
        
        q = gamma * r
        
        # Forward loop
        for k in range(len(S)):
            s_k = S[k].flatten()
            y_k = Y[k].flatten()
            beta_k = rhos[k] * np.sum(y_k * q)
            q = q + (alphas[k] - beta_k) * s_k
        
        d = -q.reshape(N, 3)
        
        # Project to tangent space
        dot_prod = np.sum(d * current, axis=1)
        d -= dot_prod[:, np.newaxis] * current
        
        # Line search (backtracking)
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
            
            # Armijo condition
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            # Try very small step
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
        sy = np.sum(s_vec * y_vec)
        if sy > 1e-12:
            S.append(s_vec)
            Y.append(y_vec)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Convergence check
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
    
    return current, energy


def _riemannian_cg(pts, time_limit=20.0):
    """Riemannian conjugate gradient with Armijo line search."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(500):
        if time.time() - start > time_limit - 0.3:
            break
        
        # Ensure d is in tangent space
        dot_prod = np.sum(d * current, axis=1)
        d -= dot_prod[:, np.newaxis] * current
        
        # Line search
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad.copy()
        best_alpha = 0.0
        
        for ls in range(40):
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
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Polak-Ribiere update
        beta_num = np.sum(grad * (grad - prev_grad))
        beta_den = np.sum(prev_grad * prev_grad)
        if beta_den > 1e-30:
            beta = max(0.0, beta_num / beta_den)
        else:
            beta = 0.0
        if beta > 1.0:
            beta = 0.0
        
        d = -grad + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _polish(pts, time_limit=5.0):
    """High-precision gradient descent with adaptive step sizes."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        # Step in negative gradient direction
        d = -grad
        
        # Adaptive line search: start with a reasonable step
        alpha = 0.1
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
            if new_energy <= energy + 1e-6 * alpha * np.sum(grad * d):
                break
            
            alpha *= 0.5
        
        if best_alpha == 0.0:
            # Try even smaller steps
            for alpha in [0.01, 0.001, 0.0001]:
                new_pts = _exponential_map(current, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < best_energy - 1e-14:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha
                    break
            if best_alpha == 0.0:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate diverse initializations (fewer but higher quality)
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N).copy())
    
    # 2. Halton sequence
    initializations.append(_halton_sphere(N).copy())
    
    # 3. Two random initializations
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        initializations.append(_normalize(pts_rand))
    
    # 4. One farthest-point sampling
    initializations.append(_farthest_point_init(N, seed=0))
    
    # 5. Perturbed Fibonacci
    rng_local = np.random.default_rng(456 + 13)
    pts_pert = _fibonacci_sphere(N) + rng_local.standard_normal((N, 3)) * 0.02
    initializations.append(_normalize(pts_pert))
    
    # 6. Different Fibonacci offset
    offset = 0.5
    i = np.arange(N) + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(_normalize(pts))
    
    best_pts = initializations[0].copy()
    best_energy = _log_energy_fast(best_pts)
    
    # Optimize each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        time_left = deadline - time.time() - 5
        if time_left < 12:
            break
        
        # Allocate time: give each init a fair share
        remaining = len(initializations) - idx
        base_time = min(time_left, max(15.0, 100.0 / remaining))
        
        try:
            # Phase 1: Riemannian L-BFGS (main workhorse)
            t1 = min(base_time * 0.55, 18.0)
            opt_pts, opt_energy = _riemannian_lbfgs(init_pts, time_limit=t1)
            
            # Phase 2: Riemannian CG polish (with fixed tangent projection)
            t2 = min(base_time * 0.3, 10.0)
            if time.time() < deadline - 10:
                opt_pts, opt_energy = _riemannian_cg(opt_pts, time_limit=t2)
            
            # Phase 3: Final high-precision polish
            if time.time() < deadline - 8:
                opt_pts, opt_energy = _polish(opt_pts, time_limit=4.0)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            # Fallback to L-BFGS-B
            try:
                x0 = init_pts.flatten()
                
                def fun_and_grad(x):
                    p = x.reshape(N, 3)
                    p = _normalize(p)
                    e, g = _energy_and_grad(p)
                    return e, g.flatten()
                
                res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                               options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
                x_opt = res.x.reshape(N, 3)
                x_opt = _normalize(x_opt)
                e_opt = _log_energy_fast(x_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = x_opt.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 3:
        final_time = min(deadline - time.time() - 2, 12.0)
        if final_time > 1.0:
            try:
                # Riemannian CG for final convergence
                if final_time > 5:
                    t_cg = min(final_time * 0.5, 6.0)
                    final_pts, final_energy = _riemannian_cg(best_pts, time_limit=t_cg)
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_pts = final_pts
                        final_time -= t_cg
                
                # High-precision polish
                if final_time > 2:
                    t_polish = min(final_time * 0.8, 6.0)
                    final_pts, final_energy = _polish(best_pts, time_limit=t_polish)
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
