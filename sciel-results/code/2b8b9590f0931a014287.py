"""Rewritten program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) == 3:
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        elif len(verts) == 4:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _spectral_energy_loss(P_flat, L=8):
    """
    Spectral energy minimization: minimize variance of pair correlation
    via spherical harmonics expansion. This promotes uniformity.
    
    P_flat: shape (3N,) raw vectors.
    L: maximum degree of spherical harmonics.
    Returns (loss, grad_flat).
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    
    n = len(Pn)
    
    # Compute all pairwise dot products
    dots = Pn @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # Spherical harmonic expansion: use Legendre polynomials for simplicity
    # For uniform distribution, the sum over pairs of P_l(cos theta) should be small
    # for l >= 1 (l=0 term is fixed)
    
    # Compute Legendre polynomials P_l(x) for x = dots
    # Use numpy's polynomial evaluation
    from numpy.polynomial import legendre as Lg
    
    # Evaluate Legendre polynomials up to degree L
    # x = dots (n x n matrix)
    # We want sum over all pairs of P_l(dots[i,j])
    
    total_energy = 0.0
    grad_dots = np.zeros_like(dots)
    
    for l in range(1, L + 1):
        # Evaluate P_l(x)
        Pl = Lg.legval(dots, Lg.leggauss(l+1, l+1)[1][:l+1])
        # Actually, let's use the explicit Legendre polynomials
        # P_0(x) = 1
        # P_1(x) = x
        # P_2(x) = (3x^2 - 1)/2
        # etc.
        
        if l == 1:
            Pl = dots
            dPl_dx = np.ones_like(dots)
        elif l == 2:
            Pl = (3.0 * dots**2 - 1.0) / 2.0
            dPl_dx = 3.0 * dots
        elif l == 3:
            Pl = (5.0 * dots**3 - 3.0 * dots) / 2.0
            dPl_dx = (15.0 * dots**2 - 3.0) / 2.0
        elif l == 4:
            Pl = (35.0 * dots**4 - 30.0 * dots**2 + 3.0) / 8.0
            dPl_dx = (140.0 * dots**3 - 60.0 * dots) / 8.0
        elif l == 5:
            Pl = (63.0 * dots**5 - 70.0 * dots**3 + 15.0 * dots) / 8.0
            dPl_dx = (315.0 * dots**4 - 210.0 * dots**2 + 15.0) / 8.0
        elif l == 6:
            Pl = (231.0 * dots**6 - 315.0 * dots**4 + 105.0 * dots**2 - 5.0) / 16.0
            dPl_dx = (1386.0 * dots**5 - 1260.0 * dots**3 + 210.0 * dots) / 16.0
        elif l == 7:
            Pl = (429.0 * dots**7 - 693.0 * dots**5 + 315.0 * dots**3 - 35.0 * dots) / 16.0
            dPl_dx = (3003.0 * dots**6 - 3465.0 * dots**4 + 945.0 * dots**2 - 35.0) / 16.0
        elif l == 8:
            Pl = (6435.0 * dots**8 - 12012.0 * dots**6 + 6930.0 * dots**4 - 1260.0 * dots**2 + 35.0) / 128.0
            dPl_dx = (51480.0 * dots**7 - 72072.0 * dots**5 + 27720.0 * dots**3 - 2520.0 * dots) / 128.0
        else:
            continue
        
        # Sum over all pairs (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        Pl_sum = np.sum(Pl[mask])
        total_energy += Pl_sum**2 / (2 * l + 1)
        
        # Gradient with respect to dots
        # d(total_energy)/d(dots[i,j]) = 2 * Pl_sum / (2l+1) * dPl_dx[i,j] for i != j
        grad_masked = 2.0 * Pl_sum / (2 * l + 1) * dPl_dx
        grad_masked = grad_masked * mask
        grad_dots += grad_masked
    
    # Back-propagate to Pn
    # dots[i,j] = Pn[i] . Pn[j]
    # d(dots[i,j])/d(Pn[i]) = Pn[j]
    # d(dots[i,j])/d(Pn[j]) = Pn[i]
    
    grad_Pn = np.zeros_like(Pn)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad_Pn[i] += grad_dots[i, j] * Pn[j]
                grad_Pn[j] += grad_dots[i, j] * Pn[i]
    
    # Project to tangent space
    grad_dot = np.sum(grad_Pn * Pn, axis=1, keepdims=True)
    grad_tan = grad_Pn - grad_dot * Pn
    
    # Back-prop through normalization
    grad_flat = (grad_tan / norms).flatten()
    
    return total_energy, grad_flat

def _combined_spectral_loss(P_flat, samples, lam=6.0, L=6, w_spectral=1.0):
    """
    Combined loss: spectral energy + soft-max covering loss.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    
    # Soft-max covering loss
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    
    cover_loss = np.sum(np.exp(lam * min_dists))
    
    # Gradient of covering loss
    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom
    
    grad_P_cover = np.zeros_like(Pn)
    np.add.at(grad_P_cover, nearest, grad_factor[:, None] * samples)
    
    grad_dot_c = np.sum(grad_P_cover * Pn, axis=1, keepdims=True)
    grad_tan_c = grad_P_cover - grad_dot_c * Pn
    grad_flat_c = (grad_tan_c / norms).flatten()
    
    # Spectral energy loss
    spec_loss, grad_flat_s = _spectral_energy_loss(P_flat, L=L)
    
    total_loss = cover_loss + w_spectral * spec_loss
    total_grad = grad_flat_c + w_spectral * grad_flat_s
    
    return total_loss, total_grad

def _local_search_greedy(P, samples, max_iters=300, deadline=None):
    """
    Greedy local search: repeatedly move the point responsible for the worst
    sample toward that sample with adaptive step size.
    """
    n_samples = len(samples)
    rng = np.random.default_rng(42)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return np.max(min_dists), min_dists, nearest

    current_max, min_dists, nearest = get_max_dist(P)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        # Direction to move
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm < 1e-8:
                continue
            grad_dir = rand / rand_norm
        else:
            grad_dir = grad_dir / grad_norm
        
        # Adaptive step
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        for step in [0.15, 0.10, 0.05, 0.02, 0.01]:
            new_p = p_old + step * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = get_max_dist(P)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            min_dists, nearest = get_max_dist(P)[1:]
        else:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.05 * rand).reshape(1, 3))[0]
            min_dists, nearest = get_max_dist(P)[1:]
    
    return P, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (limit is 150s)
    
    # 1. Initialize with Fibonacci sphere + small noise
    P = _fibonacci_sphere(N, seed_noise=0.02, seed=42)
    
    # 2. Generate samples for optimization
    S_opt = 6000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 3. First optimization: combined spectral + covering loss
    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 50.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")
    
    try:
        res = minimize(
            lambda x: _combined_spectral_loss(x, samples_opt, lam=6.0, L=6, w_spectral=0.5),
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
            callback=time_callback
        )
        if res.x is not None:
            P = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    # 4. Second optimization: more samples, higher lambda, less spectral weight
    if time.time() < deadline - 60.0:
        S_opt2 = 12000
        samples_opt2 = _fibonacci_sphere(S_opt2)
        
        def time_callback2(xk):
            if time.time() > deadline - 35.0:
                raise ValueError("Time limit approaching")
        
        try:
            res2 = minimize(
                lambda x: _combined_spectral_loss(x, samples_opt2, lam=8.0, L=5, w_spectral=0.3),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 40, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback2
            )
            if res2.x is not None:
                P = _normalize(res2.x.reshape(-1, 3))
        except Exception:
            pass
    
    # 5. Greedy local search with fine samples
    if time.time() < deadline - 15.0:
        S_fine = 25000
        samples_fine = _fibonacci_sphere(S_fine)
        P, _ = _local_search_greedy(P, samples_fine, max_iters=150, 
                                     deadline=deadline - 10.0)
    
    # 6. Final spectral refinement
    if time.time() < deadline - 5.0:
        S_final = 10000
        samples_final = _fibonacci_sphere(S_final)
        
        def time_callback3(xk):
            if time.time() > deadline - 2.0:
                raise ValueError("Time limit approaching")
        
        try:
            res3 = minimize(
                lambda x: _combined_spectral_loss(x, samples_final, lam=10.0, L=4, w_spectral=0.2),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback3
            )
            if res3.x is not None:
                P = _normalize(res3.x.reshape(-1, 3))
        except Exception:
            pass
    
    # Final normalization
    P = _normalize(P)
    
    return P

# EVOLVE-BLOCK-END
