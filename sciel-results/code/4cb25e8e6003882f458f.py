"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, direction, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard and tight tolerances.
    """
    best_x = x0.copy()
    best_f, _ = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxfun': 15000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=500, lr_init=0.05, lr_max=0.15):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space. Uses adaptive learning rate.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, -g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.15, lr_max)
        else:
            lr *= 0.5
            if lr < 1e-14:
                break
    
    return pts, f

def _ultra_polish(pts, deadline, max_iter=2000):
    """
    Ultra-precise Newton-style polish with very small steps and tight Armijo.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_t = g - proj[:, np.newaxis] * pts
    
    # Estimate local curvature scale from gradient magnitude
    g_norm = np.linalg.norm(g_t)
    if g_norm < 1e-15:
        return pts, f
    
    # Start with a small step relative to gradient scale
    alpha = 0.01
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        # Armijo backtracking
        sufficient = 1e-18 * alpha * np.sum(g_t * (-g_t))
        cand = _retract(pts, -g_t, alpha)
        cand_f, cand_g = _energy_and_grad(cand)
        cand_proj = np.einsum('ij,ij->i', cand_g, cand)
        cand_g_t = cand_g - cand_proj[:, np.newaxis] * cand
        
        if cand_f <= f - sufficient:
            pts = cand
            f = cand_f
            g = cand_g
            g_t = cand_g_t
            # Slightly increase step if we made good progress
            if it % 10 == 0:
                alpha = min(alpha * 1.05, 0.05)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
        
        # Check convergence
        if np.linalg.norm(g_t) < 1e-15:
            break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=None, m=25):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        if len(s_list) > 0:
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if abs(sy) > 1e-15 else 0.0)
            
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        # Line search with backtracking
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(50):
            candidate = _retract(x, direction, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-5 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-12:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def _deduplicate(pts_list, threshold=0.3):
    """Remove initializations that are too similar to already-accepted ones."""
    accepted = []
    for pts in pts_list:
        is_dup = False
        for acc in accepted:
            # Compute mean pairwise distance similarity
            diffs = pts[:, np.newaxis, :] - acc[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=2))
            # If the mean distance between corresponding points is small, they're similar
            mean_dist = np.mean(dists)
            if mean_dist < threshold:
                is_dup = True
                break
        if not is_dup:
            accepted.append(pts)
    return accepted

def _simulated_annealing(pts, deadline, max_iter=50000, T_init=0.05, T_min=1e-6, cooling=0.999):
    """
    Simulated annealing on the sphere to escape local minima.
    Uses small rotational perturbations.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    T = T_init
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a perturbation: rotate a random subset of points slightly
        n_pts = np.random.choice([1, 1, 2, 2, 3])
        indices = np.random.choice(N, size=n_pts, replace=False)
        
        new_pts = best_pts.copy()
        for idx in indices:
            # Small random rotation
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.randn() * T * 0.1
            c = np.cos(angle)
            s = np.sin(angle)
            k = axis
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)
            new_pts[idx] = R @ new_pts[idx]
        
        # Project to sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        
        new_energy = _energy_and_grad(new_pts)[0]
        delta = new_energy - best_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            best_pts = new_pts
            best_energy = new_energy
        
        T = max(T * cooling, T_min)
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. Multiple random starts
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci with various scales
    for seed in [11, 22, 33, 44, 55]:
        np.random.seed(seed)
        noise_scale = 0.02 + 0.03 * (seed // 11)
        noise = np.random.randn(N, 3) * noise_scale
        pts_pert = pts_fib + noise
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        inits.append(pts_pert)
    
    # 4. Icosahedral base
    def icosahedron():
        t = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        return verts / norms
    
    ico = icosahedron()
    
    # 5. Band-based structured starts
    np.random.seed(777)
    for n_bands in [8, 10, 12, 14, 16]:
        pts_bands = []
        for band in range(n_bands):
            z_b = 1.0 - 2.0 * (band + 0.5) / n_bands
            r_b = np.sqrt(max(1.0 - z_b*z_b, 0.0))
            n_pts_in_band = N // n_bands
            if band < N % n_bands:
                n_pts_in_band += 1
            phis = 2.0 * np.pi * np.arange(n_pts_in_band) / n_pts_in_band + band * 0.3
            for ph in phis:
                pts_bands.append([r_b * np.cos(ph), r_b * np.sin(ph), z_b])
        pts_bands = np.array(pts_bands[:N])
        if len(pts_bands) == N:
            inits.append(pts_bands)
    
    # 6. Concentric rings with random offsets
    for seed in [100, 200, 300]:
        np.random.seed(seed)
        n_rings = 8
        pts_rings = []
        for ring in range(n_rings):
            z_r = 1.0 - 2.0 * (ring + 0.5) / n_rings
            r_r = np.sqrt(max(1.0 - z_r*z_r, 0.0))
            n_in_ring = N // n_rings
            if ring < N % n_rings:
                n_in_ring += 1
            offset = np.random.uniform(0, 2*np.pi)
            phis = 2.0 * np.pi * np.arange(n_in_ring) / n_in_ring + offset
            for ph in phis:
                pts_rings.append([r_r * np.cos(ph), r_r * np.sin(ph), z_r])
        pts_rings = np.array(pts_rings[:N])
        if len(pts_rings) == N:
            inits.append(pts_rings)
    
    # Deduplicate similar initializations
    inits = _deduplicate(inits, threshold=0.3)
    
    # Optimize each initialization
    n_inits = len(inits)
    # Reserve time for final polish stages and SA
    polish_time = 30.0
    sa_time = 20.0
    time_per_init = (145.0 - polish_time - sa_time) / max(n_inits, 1)
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - polish_time - sa_time:
            break
        
        init_deadline = min(deadline - polish_time - sa_time, time.time() + time_per_init)
        
        # Alternate between optimizers for diversity
        if idx % 3 == 0:
            pts_opt, energy = _lbfgs_sphere_custom(init_pts, max_iter=500, tol=1e-12, deadline=init_deadline, m=25)
        elif idx % 3 == 1:
            x0 = init_pts.flatten()
            x_opt, energy = _lbfgs_sphere(x0, init_deadline, maxiter=3000)
            pts_opt = x_opt.reshape(N, 3)
        else:
            pts_opt, energy = _polish(init_pts, init_deadline, max_iter=200)
        
        # Ensure points are on the sphere
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / np.maximum(norms, 1e-15)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    if best_pts is None:
        best_pts = pts_fib
        best_energy, _ = _energy_and_grad(best_pts)
    
    # Stage 1: Simulated annealing to escape local minima
    if time.time() < deadline - polish_time:
        sa_deadline = min(deadline - polish_time, time.time() + sa_time)
        best_pts, best_energy = _simulated_annealing(best_pts, sa_deadline, max_iter=50000, T_init=0.05, T_min=1e-6, cooling=0.999)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / np.maximum(norms, 1e-15)
    
    # Stage 2: Standard polish
    if time.time() < deadline - 10.0:
        best_pts, best_energy = _polish(best_pts, deadline, max_iter=500)
    
    # Stage 3: Custom L-BFGS from the polished result
    if time.time() < deadline - 8.0:
        stage3_deadline = min(deadline - 8.0, time.time() + 10.0)
        best_pts, best_energy = _lbfgs_sphere_custom(best_pts, max_iter=300, tol=1e-13, deadline=stage3_deadline, m=25)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / np.maximum(norms, 1e-15)
    
    # Stage 4: Ultra-precise polish
    if time.time() < deadline - 3.0:
        best_pts, best_energy = _ultra_polish(best_pts, deadline, max_iter=2000)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)
    
    return best_pts
# EVOLVE-BLOCK-END
