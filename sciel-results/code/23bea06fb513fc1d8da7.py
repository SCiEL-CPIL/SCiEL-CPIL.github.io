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
    We normalize each point to the sphere inside this function.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient onto tangent space for the effective gradient
    # For f(x) = E(x/||x||), the gradient w.r.t x is (I - x x^T/||x||^2) * grad_E
    # Since pts is already normalized (||pts||=1), the projection is:
    # grad_T = grad - (grad . pts) * pts
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, grad_tangent, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts - step_size * grad_tangent
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard.
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
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 10000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=200):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = 0.05
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=200, tol=1e-10, deadline=None, m=20):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    # Project gradient to tangent space
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    # L-BFGS memory
    s_list = []  # steps
    y_list = []  # gradient differences
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Two-loop recursion to compute direction
        if len(s_list) > 0:
            # Compute rho
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if sy != 0 else 0.0)
            
            # Two-loop recursion
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0  # initial step length
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
        for _ in range(30):
            candidate = _retract(x, direction, -step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            # Armijo condition
            if new_f <= f + 1e-4 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
            new_g_t = new_g - new_proj[:, np.newaxis] * new_x
        
        # Update L-BFGS memory
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-10:
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice) - good global structure
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. Multiple random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for seed in [11, 22, 33]:
        np.random.seed(seed)
        noise = np.random.randn(N, 3) * 0.05
        pts_pert = pts_fib + noise
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        inits.append(pts_pert)
    
    # 4. Structured starts: icosahedral-like with subdivisions
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
    
    # 5. Use a high-symmetry starting point
    # Create points in bands with icosahedral-like symmetry
    np.random.seed(777)
    for n_bands in [10, 12, 14]:
        pts_bands = []
        for band in range(n_bands):
            z = 1.0 - 2.0 * (band + 0.5) / n_bands
            r = np.sqrt(max(1.0 - z*z, 0.0))
            n_pts_in_band = N // n_bands
            if band < N % n_bands:
                n_pts_in_band += 1
            phis = 2.0 * np.pi * np.arange(n_pts_in_band) / n_pts_in_band + band * 0.3
            for ph in phis:
                pts_bands.append([r * np.cos(ph), r * np.sin(ph), z])
        pts_bands = np.array(pts_bands[:N])
        if len(pts_bands) == N:
            inits.append(pts_bands)
    
    # Optimize each initialization, keeping the best result
    n_inits = len(inits)
    time_per_init = 145.0 / n_inits
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        init_deadline = min(deadline - 5.0, time.time() + time_per_init)
        
        # Alternate between custom L-BFGS and scipy L-BFGS for diversity
        if idx % 2 == 0:
            pts_opt, energy = _lbfgs_sphere_custom(init_pts, max_iter=300, tol=1e-10, deadline=init_deadline, m=20)
        else:
            x0 = init_pts.flatten()
            x_opt, energy = _lbfgs_sphere(x0, init_deadline, maxiter=2000)
            pts_opt = x_opt.reshape(N, 3)
        
        # Ensure points are on the sphere
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / np.maximum(norms, 1e-15)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    # Use the best result found
    if best_pts is None:
        best_pts = pts_fib
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _polish(best_pts, deadline, max_iter=300)
    
    return best_pts
# EVOLVE-BLOCK-END
