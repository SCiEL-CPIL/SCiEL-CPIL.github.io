"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

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

def _retract(x):
    """Retract to the sphere (normalize each point)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-15)

def _tangent_grad(g, x):
    """Project gradient onto tangent space."""
    proj = np.einsum('ij,ij->i', g, x)
    return g - proj[:, np.newaxis] * x

def _lbfgs_sphere(pts, max_iter=500, tol=1e-12, deadline=None, m=20):
    """
    L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    g_t = _tangent_grad(g, x)
    
    # L-BFGS memory
    s_list = []  # steps
    y_list = []  # gradient differences
    
    start_time = time.time()
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Two-loop recursion to compute search direction
        # Flatten for L-BFGS
        g_flat = g_t.reshape(-1)
        
        # Compute direction using two-loop recursion
        r = g_flat.copy()
        alphas = []
        qs = []
        
        for s, y in zip(reversed(s_list), reversed(y_list)):
            s_flat = s.reshape(-1)
            y_flat = y.reshape(-1)
            alpha = np.dot(s_flat, r)
            alphas.append(alpha)
            qs.append(y_flat)
            r = r - alpha * y_flat
        
        # Initial Hessian approximation
        if s_list and y_list:
            s_last = s_list[-1].reshape(-1)
            y_last = y_list[-1].reshape(-1)
            gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-30)
        else:
            gamma = 1.0
        
        q = gamma * r
        
        for s, y, alpha, qs_val in zip(s_list, y_list, reversed(alphas), reversed(qs)):
            s_flat = s.reshape(-1)
            beta = np.dot(y_flat, q)
            q = q + (alpha - beta) * s_flat
        
        # Negate to get descent direction
        p_flat = -q
        p = p_flat.reshape(n, 3)
        
        # Line search with Armijo condition
        c1 = 1e-4
        step_size = 1.0
        x_new = x
        f_new = f
        g_new = g
        g_t_new = g_t
        
        for ls_iter in range(30):
            x_trial = x + step_size * p
            x_trial = _retract(x_trial)
            
            f_trial, g_trial = _energy_and_grad(x_trial)
            g_t_trial = _tangent_grad(g_trial, x_trial)
            
            # Check Armijo condition
            if f_trial <= f + c1 * step_size * np.dot(g_flat, p_flat):
                x_new = x_trial
                f_new = f_trial
                g_new = g_trial
                g_t_new = g_t_trial
                break
            
            step_size *= 0.5
        
        # Update L-BFGS memory
        s = x_new - x
        y = g_t_new - g_t
        sy = np.dot(s.reshape(-1), y.reshape(-1))
        
        if sy > 1e-10:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = x_new
        f = f_new
        g = g_new
        g_t = g_t_new
    
    return x, f

def _polish(pts, max_iter=200, tol=1e-14, deadline=None):
    """
    Final high-precision polish using very tight tolerances.
    """
    x = pts.copy()
    f, g = _energy_and_grad(x)
    g_t = _tangent_grad(g, x)
    
    start_time = time.time()
    lr = 1e-6
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Small step gradient descent for polish
        step_size = lr / (np.linalg.norm(g_t) + 1e-15)
        x_new = x - step_size * g_t
        x_new = _retract(x_new)
        
        f_new, g_new = _energy_and_grad(x_new)
        
        if f_new < f:
            x = x_new
            f = f_new
            g = g_new
            g_t = _tangent_grad(g, x)
        else:
            lr *= 0.5
            if lr < 1e-15:
                break
    
    return x, f

def _deduplicate(pts, min_dist=1e-9):
    """Check for duplicate points and separate them if needed."""
    n = pts.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                # Separate points
                mid = (pts[i] + pts[j]) / 2
                diff = pts[i] - pts[j]
                if np.linalg.norm(diff) < 1e-15:
                    diff = np.array([1.0, 0, 0])
                diff = diff / np.linalg.norm(diff)
                pts[i] = _retract(pts[i] + 1e-8 * diff)
                pts[j] = _retract(pts[j] - 1e-8 * diff)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill
    
    best_pts = None
    best_energy = np.inf
    
    # Generate multiple diverse initial configurations
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. Random points with different seeds
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    np.random.seed(777)
    for trial in range(2):
        noise = np.random.randn(N, 3) * 0.1
        pts_pert = pts_fib + noise
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        inits.append(pts_pert)
    
    # Optimize each initialization, keeping the best result
    n_inits = len(inits)
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time: give more time to earlier inits, less to later ones
        remaining = deadline - time.time()
        inits_left = n_inits - idx
        time_per_init = remaining / inits_left * 0.9  # 10% safety margin
        
        init_deadline = time.time() + time_per_init
        
        pts_opt, energy = _lbfgs_sphere(init_pts, max_iter=500, tol=1e-12, deadline=init_deadline)
        
        # Ensure points are on the sphere and deduplicate
        pts_opt = _retract(pts_opt)
        pts_opt = _deduplicate(pts_opt)
        
        # Recompute energy after deduplication
        energy, _ = _energy_and_grad(pts_opt)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    # Use the best result found
    if best_pts is None:
        best_pts = pts_fib
    
    # Final polish on the best result
    if time.time() < deadline - 2.0:
        best_pts, best_energy = _polish(best_pts, max_iter=200, tol=1e-14, deadline=deadline - 1.0)
        best_pts = _retract(best_pts)
        best_pts = _deduplicate(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
