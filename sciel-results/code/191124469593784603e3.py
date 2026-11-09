import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient with respect to the 3D coordinates.
    The points are normalized internally to the unit sphere.
    """
    x = x.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    u = x / norms
    
    # Compute pairwise squared distances via dot product: |u_i-u_j|^2 = 2 - 2*u_i·u_j
    dot = u @ u.T
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 0.0, out=dist_sq)
    
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-12
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    iu = np.triu_indices(N, 1)
    d_sq = dist_sq[iu]
    E = np.sum(np.log(1.0 / np.sqrt(d_sq)))
    
    S1 = np.sum(inv_dist_sq, axis=1)
    S2 = inv_dist_sq @ u
    grad_u = u * S1[:, np.newaxis] - S2
    grad_x = grad_u / norms
    
    return E, grad_x.flatten()

def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _optimize_lbfgs(x0, deadline, ftol=1e-14, gtol=1e-11):
    best_E, _ = _energy_and_grad(x0)
    best_x = x0.copy()
    
    def callback(xk):
        nonlocal best_x, best_E
        E_k, _ = _energy_and_grad(xk)
        if E_k < best_E:
            best_E = E_k
            best_x = xk.copy()
        return time.time() > deadline
    
    try:
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', jac=True,
                       callback=callback, options={'maxiter': 500, 'ftol': ftol, 'gtol': gtol})
        if res.fun < best_E:
            best_E = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_E

def _conjugate_gradient(x0, max_iter=200, deadline=None):
    x = x0.copy()
    E, _ = _energy_and_grad(x)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        _, grad = _energy_and_grad(x)
        grad = grad.reshape(N, 3)
        u = x / np.linalg.norm(x, axis=1, keepdims=True)
        g_dot_u = np.sum(grad * u, axis=1, keepdims=True)
        g_t = grad - g_dot_u * u
        g_norm = np.linalg.norm(g_t)
        if g_norm < 1e-10:
            break
        
        d = -g_t / (g_norm + 1e-15)
        alpha = 1.0
        for _ in range(30):
            x_new = x + alpha * d
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            E_new, _ = _energy_and_grad(x_new)
            if E_new < E:
                x = x_new
                E = E_new
                break
            alpha *= 0.5
        
        if alpha < 1e-8:
            break
    
    return x, E

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    inits = []
    inits.append(_fibonacci_sphere(N))
    for off in [0.1, 0.3, 0.7, 0.9]:
        inits.append(_fibonacci_sphere(N, offset=off))
    for seed in [42, 123, 456, 789, 1000, 2000]:
        inits.append(_random_sphere(N, seed=seed))
    
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    rng = np.random.default_rng(77)
    ico_tiled = np.tile(ico, (9, 1))[:N]
    ico_tiled = ico_tiled + rng.standard_normal((N, 3)) * 0.3
    ico_tiled /= np.linalg.norm(ico_tiled, axis=1, keepdims=True)
    inits.append(ico_tiled)
    
    best_x = None
    best_E = np.inf
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        x_opt, E_opt = _optimize_lbfgs(x0, deadline)
        if E_opt < best_E:
            best_E = E_opt
            best_x = x_opt.copy()
        
        if time.time() < deadline - 5.0:
            x_cg, E_cg = _conjugate_gradient(x_opt, max_iter=100, deadline=deadline)
            if E_cg < best_E:
                best_E = E_cg
                best_x = x_cg.copy()
    
    if best_x is None:
        best_x = inits[0]
    
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    best_x = best_x / norms
    
    return best_x
