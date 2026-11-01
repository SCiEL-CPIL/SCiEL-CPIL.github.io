import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    def smooth_objective(x):
        pts = x.reshape(-1, D)
        # Calculate pairwise squared distances
        # (N, N, D) -> (N, N)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d2 = d2[mask]
        
        beta = 50.0
        # Soft max
        # Numerical stability: shift by max
        m = np.max(d2)
        exp_pos = np.exp(beta * (d2 - m))
        S_max = (m + np.log(np.sum(exp_pos)) / beta)
        
        # Soft min
        m = np.min(d2)
        exp_neg = np.exp(-beta * (d2 - m))
        S_min = (m - np.log(np.sum(exp_neg)) / beta)
        
        if S_min < 1e-10:
            return 1e10
        return (S_max / S_min) ** 2

    def smooth_gradient(x):
        pts = x.reshape(-1, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        
        mask = ~np.eye(N, dtype=bool)
        d2_flat = d2[mask]
        
        beta = 50.0
        
        # Weights for soft max
        m_max = np.max(d2_flat)
        exp_pos = np.exp(beta * (d2_flat - m_max))
        w_max = exp_pos / np.sum(exp_pos)
        
        # Weights for soft min
        m_min = np.min(d2_flat)
        exp_neg = np.exp(-beta * (d2_flat - m_min))
        w_min = exp_neg / np.sum(exp_neg)
        
        S_max = (m_max + np.log(np.sum(exp_pos)) / beta)
        S_min = (m_min - np.log(np.sum(exp_neg)) / beta)
        
        if S_min < 1e-10:
            return np.zeros_like(x)
        
        # dJ/dS_max, dJ/dS_min
        dJ_dSmax = 2.0 * S_max / (S_min ** 2)
        dJ_dSmin = -2.0 * (S_max ** 2) / (S_min ** 3)
        
        # dJ/d(d2_k)
        # dS_max/d(d2_k) = w_max_k
        # dS_min/d(d2_k) = w_min_k
        dJ_dd2 = dJ_dSmax * w_max + dJ_dSmin * w_min
        
        # Reconstruct d2 gradient
        # d2 is (N, N). We need gradient of J w.r.t x.
        # dJ/dx_i = sum_{j} dJ/dd2_{ij} * dd2_{ij}/dx_i
        # dd2_{ij}/dx_i = 2 * (x_i - x_j)
        
        # dJ_dd2_full is (N, N)
        dJ_dd2_full = np.zeros((N, N))
        dJ_dd2_full[mask] = dJ_dd2
        
        # Gradient w.r.t x
        # grad_x = 2 * sum_j (dJ_dd2_full[i,j] * (x_i - x_j))
        # This can be computed as:
        # grad_x[i] = 2 * sum_j dJ_dd2_full[i,j] x_i - 2 * sum_j dJ_dd2_full[i,j] x_j
        # Let A = dJ_dd2_full.
        # sum_j A[i,j] x_i = x_i * sum_j A[i,j]
        # sum_j A[i,j] x_j = (A @ x)[i]
        
        sum_A = np.sum(dJ_dd2_full, axis=1)
        Ax = dJ_dd2_full @ pts
        
        grad = 2.0 * (pts * sum_A[:, np.newaxis] - Ax)
        
        return grad.flatten()

    def exact_objective(x):
        pts = x.reshape(-1, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        if np.min(dists) < 1e-10:
            return 1e10
        return (np.max(dists) / np.min(dists)) ** 2

    # Initializers
    inits = []
    
    # 1. Fibonacci Sphere
    def fib_sphere(n):
        pts = []
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            pts.append([x, y, z])
        return np.array(pts)
    
    inits.append(fib_sphere(N).flatten())
    
    # 2. Random Spheres
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random points on sphere
        v = rng.normal(size=(N, D))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v.flatten())
        
    # 3. Icosahedron + 2
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 random points
    extra = rng.normal(size=(2, D))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    
    pts_ico = np.vstack([ico, extra])
    inits.append(pts_ico.flatten())
    
    best_points = None
    best_score = float('inf')
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        # Optimize with L-BFGS-B
        try:
            res = minimize(smooth_objective, x0, jac=smooth_gradient, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            score = exact_objective(res.x)
            if score < best_score:
                best_score = score
                best_points = res.x
        except Exception:
            pass
    
    # Polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(exact_objective, best_points, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            score = exact_objective(res.x)
            if score < best_score:
                best_score = score
                best_points = res.x
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        best_points = fib_sphere(N).flatten()
        
    return best_points.reshape(N, D)
