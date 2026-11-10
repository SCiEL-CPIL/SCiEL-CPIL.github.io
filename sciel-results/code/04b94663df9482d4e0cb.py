"""Rewritten program: minimize logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_FRACTION = 0.85  # safety margin


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the sphere."""
    pts = np.asarray(pts)
    # pairwise differences
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # avoid self terms (diagonal)
    np.fill_diagonal(dist, 1.0)
    # energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # use upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    # gradient: dE/dp_i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^2
    # Because E = -sum_{i<j} log(d_ij), dE/dp_i = sum_{j!=i} (p_i - p_j)/d_ij^2
    # More carefully: d/dp_i [-log(|p_i - p_j|)] = -(p_i - p_j)/|p_i - p_j|^2
    # Wait: d/dx log|x| = x/|x|^2, so d/dx [-log|x|] = -x/|x|^2
    # So dE/dp_i = sum_{j != i} -(p_i - p_j)/|p_i - p_j|^2 = sum_{j != i} (p_j - p_i)/|p_i - p_j|^2
    # Let's compute: for each pair (i,j), contribution to grad[i] is (p_j - p_i)/d^2, to grad[j] is (p_i - p_j)/d^2
    grad = np.zeros_like(pts)
    # Vectorized: for each i, grad[i] = sum_j (p_j - p_i)/d_ij^2
    # diff[i,j] = p_i - p_j, so (p_j - p_i) = -diff[i,j]
    # contribution to grad[i] from j: -diff[i,j] / d_ij^2
    # We can compute: grad[i] = -sum_j diff[i,j] / d_ij^2  (excluding j=i)
    inv_d2 = np.zeros_like(dist)
    inv_d2[iu] = 1.0 / (d ** 2)
    inv_d2 = (inv_d2 + inv_d2.T) / 2.0  # symmetrize, diagonal stays 0
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    # diff shape (N,N,3), inv_d2 shape (N,N)
    grad = -np.sum(diff * inv_d2[:, :, None], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n, seed=0):
    """Generate n points on sphere using Fibonacci spiral with optional rotation."""
    rng = np.random.default_rng(seed)
    # Random rotation for diversity
    u = rng.random(3)
    u /= np.linalg.norm(u)
    # Build rotation matrix
    v = np.array([0.0, 0.0, 1.0])
    w = np.cross(u, v)
    s = np.linalg.norm(w)
    c = np.dot(u, v)
    if s < 1e-10:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        K = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]])
        R = np.eye(3) + np.sin(np.arccos(np.clip(c, -1, 1))) / s * K + (1 - c) / (s ** 2) * (K @ K)
    
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return R @ pts.T


def construct():
    start_time = time.time()
    deadline = start_time + 120.0 * DEADLINE_FRACTION
    
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 8
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _fibonacci_spiral(N)
        elif restart == 1:
            pts = _fibonacci_spiral(N, seed=123)
        elif restart == 2:
            pts = _fibonacci_spiral(N, seed=456)
        else:
            # Random initial points with some structure
            pts = rng.random((N, 3))
            pts -= 0.5
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Optimize using L-BFGS-B
        def obj_and_grad(x):
            pts = x.reshape(N, 3)
            pts = _project_to_sphere(pts)
            e, g = _energy_and_grad(pts)
            return e, g
        
        # Initial point on sphere
        x0 = pts.flatten()
        
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
        except Exception:
            continue
        
        # Project result to sphere
        res_pts = _project_to_sphere(res.x.reshape(N, 3))
        res_energy, _ = _energy_and_grad(res_pts)
        
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = res_pts
        
        # Polish with gradient descent
        if time.time() > deadline:
            break
        
        pts = best_pts.copy()
        lr = 0.01
        for step in range(500):
            if time.time() > deadline:
                break
            e, g = _energy_and_grad(pts)
            # Project gradient to tangent space
            pts_norm = np.linalg.norm(pts, axis=1, keepdims=True)
            pts_unit = pts / pts_norm
            g_tangent = g - (g * pts_unit).sum(axis=1, keepdims=True) * pts_unit
            pts = pts - lr * g_tangent
            pts = _project_to_sphere(pts)
            # Reduce learning rate
            lr *= 0.995
        
        pts = _project_to_sphere(pts)
        e_final, _ = _energy_and_grad(pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts
    
    # Final polish
    if time.time() < deadline:
        pts = best_pts.copy()
        lr = 0.001
        for step in range(2000):
            if time.time() > deadline:
                break
            e, g = _energy_and_grad(pts)
            pts_norm = np.linalg.norm(pts, axis=1, keepdims=True)
            pts_unit = pts / pts_norm
            g_tangent = g - (g * pts_unit).sum(axis=1, keepdims=True) * pts_unit
            pts = pts - lr * g_tangent
            pts = _project_to_sphere(pts)
            lr *= 0.999
        
        pts = _project_to_sphere(pts)
        e_final, _ = _energy_and_grad(pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts
    
    return _project_to_sphere(best_pts)
