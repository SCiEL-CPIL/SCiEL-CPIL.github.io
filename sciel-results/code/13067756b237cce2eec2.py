import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq = np.maximum(dist_sq, 1e-30)
    iu = np.triu_indices(N, k=1)
    E = -0.5 * np.sum(np.log(dist_sq[iu]))
    inv_ds = 1.0 / dist_sq
    np.fill_diagonal(inv_ds, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_ds, diff)
    return E, grad

def _project(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(n, 1e-15)

def _angles_to_pts(angles):
    theta = angles[:, 0]
    phi = angles[:, 1]
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _pts_to_angles(pts):
    z = pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return np.column_stack([theta, phi])

def _energy_angles(angles):
    pts = _angles_to_pts(angles)
    E, _ = _energy_and_grad(pts)
    return E

def _grad_angles(angles):
    pts = _angles_to_pts(angles)
    E, g = _energy_and_grad(pts)
    theta = angles[:, 0]
    phi = angles[:, 1]
    st = np.sin(theta)
    ct = np.cos(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    dtheta = g[:, 0]*ct*cp + g[:, 1]*ct*sp - g[:, 2]*st
    dphi = -g[:, 0]*st*sp + g[:, 1]*st*cp
    return np.column_stack([dtheta, dphi])

def _riemannian_grad(pts, g):
    return g - np.sum(g * pts, axis=1, keepdims=True) * pts

def _gd_polish(pts, t0, TL, max_iters=500):
    E, g = _energy_and_grad(pts)
    best_E, best_pts = E, pts.copy()
    for it in range(max_iters):
        if time.time() - t0 > TL:
            break
        gr = _riemannian_grad(pts, g)
        gn = np.linalg.norm(gr)
        if gn < 1e-14:
            break
        alpha = 1.0 / (gn * gn)
        c = 1e-4
        for ls in range(50):
            new_pts = _project(pts - alpha * gr)
            new_E, _ = _energy_and_grad(new_pts)
            if new_E <= E - c * alpha * gn * gn:
                break
            alpha *= 0.5
        if new_E < E:
            pts, E = new_pts, new_E
            if E < best_E:
                best_E, best_pts = E, pts.copy()
        else:
            break
        _, g = _energy_and_grad(pts)
    return best_pts, best_E

def _fib():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z*z, 0.0))
    return np.column_stack([r*np.cos(phi), r*np.sin(phi), z])

def construct():
    t0 = time.time()
    TL = 140.0
    best_E = np.inf
    best_pts = None
    best_angles = None
    
    theta_b = [(1e-6, np.pi - 1e-6)] * N
    phi_b = [(-np.pi, np.pi)] * N
    bounds = theta_b + phi_b
    
    pts = _fib()
    angles = _pts_to_angles(pts)
    
    try:
        res = minimize(_energy_angles, angles, jac=_grad_angles, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 5000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_E:
            best_E = res.fun
            best_angles = res.x
            best_pts = _angles_to_pts(res.x)
    except Exception:
        pass
    
    if best_pts is None:
        best_pts = pts
        best_E = _energy_angles(angles)
        best_angles = angles
    
    # GD polish
    if time.time() - t0 < TL - 5:
        best_pts, best_E = _gd_polish(best_pts, max_iters=500, t0=t0, TL=TL)
        best_angles = _pts_to_angles(best_pts)
    
    # Restarts
    rng = np.random.RandomState(42)
    if time.time() - t0 < TL - 10:
        for r in range(15):
            if time.time() - t0 > TL - 5:
                break
            for scale in [0.1, 0.05, 0.01]:
                if time.time() - t0 > TL - 5:
                    break
                ap = best_angles + rng.randn(N, 2) * scale
                ap[:, 0] = np.clip(ap[:, 0], 1e-5, np.pi - 1e-5)
                ap[:, 1] = np.clip(ap[:, 1], -np.pi + 1e-5, np.pi - 1e-5)
                try:
                    res = minimize(_energy_angles, ap, jac=_grad_angles, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': 3000, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-12})
                    if res.fun < best_E:
                        best_E = res.fun
                        best_angles = res.x
                        best_pts = _angles_to_pts(res.x)
                except Exception:
                    pass
    
    # Final polish
    if time.time() - t0 < TL - 2:
        best_pts, best_E = _gd_polish(best_pts, max_iters=200, t0=t0, TL=TL)
    
    return _project(best_pts)
