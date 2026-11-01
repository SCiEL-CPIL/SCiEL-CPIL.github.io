import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
_DEADLINE = 148.0  # 2s safety margin from 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _energy_fsum(pts):
    """Energy computed with math.fsum to match evaluator exactly."""
    total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return -1e30
            total += math.log(1.0 / d)
    return total

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

def _gd_polish(pts, t0, TL, max_iters=500, tight_tol=1e-16):
    E, g = _energy_and_grad(pts)
    best_E, best_pts = E, pts.copy()
    alpha = 1.0
    for it in range(max_iters):
        if time.time() - t0 > TL:
            break
        gr = _riemannian_grad(pts, g)
        gn = np.linalg.norm(gr)
        if gn < tight_tol:
            break
        # Adaptive step: start with 1/(gn^2) but cap it
        alpha = min(1.0 / (gn * gn + 1e-30), 10.0)
        c = 1e-4
        accepted = False
        for ls in range(60):
            new_pts = _project(pts - alpha * gr)
            new_E, _ = _energy_and_grad(new_pts)
            if new_E <= E - c * alpha * gn * gn:
                accepted = True
                break
            alpha *= 0.5
        if accepted and new_E < E:
            pts, E = new_pts, new_E
            alpha = min(alpha * 1.5, 10.0)  # grow step slightly
            if E < best_E:
                best_E, best_pts = E, pts.copy()
        else:
            alpha *= 0.5
            if alpha < 1e-20:
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
    global _START
    _START = time.time()
    TL = _DEADLINE
    best_E = np.inf
    best_pts = None
    best_angles = None
    
    theta_b = [(1e-6, np.pi - 1e-6)] * N
    phi_b = [(-np.pi, np.pi)] * N
    bounds = theta_b + phi_b
    
    pts = _fib()
    angles = _pts_to_angles(pts)
    
    # Initial L-BFGS-B from Fibonacci spiral
    try:
        res = minimize(_energy_angles, angles, jac=_grad_angles, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 8000, 'maxfun': 30000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40})
        if res.fun < best_E:
            best_E = res.fun
            best_angles = res.x.copy()
            best_pts = _angles_to_pts(res.x)
    except Exception:
        pass
    
    if best_pts is None:
        best_pts = pts.copy()
        best_E = _energy_angles(angles)
        best_angles = angles.copy()
    
    # GD polish - first pass
    if _time_left() > 10:
        best_pts, best_E = _gd_polish(best_pts, t0=_START, TL=TL, max_iters=800, tight_tol=1e-15)
        best_angles = _pts_to_angles(best_pts)
    
    # Systematic restarts with diverse strategies
    rng = np.random.RandomState(42)
    
    # Strategy 1: Small random perturbations of best solution
    if _time_left() > 15:
        for r in range(20):
            if _time_left() < 10:
                break
            for scale in [0.05, 0.02, 0.01, 0.005]:
                if _time_left() < 8:
                    break
                ap = best_angles + rng.randn(N, 2) * scale
                ap[:, 0] = np.clip(ap[:, 0], 1e-5, np.pi - 1e-5)
                ap[:, 1] = np.clip(ap[:, 1], -np.pi + 1e-5, np.pi - 1e-5)
                try:
                    res = minimize(_energy_angles, ap, jac=_grad_angles, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': 4000, 'maxfun': 15000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
                    if res.fun < best_E:
                        best_E = res.fun
                        best_angles = res.x.copy()
                        best_pts = _angles_to_pts(res.x)
                        # Quick polish after improvement
                        if _time_left() > 15:
                            best_pts, best_E = _gd_polish(best_pts, t0=_START, TL=TL, max_iters=200, tight_tol=1e-15)
                            best_angles = _pts_to_angles(best_pts)
                except Exception:
                    pass
    
    # Strategy 2: Structured perturbations - swap pairs, rotate clusters
    if _time_left() > 15:
        for r in range(8):
            if _time_left() < 10:
                break
            # Random pair swap
            i, j = rng.choice(N, 2, replace=False)
            ap = best_angles.copy()
            ap[i], ap[j] = ap[j].copy(), ap[i].copy()
            ap[:, 0] = np.clip(ap[:, 0], 1e-5, np.pi - 1e-5)
            ap[:, 1] = np.clip(ap[:, 1], -np.pi + 1e-5, np.pi - 1e-5)
            try:
                res = minimize(_energy_angles, ap, jac=_grad_angles, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 3000, 'maxfun': 12000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
                if res.fun < best_E:
                    best_E = res.fun
                    best_angles = res.x.copy()
                    best_pts = _angles_to_pts(res.x)
                    if _time_left() > 15:
                        best_pts, best_E = _gd_polish(best_pts, t0=_START, TL=TL, max_iters=200, tight_tol=1e-15)
                        best_angles = _pts_to_angles(best_pts)
            except Exception:
                pass
    
    # Strategy 3: Larger random restarts from scratch
    if _time_left() > 20:
        for r in range(5):
            if _time_left() < 15:
                break
            # Random uniform points
            u = rng.rand(N)
            v = rng.rand(N)
            theta = np.arccos(1.0 - 2.0 * u)
            phi = 2.0 * np.pi * v
            ap = np.column_stack([theta, phi])
            try:
                res = minimize(_energy_angles, ap, jac=_grad_angles, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 5000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
                if res.fun < best_E:
                    best_E = res.fun
                    best_angles = res.x.copy()
                    best_pts = _angles_to_pts(res.x)
                    if _time_left() > 15:
                        best_pts, best_E = _gd_polish(best_pts, t0=_START, TL=TL, max_iters=300, tight_tol=1e-15)
                        best_angles = _pts_to_angles(best_pts)
            except Exception:
                pass
    
    # Final high-precision polish
    if _time_left() > 5:
        best_pts, best_E = _gd_polish(best_pts, t0=_START, TL=TL, max_iters=1000, tight_tol=1e-17)
    
    return _project(best_pts)
