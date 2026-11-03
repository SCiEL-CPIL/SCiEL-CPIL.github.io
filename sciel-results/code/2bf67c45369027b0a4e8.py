import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

# Precompute Hermite polynomial values on a fine grid
# Use a moderate range; the Gaussian e^{-2πx²} decays rapidly
_X = np.linspace(-5, 5, 4000)
_DX = _X[1] - _X[0]
_H0 = eval_hermite(0, _X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
_EXP_2PI = np.exp(-2 * np.pi * _X**2)
_EXP_4PI = np.exp(-4 * np.pi * _X**2)

# Constants for a12 from P(0) = 0
# H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640 (verify)
_H0_0 = eval_hermite(0, 0)
_H4_0 = eval_hermite(4, 0)
_H8_0 = eval_hermite(8, 0)
_H12_0 = eval_hermite(12, 0)

def _objective_2d(theta):
    """Objective depending only on direction (scale-invariant).
    theta = [theta1, theta2] parameterize the direction on the unit sphere.
    """
    t1, t2 = theta
    # Unit vector: (cos t1, sin t1 * cos t2, sin t1 * sin t2)
    a0_dir = np.cos(t1)
    a4_dir = np.sin(t1) * np.cos(t2)
    a8_dir = np.sin(t1) * np.sin(t2)
    
    # Since objective is scale-invariant, set r=1
    a0 = a0_dir
    a4 = a4_dir
    a8 = a8_dir
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    
    norm2_sq = np.sum(P**2 * _EXP_2PI) * _DX
    norm4_sq = np.sum(P**4 * _EXP_4PI) * _DX
    
    norm2_sq_hat = np.sum(P_minus**2 * _EXP_2PI) * _DX
    norm4_sq_hat = np.sum(P_minus**4 * _EXP_4PI) * _DX
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
    
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    return obj

def _objective_3d(coeffs):
    """Objective for 3D coefficients (a0, a4, a8)."""
    a0, a4, a8 = coeffs
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    
    norm2_sq = np.sum(P**2 * _EXP_2PI) * _DX
    norm4_sq = np.sum(P**4 * _EXP_4PI) * _DX
    
    norm2_sq_hat = np.sum(P_minus**2 * _EXP_2PI) * _DX
    norm4_sq_hat = np.sum(P_minus**4 * _EXP_4PI) * _DX
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
    
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    return obj

def construct():
    t0 = time.time()
    DEADLINE = 160.0  # Safety margin (evaluator kills at 210s)
    
    def _elapsed_ok():
        return time.time() - t0 < DEADLINE
    
    # Seed solution
    x0_seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    seed_score = _objective_3d(x0_seed)
    
    best_coeffs = x0_seed.copy()
    best_score = seed_score
    
    def _update_best(coeffs, score):
        nonlocal best_coeffs, best_score
        if score < best_score - 1e-12:
            best_coeffs = coeffs.copy()
            best_score = score
    
    # Phase 1: Global search in 2D spherical coordinates via differential_evolution
    if _elapsed_ok():
        try:
            # theta1 in (-pi, pi), theta2 in (-pi, pi)
            bounds_2d = [(-np.pi, np.pi), (-np.pi, np.pi)]
            de_res = differential_evolution(
                _objective_2d,
                bounds=bounds_2d,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=True
            )
            if _elapsed_ok() and de_res.fun < best_score:
                t1, t2 = de_res.x
                a0 = np.cos(t1)
                a4 = np.sin(t1) * np.cos(t2)
                a8 = np.sin(t1) * np.sin(t2)
                _update_best(np.array([a0, a4, a8]), de_res.fun)
        except Exception:
            pass
    
    # Phase 2: Additional DE runs with different seeds/params if time remains
    if _elapsed_ok():
        for seed_try in [1, 7, 123]:
            if not _elapsed_ok():
                break
            try:
                de_res2 = differential_evolution(
                    _objective_2d,
                    bounds=bounds_2d,
                    seed=seed_try,
                    maxiter=50,
                    popsize=10,
                    tol=1e-7,
                    mutation=(0.5, 1.0),
                    recombination=0.8,
                    polish=True
                )
                if _elapsed_ok() and de_res2.fun < best_score:
                    t1, t2 = de_res2.x
                    a0 = np.cos(t1)
                    a4 = np.sin(t1) * np.cos(t2)
                    a8 = np.sin(t1) * np.sin(t2)
                    _update_best(np.array([a0, a4, a8]), de_res2.fun)
            except Exception:
                pass
    
    # Phase 3: Local polish in 3D from best 2D solution and seed
    if _elapsed_ok():
        polish_starts = [best_coeffs.copy(), x0_seed.copy()]
        # Add some perturbations
        for _ in range(3):
            if not _elapsed_ok():
                break
            polish_starts.append(best_coeffs + np.random.normal(0, 0.05, 3))
        
        for ps in polish_starts:
            if not _elapsed_ok():
                break
            try:
                res = minimize(_objective_3d, ps, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                if res.fun < best_score:
                    _update_best(res.x, res.fun)
            except Exception:
                pass
    
    # Phase 4: Fine local polish in 2D spherical coordinates
    if _elapsed_ok():
        # Convert best_coeffs back to spherical
        r = np.linalg.norm(best_coeffs)
        if r > 1e-10:
            t1_best = np.arccos(np.clip(best_coeffs[0]/r, -1, 1))
            t2_best = np.arctan2(best_coeffs[2], best_coeffs[1])
        else:
            t1_best, t2_best = 0.0, 0.0
        
        for start_2d in [(t1_best, t2_best), (0, 0), (np.pi/4, np.pi/4), (-np.pi/3, np.pi/3)]:
            if not _elapsed_ok():
                break
            try:
                res2d = minimize(_objective_2d, np.array(start_2d), method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14})
                if res2d.fun < best_score:
                    t1, t2 = res2d.x
                    a0 = np.cos(t1)
                    a4 = np.sin(t1) * np.cos(t2)
                    a8 = np.sin(t1) * np.sin(t2)
                    _update_best(np.array([a0, a4, a8]), res2d.fun)
            except Exception:
                pass
    
    return best_coeffs

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

# Precompute Hermite polynomial values on a fine grid
_X = np.linspace(-5, 5, 4000)
_DX = _X[1] - _X[0]
_H0 = eval_hermite(0, _X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
_EXP_2PI = np.exp(-2 * np.pi * _X**2)
_EXP_4PI = np.exp(-4 * np.pi * _X**2)

def _objective_2d(theta):
    t1, t2 = theta
    a0_dir = np.cos(t1)
    a4_dir = np.sin(t1) * np.cos(t2)
    a8_dir = np.sin(t1) * np.sin(t2)
    a0 = a0_dir
    a4 = a4_dir
    a8 = a8_dir
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    norm2_sq = np.sum(P**2 * _EXP_2PI) * _DX
    norm4_sq = np.sum(P**4 * _EXP_4PI) * _DX
    norm2_sq_hat = np.sum(P_minus**2 * _EXP_2PI) * _DX
    norm4_sq_hat = np.sum(P_minus**4 * _EXP_4PI) * _DX
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
    return (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)

def _objective_3d(coeffs):
    a0, a4, a8 = coeffs
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    norm2_sq = np.sum(P**2 * _EXP_2PI) * _DX
    norm4_sq = np.sum(P**4 * _EXP_4PI) * _DX
    norm2_sq_hat = np.sum(P_minus**2 * _EXP_2PI) * _DX
    norm4_sq_hat = np.sum(P_minus**4 * _EXP_4PI) * _DX
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
    return (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)

def construct():
    t0 = time.time()
    DEADLINE = 160.0
    def _ok():
        return time.time() - t0 < DEADLINE
    x0_seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_coeffs = x0_seed.copy()
    best_score = _objective_3d(x0_seed)
    def _upd(c, s):
        nonlocal best_coeffs, best_score
        if s < best_score - 1e-12:
            best_coeffs = c.copy()
            best_score = s
    if _ok():
        try:
            bounds_2d = [(-np.pi, np.pi), (-np.pi, np.pi)]
            de_res = differential_evolution(_objective_2d, bounds=bounds_2d, seed=42, maxiter=80, popsize=15, tol=1e-8, mutation=(0.3, 1.0), recombination=0.9, polish=True)
            if _ok() and de_res.fun < best_score:
                t1, t2 = de_res.x
                _upd(np.array([np.cos(t1), np.sin(t1)*np.cos(t2), np.sin(t1)*np.sin(t2)]), de_res.fun)
        except Exception:
            pass
    if _ok():
        for ss in [1, 7, 123]:
            if not _ok(): break
            try:
                r2 = differential_evolution(_objective_2d, bounds=bounds_2d, seed=ss, maxiter=50, popsize=10, tol=1e-7, mutation=(0.5, 1.0), recombination=0.8, polish=True)
                if _ok() and r2.fun < best_score:
                    t1, t2 = r2.x
                    _upd(np.array([np.cos(t1), np.sin(t1)*np.cos(t2), np.sin(t1)*np.sin(t2)]), r2.fun)
            except Exception:
                pass
    if _ok():
        starts3 = [best_coeffs.copy(), x0_seed.copy()]
        for _ in range(3):
            if not _ok(): break
            starts3.append(best_coeffs + np.random.normal(0, 0.05, 3))
        for ps in starts3:
            if not _ok(): break
            try:
                r = minimize(_objective_3d, ps, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                if r.fun < best_score: _upd(r.x, r.fun)
            except Exception: pass
    if _ok():
        r = np.linalg.norm(best_coeffs)
        t1b = np.arccos(np.clip(best_coeffs[0]/r, -1, 1)) if r > 1e-10 else 0.0
        t2b = np.arctan2(best_coeffs[2], best_coeffs[1]) if r > 1e-10 else 0.0
        for s2 in [(t1b, t2b), (0, 0), (np.pi/4, np.pi/4), (-np.pi/3, np.pi/3)]:
            if not _ok(): break
            try:
                r2 = minimize(_objective_2d, np.array(s2), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14})
                if r2.fun < best_score:
                    t1, t2 = r2.x
                    _upd(np.array([np.cos(t1), np.sin(t1)*np.cos(t2), np.sin(t1)*np.sin(t2)]), r2.fun)
            except Exception: pass
    return best_coeffs
# EVOLVE-BLOCK-END
