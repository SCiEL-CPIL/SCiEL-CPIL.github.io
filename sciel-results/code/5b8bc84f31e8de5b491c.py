"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(n, 1)
    return dists_sq[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d_sq = _pairwise_dists_sq(pts)
    d_sq = np.maximum(d_sq, 1e-12)
    return (np.max(d_sq) / np.min(d_sq))

def _soft_objective(pts, beta):
    """
    Smooth surrogate for the ratio of max to min squared distances.
    Uses log-sum-exp for soft max and soft min.
    Objective: soft_max(d_sq) / soft_min(d_sq)
    """
    d_sq = _pairwise_dists_sq(pts)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max: (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, subtract max
    max_d_sq = np.max(d_sq)
    soft_max_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min: - (1/beta) * log(sum(exp(-beta * x)))
    min_d_sq = np.min(d_sq)
    soft_min_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min_sq < 1e-10:
        return 1e10
        
    return soft_max_sq / soft_min_sq

# Fourier-series radial parameterization
# r(theta) = r0 * (1 + sum_k (a_k * cos(k*theta) + b_k * sin(k*theta)))
# Using even harmonics k = 2, 4, 6 to enforce central symmetry (r(theta) = r(theta+pi))
# Parameters: [r0, a2, b2, a4, b4, a6, b6] -> 7 parameters instead of 32

N_FOURIER_TERMS = 3  # Use harmonics k=2, 4, 6
N_FOURIER_PARAMS = 1 + 2 * N_FOURIER_TERMS  # r0 + 2*N_FOURIER_TERMS

def _fourier_to_points(fourier_params):
    """
    Convert Fourier parameters to 16 points in 2D.
    Points are placed at equally spaced angles, with radius given by Fourier series.
    """
    r0 = fourier_params[0]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    radii = np.ones(N)
    for k in range(N_FOURIER_TERMS):
        harmonic = 2 * (k + 1)  # k=0 -> harmonic 2, k=1 -> harmonic 4, etc.
        a = fourier_params[1 + 2 * k]
        b = fourier_params[2 + 2 * k]
        radii += a * np.cos(harmonic * angles) + b * np.sin(harmonic * angles)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1 * abs(r0) + 1e-6)
    
    pts = np.column_stack([r0 * radii * np.cos(angles), r0 * radii * np.sin(angles)])
    return pts

def _fourier_score(fourier_params):
    """Score the configuration defined by Fourier parameters."""
    pts = _fourier_to_points(fourier_params)
    return _score(pts)

def _fourier_soft_objective(fourier_params, beta):
    """Soft objective for Fourier parameterization."""
    pts = _fourier_to_points(fourier_params)
    return _soft_objective(pts, beta)

def _init_hexagonal_cluster():
    """
    Create a hexagonal cluster of 16 points.
    """
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]

def _init_rings():
    """Two rings: 8 inner, 8 outer."""
    inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi/8
    r_in = 1.0
    r_out = 1.8
    inner = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    outer = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    return np.vstack([inner, outer])

def _init_grid():
    """4x4 grid."""
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts

def _init_random(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-2, 2, size=(N, D))

def _init_fourier_random(seed):
    """Random initialization in Fourier parameter space."""
    rng = np.random.RandomState(seed)
    r0 = rng.uniform(0.5, 2.0)
    coeffs = rng.uniform(-0.3, 0.3, size=2 * N_FOURIER_TERMS)
    return np.concatenate([[r0], coeffs])

def _init_fourier_circle():
    """Perfect circle (only r0 nonzero)."""
    params = np.zeros(N_FOURIER_PARAMS)
    params[0] = 1.0
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    # === Phase 1: Fourier-space optimization (low-dimensional, smooth) ===
    
    # Initial Fourier candidates
    fourier_candidates = [
        _init_fourier_circle(),
        _init_fourier_random(1),
        _init_fourier_random(2),
        _init_fourier_random(3),
        _init_fourier_random(4),
        _init_fourier_random(5),
    ]
    
    # Score initial Fourier candidates
    best_fourier_params = None
    for fp in fourier_candidates:
        pts = _fourier_to_points(fp)
        sc = _score(pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()
            best_fourier_params = fp.copy()
    
    # Fourier-space DE (7D search space - much faster)
    if time.time() < deadline - 60:
        try:
            fourier_bounds = [
                (0.3, 3.0),      # r0
                (-0.4, 0.4),     # a2
                (-0.4, 0.4),     # b2
                (-0.2, 0.2),     # a4
                (-0.2, 0.2),     # b4
                (-0.1, 0.1),     # a6
                (-0.1, 0.1),     # b6
            ]
            
            beta_de = 15.0
            res_de_fourier = differential_evolution(
                lambda p: _fourier_soft_objective(p, beta_de),
                bounds=fourier_bounds,
                seed=42,
                maxiter=150,
                tol=1e-8,
                polish=False,
                workers=1
            )
            fp_de = res_de_fourier.x
            pts_de = _fourier_to_points(fp_de)
            sc_de = _score(pts_de)
            if sc_de < best_score:
                best_score = sc_de
                best_pts = pts_de.copy()
                best_fourier_params = fp_de.copy()
        except Exception:
            pass
    
    # Fourier-space L-BFGS-B polishing with annealed beta
    if best_fourier_params is not None and time.time() < deadline - 30:
        for beta in [5.0, 15.0, 40.0, 100.0]:
            if time.time() > deadline - 10:
                break
            try:
                res_fourier = minimize(
                    lambda p: _fourier_soft_objective(p, beta),
                    best_fourier_params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14}
                )
                fp_opt = res_fourier.x
                pts_opt = _fourier_to_points(fp_opt)
                sc_opt = _score(pts_opt)
                if sc_opt < best_score:
                    best_score = sc_opt
                    best_pts = pts_opt.copy()
                    best_fourier_params = fp_opt.copy()
            except Exception:
                pass
    
    # === Phase 2: Unconstrained point-space optimization (refine) ===
    
    # Also try standard initial candidates in point space
    point_candidates = [
        _init_hexagonal_cluster(),
        _init_rings(),
        _init_grid(),
        _init_random(1),
        _init_random(2),
    ]
    
    for pts in point_candidates:
        sc = _score(pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()
    
    # Point-space DE (32D - more expensive, use remaining time)
    if time.time() < deadline - 40:
        try:
            bounds = [(-5, 5)] * (N * D)
            beta_de_pt = 10.0
            res_de_pt = differential_evolution(
                lambda p: _soft_objective(p.reshape(N, D), beta_de_pt),
                bounds=bounds,
                seed=123,
                maxiter=100,
                tol=1e-8,
                polish=False,
                workers=1
            )
            pts_de_pt = res_de_pt.x.reshape(N, D)
            sc_de_pt = _score(pts_de_pt)
            if sc_de_pt < best_score:
                best_score = sc_de_pt
                best_pts = pts_de_pt.copy()
        except Exception:
            pass
    
    # Point-space L-BFGS-B polishing
    if best_pts is not None and time.time() < deadline - 15:
        def polish(pts, beta, method='L-BFGS-B'):
            if time.time() > deadline - 5:
                return pts, _score(pts)
            x0 = pts.flatten()
            def obj_func(x):
                return _soft_objective(x.reshape(N, D), beta)
            try:
                res = minimize(obj_func, x0, method=method, options={'maxiter': 400, 'ftol': 1e-12})
                pts_opt = res.x.reshape(N, D)
                sc_opt = _score(pts_opt)
                return pts_opt, sc_opt
            except Exception:
                return pts, _score(pts)
        
        for beta in [10.0, 30.0, 80.0]:
            if time.time() > deadline - 10:
                break
            pts_polished, sc_polished = polish(best_pts, beta)
            if sc_polished < best_score:
                best_score = sc_polished
                best_pts = pts_polished.copy()
            
            # Powell on exact objective
            if time.time() < deadline - 8:
                try:
                    res_powell = minimize(
                        lambda x: _score(x.reshape(N, D)), 
                        pts_polished.flatten(), 
                        method='Powell', 
                        options={'maxiter': 150, 'maxfev': 1500}
                    )
                    pts_powell = res_powell.x.reshape(N, D)
                    sc_powell = _score(pts_powell)
                    if sc_powell < best_score:
                        best_score = sc_powell
                        best_pts = pts_powell.copy()
                except Exception:
                    pass
    
    # Final random perturbations + polish
    if best_pts is not None and time.time() < deadline - 12:
        rng = np.random.default_rng(999)
        for _ in range(8):
            if time.time() > deadline - 6:
                break
            noise = rng.normal(0, 0.08, size=(N, D))
            pts_perturbed = best_pts + noise
            pts_polished, sc_polished = polish(pts_perturbed, 25.0)
            if sc_polished < best_score:
                best_score = sc_polished
                best_pts = pts_polished.copy()
    
    # Final check for distinctness
    if best_pts is None:
        best_pts = _init_hexagonal_cluster()
        
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-8:
        best_pts = _init_hexagonal_cluster()

    return best_pts
# EVOLVE-BLOCK-END
