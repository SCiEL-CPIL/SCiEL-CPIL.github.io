"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all triangle areas."""
    # pts shape (N, 2)
    # For each triangle (i,j,k): area = 0.5 * |cross(pj-pi, pk-pi)|
    areas = np.empty(_N_TRI)
    for t, (i, j, k) in enumerate(_TRI_IDX):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        areas[t] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    areas = _all_areas(pts)
    return float(np.min(areas))

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area using log-sum-exp.
    Returns -logsumexp(-log(areas))/beta which approximates min(areas).
    """
    areas = _all_areas(pts)
    # Use log-sum-exp for numerical stability
    log_areas = np.log(areas + 1e-15)
    # soft min = -logsumexp(-log_areas) / beta * ... 
    # Actually: softmin(x) ≈ -(1/beta)*logsumexp(-beta*x)
    # But we want to MINIMIZE the soft-min of areas, i.e., maximize it
    # soft_min(areas) = -(1/beta) * log(sum(exp(-beta*areas)))
    # For stability: let m = min(areas), then soft_min ≈ m - (1/beta)*log(sum(exp(-beta*(areas-m))))
    m = np.min(areas)
    shifted = -beta * (areas - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-10
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _objective_smooth(flat_pts, beta=50.0):
    """Objective to minimize: negative soft-min area (smooth)."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area (non-smooth, for final polish)."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _initial_placement():
    """Structured initial placement: triangular lattice pattern."""
    pts = []
    # Row 0 (bottom): 4 points
    y0 = 0.05
    for i in range(4):
        x = 0.05 + i * (0.9 / 3)
        pts.append([x, y0])
    # Row 1 (middle): 4 points
    y1 = 0.35
    for i in range(4):
        x = 0.15 + i * (0.7 / 3)
        pts.append([x, y1])
    # Row 2 (top): 3 points
    y2 = 0.65
    for i in range(3):
        x = 0.25 + i * (0.5 / 2)
        pts.append([x, y2])
    return np.array(pts)

def _initial_placement_symmetric():
    """Symmetric initial placement using barycentric coordinates."""
    # Use a 3x3 triangular grid in barycentric coords, pick 11 points
    # Barycentric: (a, b, c) with a+b+c=1, a,b,c >= 0
    # Point = a*V0 + b*V1 + c*V2
    pts = []
    # Symmetric arrangement: 3 near vertices, 3 near edge midpoints, 1 center, 4 interior
    # Near vertices (pulled inward)
    margin = 0.08
    pts.append(margin * V0 + (1-margin)/2 * V1 + (1-margin)/2 * V2)  # near V0
    pts.append((1-margin)/2 * V0 + margin * V1 + (1-margin)/2 * V2)  # near V1
    pts.append((1-margin)/2 * V0 + (1-margin)/2 * V1 + margin * V2)  # near V2
    # Edge midpoints (pulled inward)
    m2 = 0.05
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)  # mid V0-V1
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)  # mid V1-V2
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)  # mid V0-V2
    # Center
    pts.append((V0 + V1 + V2) / 3)
    # 4 more interior points (symmetric around center)
    cx, cy = 1/3, H/3
    r = 0.12
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _initial_placement_random(seed):
    """Random initial placement inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        # Random point in triangle via barycentric
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Generate multiple diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement())
    init_candidates.append(_initial_placement_symmetric())
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        init_candidates.append(_initial_placement_random(seed))
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Smooth optimization from each initialization with multiple beta values
    for init in init_candidates:
        if time_left() < 8.0:
            break
        flat_init = init.flatten()
        
        for beta in [30.0, 50.0, 80.0]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_smooth(x, b),
                    flat_init,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_pts = opt_pts.copy()
                    best_score = opt_score
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _initial_placement()
        best_score = _min_area(best_pts)
    
    # Phase 2: Multi-start perturbation + smooth optimization
    rng = np.random.default_rng(42)
    n_perturb = 15
    for trial in range(n_perturb):
        if time_left() < 6.0:
            break
        # Perturb from best
        scale = 0.03 + 0.02 * trial
        trial_pts = best_pts + rng.normal(0, scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        flat_trial = trial_pts.flatten()
        
        for beta in [40.0, 70.0]:
            if time_left() < 4.0:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_smooth(x, b),
                    flat_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_pts = opt_pts.copy()
                    best_score = opt_score
            except Exception:
                pass
    
    # Phase 3: Simulated annealing with smooth objective for exploration
    if time_left() > 15.0:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.015
        beta_sa = 40.0
        for iteration in range(3000):
            if time_left() < 5.0:
                break
            
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] = _clamp_to_triangle(current_pts[idx] + delta)
            
            # Use smooth objective for acceptance (faster, smoother)
            new_soft = _soft_min_area(new_pts, beta_sa)
            cur_soft = _soft_min_area(current_pts, beta_sa)
            
            if new_soft > cur_soft:
                accept = True
            else:
                diff = new_soft - cur_soft
                accept = rng.random() < np.exp(diff / max(T * 0.1, 1e-10))
            
            if accept:
                current_pts = new_pts
                # Use hard score for tracking best
                hard_score = _min_area(current_pts)
                current_score = hard_score
                if hard_score > best_score:
                    best_pts = current_pts.copy()
                    best_score = hard_score
            
            T *= 0.9985
            if T < 0.001:
                T = 0.001
    
    # Phase 4: Nelder-Mead local search (derivative-free, good for final refinement)
    if time_left() > 8.0:
        try:
            res_nm = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            if nm_score > best_score:
                best_pts = nm_pts.copy()
                best_score = nm_score
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish with L-BFGS-B on hard objective
    if time_left() > 4.0:
        try:
            res_final = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_pts = final_pts.copy()
                best_score = final_score
        except Exception:
            pass
    
    # Phase 6: One more Nelder-Mead pass from the polished result
    if time_left() > 3.0:
        try:
            res_nm2 = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            nm2_pts = _clamp_all(res_nm2.x.reshape(-1, 2))
            nm2_score = _min_area(nm2_pts)
            if nm2_score > best_score:
                best_pts = nm2_pts.copy()
                best_score = nm2_score
        except Exception:
            pass
    
    # Final clamp and return
    best_pts = _clamp_all(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
