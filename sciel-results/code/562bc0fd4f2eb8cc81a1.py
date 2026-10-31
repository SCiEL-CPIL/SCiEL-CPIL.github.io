"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum area of any triangle formed by three points."""
    return float(np.min(_all_areas(pts)))

def _soft_min_area(pts, beta=80.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coords."""
    x, y = p
    # Convert to barycentric
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    eps = 1e-9
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    s = a + b + c
    a /= s; b /= s; c /= s
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _objective_smooth(flat_pts, beta=80.0):
    """Objective to minimize: negative soft-min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta)

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _initial_placement_structured():
    """Structured initial placement with 3-fold symmetry."""
    pts = []
    # 3 points near vertices (pulled inward)
    m = 0.10
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.12
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # Center
    pts.append((V0 + V1 + V2) / 3)
    # 4 points in a ring around center
    cx, cy = 1/3, H/3
    r = 0.14
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def _initial_placement_grid():
    """Triangular grid initial placement."""
    pts = []
    y0 = 0.06
    for i in range(4):
        x = 0.06 + i * (0.88 / 3)
        pts.append([x, y0])
    y1 = 0.34
    for i in range(4):
        x = 0.14 + i * (0.72 / 3)
        pts.append([x, y1])
    y2 = 0.60
    for i in range(3):
        x = 0.24 + i * (0.52 / 2)
        pts.append([x, y2])
    return np.array(pts)

def _initial_placement_random(seed):
    """Random initial placement inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def _optimize_from_start(start_pts, beta, maxiter, deadline_fn):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        res = minimize(
            lambda x, b=beta: _objective_smooth(x, b),
            start_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt_pts = _clamp_all(res.x.reshape(-1, 2))
        return opt_pts, _min_area(opt_pts)
    except Exception:
        return None, -1.0

def _nelder_mead_polish(pts, deadline_fn):
    """Nelder-Mead polish with hard objective."""
    try:
        res = minimize(
            _objective_hard,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        nm_pts = _clamp_all(res.x.reshape(-1, 2))
        return nm_pts, _min_area(nm_pts)
    except Exception:
        return pts, _min_area(pts)

def _coordinate_descent_polish(pts, n_iters=500, step_sizes=None):
    """Coordinate descent polish: optimize each point one at a time."""
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current = pts.copy()
    current_score = _min_area(current)
    
    for step in step_sizes:
        for _ in range(n_iters // len(step_sizes)):
            improved = False
            for i in range(N):
                # Try small perturbations in x and y
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current.copy()
                        trial[i] = _clamp_to_triangle(current[i] + np.array([dx, dy]))
                        trial_score = _min_area(trial)
                        if trial_score > current_score + 1e-15:
                            current = trial
                            current_score = trial_score
                            improved = True
            if not improved:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement_structured())
    init_candidates.append(_initial_placement_grid())
    for seed in [1, 7, 13, 29, 42, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        init_candidates.append(_initial_placement_random(seed))
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Multi-start smooth optimization with varying beta
    # Use fewer betas but more iterations for better convergence
    for init in init_candidates:
        if time_left() < 10.0:
            break
        for beta in [50.0, 100.0, 200.0]:
            if time_left() < 6.0:
                break
            opt_pts, opt_score = _optimize_from_start(init, beta, 1200, time_left)
            if opt_pts is not None and opt_score > best_score:
                best_pts = opt_pts.copy()
                best_score = opt_score
    
    if best_pts is None:
        best_pts = _initial_placement_structured()
        best_score = _min_area(best_pts)
    
    # Phase 2: Perturbation-based multi-start from best
    rng = np.random.default_rng(12345)
    n_perturb = 20
    for trial in range(n_perturb):
        if time_left() < 8.0:
            break
        scale = 0.012 + 0.006 * trial
        trial_pts = best_pts + rng.normal(0, scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        
        for beta in [80.0, 160.0]:
            if time_left() < 5.0:
                break
            opt_pts, opt_score = _optimize_from_start(trial_pts, beta, 800, time_left)
            if opt_pts is not None and opt_score > best_score:
                best_pts = opt_pts.copy()
                best_score = opt_score
    
    # Phase 3: Simulated annealing with hard objective - improved
    if time_left() > 20.0:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.03
        n_sa = 12000
        for iteration in range(n_sa):
            if time_left() < 6.0:
                break
            
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] = _clamp_to_triangle(current_pts[idx] + delta)
            
            new_score = _min_area(new_pts)
            
            if new_score > current_score:
                accept = True
            else:
                diff = new_score - current_score
                accept = rng.random() < np.exp(diff / max(T * 0.03, 1e-12))
            
            if accept:
                current_pts = new_pts
                current_score = new_score
                if new_score > best_score:
                    best_pts = current_pts.copy()
                    best_score = new_score
            
            T *= 0.9992
            if T < 0.001:
                T = 0.001
    
    # Phase 4: Coordinate descent polish
    if time_left() > 15.0:
        cd_pts, cd_score = _coordinate_descent_polish(best_pts, n_iters=1000)
        if cd_score > best_score:
            best_pts = cd_pts.copy()
            best_score = cd_score
    
    # Phase 5: Nelder-Mead polish
    if time_left() > 12.0:
        nm_pts, nm_score = _nelder_mead_polish(best_pts, time_left)
        if nm_score > best_score:
            best_pts = nm_pts.copy()
            best_score = nm_score
    
    # Phase 6: Final high-precision L-BFGS-B polish
    if time_left() > 8.0:
        try:
            res_final = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_pts = final_pts.copy()
                best_score = final_score
        except Exception:
            pass
    
    # Phase 7: Final coordinate descent + Nelder-Mead
    if time_left() > 5.0:
        cd_pts2, cd_score2 = _coordinate_descent_polish(best_pts, n_iters=500)
        if cd_score2 > best_score:
            best_pts = cd_pts2.copy()
            best_score = cd_score2
    
    if time_left() > 3.0:
        nm_pts2, nm_score2 = _nelder_mead_polish(best_pts, time_left)
        if nm_score2 > best_score:
            best_pts = nm_pts2.copy()
            best_score = nm_score2
    
    # Final clamp
    best_pts = _clamp_all(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
