"""Rewritten program using Simulated Annealing + BFGS refinement for Heilbronn triangles."""
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
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coords."""
    x, y = p
    # Convert to barycentric coordinates relative to V0, V1, V2
    # p = a*V0 + b*V1 + c*V2 where a+b+c=1, a,b,c >= 0
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # x = b + 0.5*c
    # y = H*c
    # So c = y/H, b = x - 0.5*c, a = 1 - b - c
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    eps = 1e-12
    a = max(eps, min(1.0 - 2*eps, a))
    b = max(eps, min(1.0 - 2*eps, b))
    c = max(eps, min(1.0 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _random_in_triangle(rng):
    """Generate a random point uniformly in the triangle."""
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1.0:
        r1 = 1.0 - r1
        r2 = 1.0 - r2
    return r1 * V1 + r2 * V2

def _initial_placement_symmetric():
    """Symmetric initial placement with good spread."""
    pts = []
    # Center
    pts.append([1/3, H/3])
    # 3 points near vertices (slightly inset)
    m = 0.12
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.15
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # 3 more points in a ring around center
    cx, cy = 1/3, H/3
    r = 0.22
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return _clamp_all(np.array(pts))

def _initial_placement_random(seed):
    """Random initial placement."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        pts.append(_random_in_triangle(rng))
    return _clamp_all(np.array(pts))

def _simulated_annealing(pts, time_budget, rng, t0=0.05, t_min=0.001, alpha=0.995, max_iter=None):
    """
    Simulated Annealing for maximizing min triangle area.
    Returns the best configuration found.
    """
    start_time = time.time()
    current = _clamp_all(pts.copy())
    current_score = _min_area(current)
    best = current.copy()
    best_score = current_score
    
    t = t0
    step_size = 0.08
    
    # Adaptive step size: start larger, shrink over time
    step_decay = 0.9995
    
    while time.time() - start_time < time_budget:
        # Propose a move: pick random point, move it by random offset
        idx = rng.integers(0, N)
        noise = rng.normal(0, step_size, size=2)
        trial = current.copy()
        trial[idx] = _clamp_to_triangle(trial[idx] + noise)
        
        trial_score = _min_area(trial)
        delta = trial_score - current_score
        
        # Accept if improvement, or with probability exp(delta/t)
        if delta > 0 or (t > t_min and rng.random() < np.exp(delta / t)):
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        # Cool down
        t *= alpha
        step_size *= step_decay
        if t < t_min:
            t = t_min
    
    return best, best_score

def _bfgs_refine(pts, time_budget):
    """
    BFGS refinement using a smooth objective.
    Returns the best configuration found.
    """
    start_time = time.time()
    
    def objective(flat_pts):
        p = flat_pts.reshape(-1, 2)
        p = _clamp_all(p)
        areas = _all_areas(p)
        # Use a smooth min approximation
        m = np.min(areas)
        if m <= 1e-15:
            return m
        beta = 80.0
        shifted = -beta * (areas - m)
        # Clip for numerical stability
        shifted = np.clip(shifted, -500, 500)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val = m - log_sum / beta
        return -soft_min_val
    
    res = minimize(
        objective,
        pts.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    opt_pts = _clamp_all(res.x.reshape(-1, 2))
    opt_score = _min_area(opt_pts)
    
    # Also try Nelder-Mead for non-smooth refinement
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2.0:
        def hard_obj(flat_pts):
            p = flat_pts.reshape(-1, 2)
            p = _clamp_all(p)
            return -_min_area(p)
        
        res_nm = minimize(
            hard_obj,
            opt_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
        nm_score = _min_area(nm_pts)
        if nm_score > opt_score:
            opt_pts = nm_pts
            opt_score = nm_score
    
    return opt_pts, opt_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    best_pts = None
    best_score = -1.0
    
    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement_symmetric())
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        init_candidates.append(_initial_placement_random(seed))
    
    # Phase 1: Simulated Annealing from each initialization
    # Allocate time budget based on number of initializations
    n_inits = len(init_candidates)
    sa_budget_per = min(8.0, max(3.0, (time_left() * 0.5) / n_inits))
    
    sa_results = []
    for i, init in enumerate(init_candidates):
        if time_left() < 5.0:
            break
        budget = min(sa_budget_per, time_left() - 2.0)
        if budget < 1.0:
            break
        
        rng = np.random.default_rng(1000 + i * 7)
        sa_pts, sa_score = _simulated_annealing(init, budget, rng)
        sa_results.append((sa_score, sa_pts))
        
        if sa_score > best_score:
            best_pts = sa_pts.copy()
            best_score = sa_score
    
    # Sort SA results and take top 4 for BFGS refinement
    sa_results.sort(key=lambda x: x[0], reverse=True)
    top_results = sa_results[:4]
    
    # Phase 2: BFGS + Nelder-Mead refinement on top SA results
    bfgs_budget_per = min(15.0, max(5.0, (time_left() * 0.4) / len(top_results)))
    
    for score, pts in top_results:
        if time_left() < 5.0:
            break
        budget = min(bfgs_budget_per, time_left() - 2.0)
        if budget < 2.0:
            break
        
        try:
            ref_pts, ref_score = _bfgs_refine(pts, budget)
            if ref_score > best_score:
                best_pts = ref_pts.copy()
                best_score = ref_score
        except Exception:
            pass
    
    # Phase 3: More SA rounds from the best configuration with different params
    if best_pts is None:
        best_pts = _initial_placement_symmetric()
        best_score = _min_area(best_pts)
    
    if time_left() > 10.0:
        # Run SA from the best point with different temperature schedules
        for t0_val in [0.03, 0.06, 0.01]:
            if time_left() < 5.0:
                break
            budget = min(6.0, time_left() - 2.0)
            if budget < 2.0:
                break
            rng = np.random.default_rng(9999 + int(t0_val * 10000))
            sa_pts, sa_score = _simulated_annealing(best_pts, budget, rng, t0=t0_val, alpha=0.997)
            if sa_score > best_score:
                best_pts = sa_pts.copy()
                best_score = sa_score
    
    # Final polish with BFGS on the overall best
    if time_left() > 5.0:
        try:
            final_pts, final_score = _bfgs_refine(best_pts, min(10.0, time_left() - 1.0))
            if final_score > best_score:
                best_pts = final_pts.copy()
                best_score = final_score
        except Exception:
            pass
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
