"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute all 165 triangle index combinations
_tri_list = list(combinations(range(N), 3))
_I = np.array([t[0] for t in _tri_list])
_J = np.array([t[1] for t in _tri_list])
_K = np.array([t[2] for t in _tri_list])

def _all_areas(pts):
    """Vectorized computation of all 165 triangle areas."""
    Pi = pts[_I]
    Pj = pts[_J]
    Pk = pts[_K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_area(pts):
    """Minimum area of any triangle formed by 3 points."""
    return float(np.min(_all_areas(pts)))

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area via log-sum-exp."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _penalty(pts):
    """Differentiable penalty for points outside the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    # Three half-plane constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    v1 = np.maximum(0.0, -y)
    v2 = np.maximum(0.0, y - S3 * x)
    v3 = np.maximum(0.0, y - S3 * (1.0 - x))
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective_smooth(flat_pts, beta=50.0):
    """Objective to minimize: -soft_min_area + penalty."""
    pts = flat_pts.reshape(-1, 2)
    sm = _soft_min_area(pts, beta)
    pen = _penalty(pts)
    return -sm + 1000.0 * pen

def _objective_hard(flat_pts):
    """Objective to minimize: -min_area + penalty."""
    pts = flat_pts.reshape(-1, 2)
    ma = _min_area(pts)
    pen = _penalty(pts)
    return -ma + 1000.0 * pen

def _clamp_to_triangle(p):
    """Clamp a single point to be inside the equilateral triangle."""
    x, y = float(p[0]), float(p[1])
    eps = 1e-12
    # Barycentric coordinates
    c = 2.0 * y / S3
    b = x - 0.5 * c
    a = 1.0 - b - c
    a = max(eps, min(1.0 - 2*eps, a))
    b = max(eps, min(1.0 - 2*eps, b))
    c = max(eps, min(1.0 - 2*eps, c))
    s = a + b + c
    a /= s; b /= s; c /= s
    return np.array([b + 0.5*c, H*c])

def _clamp_all(pts):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    while True:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_placement_structured():
    """Structured initial placement with 3-fold symmetry."""
    pts = []
    # 3 points near vertices
    m = 0.10
    pts.append(np.array([0.5*(1-m), H*(1-m)/2]))
    pts.append(np.array([0.5*(1+m), H*(1-m)/2]))
    pts.append(np.array([0.5, H*(1-m)]))
    # 3 points near edge midpoints
    m2 = 0.08
    pts.append(np.array([0.5, H*m2]))
    pts.append(np.array([1.0 - 0.5*(1-m2), H*(1-m2)/2]))
    pts.append(np.array([0.5*(1-m2), H*(1-m2)/2]))
    # 1 center
    pts.append(np.array([0.5, H/3.0]))
    # 4 points in a ring around center
    cx, cy = 0.5, H/3.0
    r = 0.14
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append(np.array([px, py]))
    return np.array(pts)

def _initial_placement_random(rng, seed):
    """Random initial placement."""
    rng_local = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng_local))
    return np.array(pts)

def _coordinate_descent_polish(pts, time_budget=3.0):
    """Coordinate descent: optimize each point's x and y independently."""
    start = time.time()
    pts = pts.copy()
    current_score = _min_area(pts)
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        if time.time() - start > time_budget:
            break
        for _ in range(200):
            if time.time() - start > time_budget:
                break
            improved = False
            for i in range(N):
                for dim in range(2):
                    # Try small perturbations in each direction
                    for direction in [-1, 1]:
                        new_pts = pts.copy()
                        new_pts[i, dim] += direction * step
                        new_pts = _clamp_all(new_pts)
                        new_score = _min_area(new_pts)
                        if new_score > current_score + 1e-14:
                            pts = new_pts
                            current_score = new_score
                            improved = True
                            break
            if not improved:
                break
    
    return pts, current_score

def _simulated_annealing(initial_points, time_limit, rng):
    """Adaptive simulated annealing with penalty-based objective."""
    current_pts = initial_points.copy()
    current_score = _min_area(current_pts) - 1000.0 * _penalty(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = 0.03
    cooling_rate = 0.9995
    min_T = 1e-6
    step_scale = 0.04
    
    start_time = time.time()
    no_improve = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.integers(0, N)
        noise = rng.normal(0, step_scale, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] += noise
        
        new_score = _min_area(new_pts) - 1000.0 * _penalty(new_pts)
        delta = new_score - current_score
        
        if delta > 0 or (T > min_T and rng.random() < np.exp(delta / max(T, 1e-10))):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Adaptive step size
        if no_improve > 150:
            step_scale = max(0.002, step_scale * 0.85)
            no_improve = 0
        elif no_improve == 0:
            step_scale = min(0.08, step_scale * 1.03)
        
        # Occasional random re-seed of one point
        if rng.random() < 0.003:
            idx = rng.integers(0, N)
            new_pts = current_pts.copy()
            new_pts[idx] = _random_point_in_triangle(rng)
            new_score = _min_area(new_pts) - 1000.0 * _penalty(new_pts)
            delta = new_score - current_score
            if delta > 0 or (T > min_T and rng.random() < np.exp(delta / max(T, 1e-10))):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        T *= cooling_rate
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = -1e10
    
    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(_initial_placement_structured())
    for seed in [1, 7, 13, 29, 42, 99, 123, 256, 512, 1024]:
        init_candidates.append(_initial_placement_random(rng, seed))
    
    # Phase 1: Smooth optimization from each initialization
    for init in init_candidates:
        if time_left() < 5.0:
            break
        flat_init = init.flatten()
        
        for beta in [30.0, 60.0]:
            if time_left() < 3.0:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_smooth(x, b),
                    flat_init,
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _initial_placement_structured()
        best_score = _min_area(best_pts)
    
    # Phase 2: Multi-start perturbation + smooth optimization
    n_perturb = 15
    for trial in range(n_perturb):
        if time_left() < 4.0:
            break
        scale = 0.02 + 0.012 * trial
        trial_pts = best_pts + rng.normal(0, scale, size=(N, 2))
        trial_pts = _clamp_all(trial_pts)
        flat_trial = trial_pts.flatten()
        
        for beta in [40.0, 80.0]:
            if time_left() < 2.5:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_smooth(x, b),
                    flat_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 15000}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Adaptive simulated annealing for exploration
    if time_left() > 15.0:
        sa_time = min(10.0, time_left() - 5.0)
        sa_pts, sa_score = _simulated_annealing(best_pts, sa_time, rng)
        sa_pts = _clamp_all(sa_pts)
        sa_score = _min_area(sa_pts)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
    
    # Phase 4: Coordinate descent polish
    if time_left() > 4.0:
        cd_pts, cd_score = _coordinate_descent_polish(best_pts, time_budget=min(3.0, time_left() - 1.0))
        if cd_score > best_score:
            best_score = cd_score
            best_pts = cd_pts.copy()
    
    # Phase 5: Nelder-Mead local search on hard objective
    if time_left() > 6.0:
        try:
            res_nm = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
        except Exception:
            pass
    
    # Phase 6: Final L-BFGS-B polish on hard objective
    if time_left() > 3.0:
        try:
            res_final = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxfun': 50000}
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Phase 7: One more coordinate descent pass
    if time_left() > 2.0:
        cd_pts2, cd_score2 = _coordinate_descent_polish(best_pts, time_budget=min(1.5, time_left() - 0.5))
        if cd_score2 > best_score:
            best_score = cd_score2
            best_pts = cd_pts2.copy()
    
    # Final clamp
    best_pts = _clamp_all(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
