import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
_tri_i = []
_tri_j = []
_tri_k = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _tri_i.append(i)
            _tri_j.append(j)
            _tri_k.append(k)
_tri_i = np.array(_tri_i)
_tri_j = np.array(_tri_j)
_tri_k = np.array(_tri_k)
N_TRI = len(_tri_i)

def _all_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_tri_i]
    pj = pts[_tri_j]
    pk = pts[_tri_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum area of any triangle formed by three points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coords."""
    x, y = p
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    eps = 1e-9
    a = max(eps, min(1 - 2 * eps, a))
    b = max(eps, min(1 - 2 * eps, b))
    c = max(eps, min(1 - 2 * eps, c))
    s = a + b + c
    a /= s
    b /= s
    c /= s
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _delaunay_smooth_obj(flat_pts, beta=80.0, lam=0.15):
    """
    Smooth objective that combines soft-min with variance regularization
    on the smallest few triangle areas (Delaunay-aware).
    """
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    areas = _all_areas(pts)
    m = np.min(areas)
    
    # Soft min
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = m - log_sum / beta
    
    # Variance of the k smallest areas (encourages balance among critical triangles)
    k = min(20, len(areas))
    sorted_areas = np.sort(areas)
    small_areas = sorted_areas[:k]
    variance_term = np.var(small_areas)
    
    # Objective: maximize soft_min, minimize variance of small areas
    return -(soft_min - lam * variance_term)

def _hard_min_obj(flat_pts):
    """Objective to minimize: negative min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _generate_hexagonal_init():
    """Hexagonal lattice initial placement with 3-fold symmetry."""
    pts = []
    cx, cy = 1/3, H/3
    pts.append([cx, cy])
    r1 = 0.15
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        px = cx + r1 * np.cos(angle)
        py = cy + r1 * np.sin(angle)
        pts.append([px, py])
    r2 = 0.28
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        px = cx + r2 * np.cos(angle)
        py = cy + r2 * np.sin(angle)
        pts.append([px, py])
    return _clamp_all(np.array(pts))

def _generate_random_init(seed):
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
    return _clamp_all(np.array(pts))

def _generate_perturbed_init(base_pts, scale, seed):
    """Generate perturbed version of base points."""
    rng = np.random.default_rng(seed)
    pts = base_pts + rng.normal(0, scale, size=(N, 2))
    return _clamp_all(pts)

def _local_polish_bary(pts, n_iters=3000):
    """
    Local polish using barycentric coordinate descent with adaptive steps.
    Optimizes each point one at a time, moving in the direction that
    increases the minimum area.
    """
    current = pts.copy()
    current_score = _min_area(current)
    
    # Adaptive step size
    step = 0.01
    no_improve_count = 0
    max_no_improve = 20
    
    for it in range(n_iters):
        improved = False
        for i in range(N):
            # Try small perturbations in barycentric space
            best_trial = None
            best_trial_score = current_score
            
            # Generate candidate moves
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            for angle in angles:
                dx = step * np.cos(angle)
                dy = step * np.sin(angle)
                trial = current.copy()
                trial[i] = _clamp_to_triangle(current[i] + np.array([dx, dy]))
                trial_score = _min_area(trial)
                if trial_score > best_trial_score + 1e-15:
                    best_trial = trial
                    best_trial_score = trial_score
            
            if best_trial is not None:
                current = best_trial
                current_score = best_trial_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            step *= 0.7
            no_improve_count = 0
            if step < 1e-6:
                break
        
        if no_improve_count > max_no_improve:
            step *= 0.8
            no_improve_count = 0
    
    return current, current_score

def _penalty_gradient_ascent(pts, n_iters=2000, lr_init=0.005, lr_decay=0.999):
    """
    Custom penalty-based gradient ascent.
    Explicitly penalizes any triangle area falling below the current minimum.
    Uses numerical gradients in barycentric coordinates.
    """
    current = pts.copy()
    current_score = _min_area(current)
    lr = lr_init
    
    for it in range(n_iters):
        # Compute penalty gradient: for each point, find direction that
        # most increases the minimum area, with extra weight on the
        # triangles that are at or near the minimum.
        
        areas = _all_areas(current)
        m = current_score
        # Weight triangles by how close they are to the minimum
        weights = np.exp(-100 * (areas - m))
        weights /= np.sum(weights) + 1e-15
        
        # For each point, compute the gradient of the weighted sum of areas
        grad = np.zeros((N, 2))
        for i in range(N):
            # Numerical gradient for point i
            eps = 1e-5
            for d in range(2):
                plus = current.copy()
                minus = current.copy()
                plus[i, d] += eps
                minus[i, d] -= eps
                plus = _clamp_all(plus)
                minus = _clamp_all(minus)
                
                areas_plus = _all_areas(plus)
                areas_minus = _all_areas(minus)
                
                # Weighted sum
                obj_plus = np.sum(weights * areas_plus)
                obj_minus = np.sum(weights * areas_minus)
                
                grad[i, d] = (obj_plus - obj_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        trial = current + lr * grad
        trial = _clamp_all(trial)
        trial_score = _min_area(trial)
        
        if trial_score > current_score - 1e-10:
            current = trial
            current_score = trial_score
        else:
            lr *= 0.9
        
        lr = max(lr * lr_decay, 1e-7)
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = 145.0

    def time_left():
        return deadline - (time.time() - start_time)

    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(_generate_hexagonal_init())
    for seed in [1, 7, 13, 29, 42, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        init_candidates.append(_generate_random_init(seed))

    best_pts = None
    best_score = -1.0

    # Phase 1: Multi-start optimization with Delaunay-aware smooth objective
    for init in init_candidates:
        if time_left() < 10.0:
            break
        for beta in [50.0, 80.0, 120.0]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _delaunay_smooth_obj(x, b, 0.15),
                    init.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_pts = opt_pts.copy()
                    best_score = opt_score
            except Exception:
                pass

    if best_pts is None:
        best_pts = _generate_hexagonal_init()
        best_score = _min_area(best_pts)

    # Phase 2: Local polish with barycentric coordinate descent
    if time_left() > 10.0:
        polished_pts, polished_score = _local_polish_bary(best_pts, n_iters=3000)
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score

    # Phase 3: Penalty-based gradient ascent
    if time_left() > 10.0:
        pg_pts, pg_score = _penalty_gradient_ascent(best_pts, n_iters=2000)
        if pg_score > best_score:
            best_pts = pg_pts
            best_score = pg_score

    # Phase 4: Perturbation-based multi-start
    rng = np.random.default_rng(12345)
    for trial in range(15):
        if time_left() < 8.0:
            break
        scale = 0.01 + 0.008 * trial
        trial_pts = _generate_perturbed_init(best_pts, scale, seed=1000 + trial)
        
        for beta in [60.0, 100.0]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _delaunay_smooth_obj(x, b, 0.15),
                    trial_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                opt_pts = _clamp_all(res.x.reshape(-1, 2))
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_pts = opt_pts.copy()
                    best_score = opt_score
            except Exception:
                pass

    # Phase 5: Final local polish
    if time_left() > 5.0:
        polished_pts, polished_score = _local_polish_bary(best_pts, n_iters=2000)
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score

    # Final safety check
    best_pts = _clamp_all(best_pts)

    return best_pts
