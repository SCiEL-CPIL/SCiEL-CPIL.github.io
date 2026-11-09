"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2.0

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

def _cart_to_bary(pts):
    """Convert Cartesian (x,y) to barycentric (λ1, λ2, λ3)."""
    lam3 = pts[:, 1] / H
    lam2 = pts[:, 0] - 0.5 * lam3
    lam1 = 1.0 - lam2 - lam3
    return np.stack([lam1, lam2, lam3], axis=1)

def _is_inside(pts, margin=1e-9):
    """Check if all points are inside or on the triangle."""
    lam = _cart_to_bary(pts)
    return np.all(lam >= -margin) and np.all(lam <= 1.0 + margin)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    p1 = pts[_idx_i]
    p2 = pts[_idx_j]
    p3 = pts[_idx_k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective(params, use_soft=False, temperature=0.01):
    """Objective to minimize: negative of min triangle area."""
    pts = params.reshape(N, 2)
    if not _is_inside(pts, margin=1e-6):
        return 1e10
    if use_soft:
        p1 = pts[_idx_i]
        p2 = pts[_idx_j]
        p3 = pts[_idx_k]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        neg_areas = -areas / temperature
        max_neg = np.max(neg_areas)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        return temperature * log_sum  # This is -soft_min, so minimize this
    else:
        return -_min_triangle_area(pts)

def _initial_grid_config():
    """Create a grid-like initial configuration inside the triangle."""
    pts = []
    n_per_side = 4
    for i in range(n_per_side):
        for j in range(n_per_side - i):
            k = n_per_side - 1 - i - j
            lam1 = i / (n_per_side - 1)
            lam2 = j / (n_per_side - 1)
            lam3 = k / (n_per_side - 1)
            x = lam2 * 1.0 + lam3 * 0.5
            y = lam3 * H
            pts.append([x, y])
    pts.append([0.5, H / 3.0])
    return np.array(pts)

def _reflect_into_triangle(pts):
    """Ensure points are inside the triangle."""
    pts = pts.copy()
    for i in range(N):
        x, y = pts[i]
        if x < 0:
            x = abs(x) * 0.3
        if y < 0:
            y = abs(y) * 0.3
        if x + y / H > 1.0:
            scale = 0.99 / (x + y / H)
            x *= scale
            y *= scale
        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(H, y))
        if x + y / H > 1.0:
            scale = 0.99 / (x + y / H)
            x *= scale
            y *= scale
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = _initial_grid_config()
    best_score = _min_triangle_area(best_pts)
    
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    def _callback(x, convergence):
        return time.time() > deadline - 25
    
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = best_pts.flatten()
        else:
            rng = np.random.RandomState(restart * 123 + 45)
            x0 = best_pts.flatten() + rng.randn(2 * N) * 0.08
            x0 = _reflect_into_triangle(x0.reshape(N, 2)).flatten()
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(False,),
                x0=x0,
                seed=restart * 42 + 7,
                maxiter=150,
                popsize=12,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                init='latinhypercube',
                polish=False,
                update_interval=3,
                callback=_callback
            )
            
            candidate_pts = result.x.reshape(N, 2)
            if _is_inside(candidate_pts):
                score = _min_triangle_area(candidate_pts)
                if score > best_score:
                    best_score = score
                    best_pts = candidate_pts.copy()
        except Exception:
            continue
    
    # Polish with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_pts.flatten(),
                args=(False,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            candidate_pts = result.x.reshape(N, 2)
            if _is_inside(candidate_pts):
                score = _min_triangle_area(candidate_pts)
                if score > best_score:
                    best_score = score
                    best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    best_pts = _reflect_into_triangle(best_pts)
    return best_pts
