"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
N = 11
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])


def _min_area(pts):
    """Minimum area of any triangle formed by three points."""
    pi = pts[_ci]
    pj = pts[_cj]
    pk = pts[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle."""
    h = H
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0.0
        if y > h * x:
            y = h * x
        if y > h * (1.0 - x):
            y = h * (1.0 - x)
        pts[i] = [x, y]
    return pts


def _generate_structured_initial(rng):
    """Generate a structured initial configuration with symmetry."""
    # Use a grid-like pattern with some randomness
    pts = np.zeros((N, 2))
    
    # Place points in a roughly triangular grid
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point (top)
    
    jitter = 0.02 * rng.normal(0, 1, size=(N, 2))
    
    # Bottom row
    pts[0] = [0.05, 0.02] + jitter[0]
    pts[1] = [0.35, 0.02] + jitter[1]
    pts[2] = [0.65, 0.02] + jitter[2]
    pts[3] = [0.95, 0.02] + jitter[3]
    
    # Second row
    pts[4] = [0.2, H * 0.2] + jitter[4]
    pts[5] = [0.5, H * 0.2] + jitter[5]
    pts[6] = [0.8, H * 0.2] + jitter[6]
    
    # Third row
    pts[7] = [0.35, H * 0.4] + jitter[7]
    pts[8] = [0.65, H * 0.4] + jitter[8]
    
    # Top
    pts[9] = [0.5, H * 0.6] + jitter[9]
    pts[10] = [0.5, H * 0.8] + jitter[10]
    
    return _project_to_triangle(pts)


def _max_min_distance_init(rng):
    """Initialize using max-min distance heuristic for better spread."""
    pts = np.zeros((N, 2))
    
    # Start with corners and center
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, H]
    pts[3] = [0.5, H/3]
    
    for i in range(4, N):
        # Find position that maximizes minimum distance to existing points
        best_pos = None
        best_min_dist = -1
        
        # Sample candidate positions
        candidates = []
        for _ in range(200):
            x = rng.uniform(0, 1)
            y_max = min(H * x, H * (1 - x))
            y = rng.uniform(0, max(0, y_max))
            candidates.append([x, y])
        
        for pos in candidates:
            dists = np.sqrt(np.sum((pts[:i] - np.array(pos))**2, axis=1))
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        
        if best_pos is not None:
            pts[i] = best_pos
    
    return _project_to_triangle(pts)


def _gradient_ascent(pts, t, lr=0.01, n_steps=500):
    """
    Projected gradient ascent on the epigraph formulation.
    Maximize t subject to area(i,j,k) >= t for all triples.
    """
    for step in range(n_steps):
        # Compute all areas
        pi = pts[_ci]
        pj = pts[_cj]
        pk = pts[_ck]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Find which triangles are at or below t (active constraints)
        active = areas <= t + 1e-10
        
        if not np.any(active):
            break
        
        # Compute gradient of t with respect to points
        # For each active triangle, gradient of area w.r.t. points
        grad = np.zeros_like(pts)
        
        for idx in np.where(active)[0]:
            i, j, k = _ci[idx], _cj[idx], _ck[idx]
            pi_pos = pts[i]
            pj_pos = pts[j]
            pk_pos = pts[k]
            
            v1 = pj_pos - pi_pos
            v2 = pk_pos - pi_pos
            cross_val = v1[0] * v2[1] - v1[1] * v2[0]
            
            # Gradient of |cross| w.r.t. each point
            sign = np.sign(cross_val) if cross_val != 0 else 1.0
            
            # d/dpi = -sign * (v2 cross direction)
            # d/dpj = sign * (v2 cross direction)  
            # d/dpk = -sign * (v1 cross direction)
            
            # More carefully: area = 0.5 * |cross(v1, v2)|
            # grad w.r.t. pi: -0.5 * sign * [v2[1], -v2[0]]
            # grad w.r.t. pj: 0.5 * sign * [v2[1], -v2[0]]
            # grad w.r.t. pk: -0.5 * sign * [v1[1], -v1[0]]
            
            grad[i] += -0.5 * sign * np.array([v2[1], -v2[0]])
            grad[j] += 0.5 * sign * np.array([v2[1], -v2[0]])
            grad[k] += -0.5 * sign * np.array([v1[1], -v1[0]])
        
        # Update points
        pts = pts + lr * grad
        pts = _project_to_triangle(pts)
        
        # Update t: t = min(areas) after update
        pi = pts[_ci]
        pj = pts[_cj]
        pk = pts[_ck]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        t = np.min(areas)
        
        # Adaptive learning rate
        if step % 50 == 0:
            lr = max(0.001, min(0.05, lr * 0.8))
    
    return pts, t


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    inits = []
    
    # Structured initial
    inits.append(_generate_structured_initial(rng))
    
    # Max-min distance init
    inits.append(_max_min_distance_init(rng))
    
    # Random inits
    for _ in range(3):
        pts = rng.uniform(0, 1, size=(N, 2))
        # Ensure y is within triangle
        for i in range(N):
            pts[i][1] *= min(H * pts[i][0], H * (1 - pts[i][0])) / max(H * pts[i][0], 1e-10)
        inits.append(_project_to_triangle(pts))
    
    best_points = None
    best_area = 0
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_area = _min_area(init_pts)
        if current_area > best_area:
            best_area = current_area
            best_points = init_pts.copy()
        
        # Try gradient ascent from this init
        t0 = current_area
        try:
            opt_pts, t_final = _gradient_ascent(init_pts.copy(), t0, lr=0.01, n_steps=300)
            opt_area = _min_area(opt_pts)
            
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        # Also try SLSQP from the best point found so far
        if time.time() > deadline:
            break
        
        try:
            def obj(x):
                p = x.reshape(N, 2)
                p = _project_to_triangle(p.copy())
                return -_min_area(p)
            
            result = minimize(
                obj,
                best_points.flatten(),
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if result.fun < -best_area:
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_to_triangle(opt_pts.copy())
                opt_area = _min_area(opt_pts)
                
                if opt_area > best_area:
                    best_area = opt_area
                    best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement with more iterations if time permits
    if time.time() < deadline - 20 and best_points is not None:
        try:
            t0 = best_area
            opt_pts, t_final = _gradient_ascent(best_points.copy(), t0, lr=0.005, n_steps=500)
            opt_area = _min_area(opt_pts)
            
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    return best_points if best_points is not None else _generate_structured_initial(rng)
