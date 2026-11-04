"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

_DEADLINE = 140.0  # Safety margin before 150s kill


def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_a:
                    min_a = area
    return min_a


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback: use all points as hull
        pts = np.array(pts)
        angles = np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean())
        order = np.argsort(angles)
        hull_pts = pts[order]
        n = len(hull_pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hull_pts[i, 0] * hull_pts[j, 1] - hull_pts[j, 0] * hull_pts[i, 1]
        return abs(area) * 0.5


def _score(pts):
    """Score = min triangle area / convex hull area."""
    ch_area = _convex_hull_area(pts)
    if ch_area < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ch_area


def _soft_min_obj(x, beta):
    """Soft-min surrogate: log-sum-exp of small triangle areas."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log-sum-exp(-areas / beta) / beta * beta = soft min
    # Actually: soft_min = -beta * log(sum(exp(-a_i / beta)))
    # To make it smooth and differentiable, use:
    max_neg = np.max(-areas / beta)
    log_sum = max_neg + np.log(np.sum(np.exp(-areas / beta - max_neg)))
    soft_min_val = -beta * log_sum
    # Add penalty if points are too close or hull area is tiny
    ch_area = _convex_hull_area(pts)
    if ch_area < 1e-6:
        return 1e10
    return -soft_min_val  # We minimize, so negate to maximize soft min


def _hard_min_obj(x):
    """Hard min triangle area (negated for minimization)."""
    pts = x.reshape(-1, 2)
    return -_min_triangle_area(pts)


def _make_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two concentric rings (7+7)
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner_angles = outer_angles + np.pi / 7
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    configs.append(np.vstack([outer, inner]))
    
    # Grid-like: 4x3.5 arrangement
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.array([[x, y] for x in xs for y in ys])
    configs.append(grid[:N])
    
    # Random with fixed seed
    rng = np.random.RandomState(42)
    configs.append(rng.uniform(-1, 1, size=(N, 2)))
    
    # Hexagonal packing approximation
    hex_pts = []
    for ring in range(3):
        if ring == 0:
            hex_pts.append([0, 0])
        else:
            for i in range(6 * ring):
                angle = 2 * np.pi * i / (6 * ring)
                hex_pts.append([ring * np.cos(angle), ring * np.sin(angle)])
    hex_pts = np.array(hex_pts[:N])
    configs.append(hex_pts)
    
    # Jittered regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = 1 + 0.1 * np.sin(3 * angles)
    configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    return configs


def _polish_local(pts, time_budget=10.0):
    """Polish a local configuration using L-BFGS-B with soft-min then hard min."""
    start = time.time()
    x0 = pts.flatten()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Stage 1: Soft-min with beta=50
    if time.time() - start < time_budget * 0.4:
        try:
            res = minimize(
                _soft_min_obj, x0, args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            pts1 = res.x.reshape(-1, 2)
            s1 = _score(pts1)
            if s1 > best_score:
                best_score = s1
                best_pts = pts1.copy()
        except Exception:
            pass
    
    # Stage 2: Soft-min with beta=250 (sharper)
    if time.time() - start < time_budget * 0.7:
        try:
            x0 = best_pts.flatten()
            res = minimize(
                _soft_min_obj, x0, args=(250.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            pts2 = res.x.reshape(-1, 2)
            s2 = _score(pts2)
            if s2 > best_score:
                best_score = s2
                best_pts = pts2.copy()
        except Exception:
            pass
    
    # Stage 3: Hard min polish
    if time.time() - start < time_budget:
        try:
            x0 = best_pts.flatten()
            res = minimize(
                _hard_min_obj, x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            pts3 = res.x.reshape(-1, 2)
            s3 = _score(pts3)
            if s3 > best_score:
                best_score = s3
                best_pts = pts3.copy()
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Generate initial configs
    configs = _make_initial_configs()
    
    best_pts = None
    best_score = 0.0
    
    # Evaluate and polish each initial config
    for i, cfg in enumerate(configs):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Normalize to unit area for scoring consistency
        ch_area = _convex_hull_area(cfg)
        if ch_area > 1e-6:
            cfg = cfg / np.sqrt(ch_area)  # Scale to unit area
        
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
        
        # Polish if time allows
        if time.time() - start_time < _DEADLINE - 15:
            polished, s_polished = _polish_local(cfg, time_budget=8.0)
            if s_polished > best_score:
                best_score = s_polished
                best_pts = polished.copy()
    
    # If we have time, do a few random restarts
    if time.time() - start_time < _DEADLINE - 30:
        rng = np.random.RandomState(123)
        for trial in range(20):
            if time.time() - start_time > _DEADLINE - 10:
                break
            cfg = rng.uniform(-1, 1, size=(N, 2))
            ch_area = _convex_hull_area(cfg)
            if ch_area > 1e-6:
                cfg = cfg / np.sqrt(ch_area)
            s = _score(cfg)
            if s > best_score:
                best_score = s
                best_pts = cfg.copy()
            if time.time() - start_time < _DEADLINE - 20:
                polished, s_polished = _polish_local(cfg, time_budget=5.0)
                if s_polished > best_score:
                    best_score = s_polished
                    best_pts = polished.copy()
    
    # Final safety: return best found
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure valid output
    best_pts = np.array(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END
