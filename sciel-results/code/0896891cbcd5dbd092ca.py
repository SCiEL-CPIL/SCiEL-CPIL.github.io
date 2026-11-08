"""Optimized solution for Heilbronn convex regions - 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area
        from scipy.spatial import ConvexHull as CH
        # Just compute area via shoelace on hull vertices
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return -1.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a

def _negative_score(pts):
    return -_score(pts)

def _make_regular_polygon(n, radius=1.0, center=(0,0), phase=0.0):
    """Create a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n + phase
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.column_stack((x, y))

def _make_multi_shell(n_outer, n_inner, r_outer=1.0, r_inner=0.5, phase_outer=0.0, phase_inner=0.0):
    """Create multi-shell configuration: outer regular polygon + inner regular polygon."""
    outer = _make_regular_polygon(n_outer, r_outer, (0,0), phase_outer)
    inner = _make_regular_polygon(n_inner, r_inner, (0,0), phase_inner)
    return np.vstack([outer, inner])

def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(_make_regular_polygon(13, 1.0, (0,0), 0.0))
    
    # 2. Multi-shell: 7 outer + 6 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for phase in [0.0, np.pi/7, np.pi/6, np.pi/12]:
            initial_configs.append(_make_multi_shell(7, 6, 1.0, r_inner, 0.0, phase))
    
    # 3. Multi-shell: 6 outer + 7 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for phase in [0.0, np.pi/6, np.pi/7, np.pi/12]:
            initial_configs.append(_make_multi_shell(6, 7, 1.0, r_inner, 0.0, phase))
    
    # 4. Multi-shell: 8 outer + 5 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        for phase in [0.0, np.pi/8, np.pi/5]:
            initial_configs.append(_make_multi_shell(8, 5, 1.0, r_inner, 0.0, phase))
    
    # 5. Multi-shell: 5 outer + 8 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        for phase in [0.0, np.pi/5, np.pi/8]:
            initial_configs.append(_make_multi_shell(5, 8, 1.0, r_inner, 0.0, phase))
    
    # 6. Multi-shell: 9 outer + 4 inner
    for r_inner in [0.3, 0.4, 0.5]:
        initial_configs.append(_make_multi_shell(9, 4, 1.0, r_inner, 0.0, 0.0))
    
    # 7. Multi-shell: 4 outer + 9 inner
    for r_inner in [0.3, 0.4, 0.5]:
        initial_configs.append(_make_multi_shell(4, 9, 1.0, r_inner, 0.0, 0.0))
    
    # 8. Regular 13-gon with different phases (should be same by symmetry but just in case)
    for phase in [np.pi/13, 2*np.pi/13]:
        initial_configs.append(_make_regular_polygon(13, 1.0, (0,0), phase))
    
    # Evaluate all initial configs
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
    
    # Now try to optimize from the best initial config
    if time.time() < deadline - 5.0 and best_pts is not None:
        # Optimize the best config
        x0 = best_pts.flatten()
        
        # Use Nelder-Mead for robustness (no gradients needed)
        try:
            result = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() < deadline:
                opt_pts = result.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Try optimizing from a few other good initial configs
    if time.time() < deadline - 10.0:
        # Try top few initial configs
        scored_configs = []
        for cfg in initial_configs:
            s = _score(cfg)
            scored_configs.append((s, cfg))
        scored_configs.sort(key=lambda x: -x[0])
        
        # Try optimizing top 3 distinct configs
        tried = 0
        for s, cfg in scored_configs:
            if time.time() > deadline - 5.0:
                break
            if tried >= 3:
                break
            # Skip if too similar to already tried
            if np.allclose(cfg, best_pts, atol=1e-6):
                continue
            tried += 1
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _negative_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if time.time() < deadline:
                    opt_pts = result.x.reshape(N, 2)
                    s_opt = _score(opt_pts)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Final: ensure we return something valid
    if best_pts is None:
        best_pts = _make_regular_polygon(13, 1.0, (0,0), 0.0)
    
    return best_pts
