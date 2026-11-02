import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 115.0  # Safety margin before 120s limit

def _min_tri_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized approach for small n
    # For each pair i, j, compute area with all k
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            rel = pts - pts[i]
            cross = rel[:, 0] * v[1] - rel[:, 1] * v[0]
            areas = np.abs(cross) * 0.5
            areas[i] = np.inf
            areas[j] = np.inf
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback
        order = np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), 
                                       pts[:, 0] - pts[:, 0].mean()))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_tri_area(pts)
    return min_tri / hull_a

def _make_1_6_6(params):
    """
    1 center, 6 on hexagon r1, 6 on hexagon r2.
    params: [r1, r2, phi_rel]
    """
    r1, r2, phi_rel = params
    
    # Center
    pts_center = np.array([[0.0, 0.0]])
    
    # Ring 1: Hexagon
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts_ring1 = np.column_stack([
        r1 * np.cos(angles1),
        r1 * np.sin(angles1)
    ])
    
    # Ring 2: Hexagon, rotated by phi_rel
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phi_rel
    pts_ring2 = np.column_stack([
        r2 * np.cos(angles2),
        r2 * np.sin(angles2)
    ])
    
    return np.vstack([pts_center, pts_ring1, pts_ring2])

def _make_1_12(params):
    """
    1 center, 12 on dodecagon r.
    params: [r]
    """
    r = params[0]
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_ring = np.column_stack([
        r * np.cos(angles),
        r * np.sin(angles)
    ])
    pts_center = np.array([[0.0, 0.0]])
    return np.vstack([pts_center, pts_ring])

def _make_1_4_8(params):
    """
    1 center, 4 on square r1, 8 on octagon r2.
    params: [r1, r2, phi_rel]
    """
    r1, r2, phi_rel = params
    
    pts_center = np.array([[0.0, 0.0]])
    
    angles1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts_ring1 = np.column_stack([
        r1 * np.cos(angles1),
        r1 * np.sin(angles1)
    ])
    
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + phi_rel
    pts_ring2 = np.column_stack([
        r2 * np.cos(angles2),
        r2 * np.sin(angles2)
    ])
    
    return np.vstack([pts_center, pts_ring1, pts_ring2])

def _make_1_5_7(params):
    """
    1 center, 5 on pentagon r1, 7 on heptagon r2.
    params: [r1, r2, phi_rel]
    """
    r1, r2, phi_rel = params
    
    pts_center = np.array([[0.0, 0.0]])
    
    angles1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts_ring1 = np.column_stack([
        r1 * np.cos(angles1),
        r1 * np.sin(angles1)
    ])
    
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phi_rel
    pts_ring2 = np.column_stack([
        r2 * np.cos(angles2),
        r2 * np.sin(angles2)
    ])
    
    return np.vstack([pts_center, pts_ring1, pts_ring2])

def construct():
    t_start = time.time()
    
    best_score = 0.0
    best_pts = None
    
    # Define strategies: (make_fn, bounds, initial_guess)
    strategies = [
        # 1+6+6 is the most promising for triangular lattice symmetry
        (_make_1_6_6, [(0.1, 1.5), (0.1, 1.5), (0, 2 * np.pi)], [0.5, 0.8, 0.0]),
        # 1+12
        (_make_1_12, [(0.1, 1.5)], [0.8]),
        # 1+4+8
        (_make_1_4_8, [(0.1, 1.5), (0.1, 1.5), (0, 2 * np.pi)], [0.5, 0.8, 0.0]),
        # 1+5+7
        (_make_1_5_7, [(0.1, 1.5), (0.1, 1.5), (0, 2 * np.pi)], [0.5, 0.8, 0.0]),
    ]
    
    for make_fn, bounds, init in strategies:
        if time.time() - t_start > DEADLINE:
            break
        
        try:
            # Use differential evolution for global search in low-dim space
            # popsize and maxiter tuned for speed
            result = differential_evolution(
                lambda x: -_score(make_fn(x)),
                bounds,
                seed=42,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                polish=True
            )
            
            if time.time() - t_start > DEADLINE:
                break
                
            pts = make_fn(result.x)
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    # If we found a good symmetric solution, try to refine it by breaking symmetry
    # using full 26D optimization. This can sometimes squeeze out a bit more.
    if best_pts is not None and time.time() - t_start < DEADLINE - 10:
        try:
            def neg_score_flat(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            x0 = best_pts.flatten()
            
            # Nelder-Mead is good for this small dimension
            res = minimize(
                neg_score_flat, x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() - t_start < DEADLINE:
                pts = res.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
            
    # Fallback if everything failed
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
