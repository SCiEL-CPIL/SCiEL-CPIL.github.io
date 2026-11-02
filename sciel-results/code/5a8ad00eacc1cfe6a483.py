"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 13
DEADLINE = 140.0  # safety margin before 150s kill


def _min_tri_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    # Vectorized: for each pair (i,j), cross product with all k
    # Use a loop over pairs for memory efficiency with 13 points
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            # cross product of v with (pts[k] - pts[i]) for all k
            rel = pts - pts[i]
            cross = rel[:, 0] * v[1] - rel[:, 1] * v[0]
            areas = np.abs(cross) * 0.5
            # exclude i and j themselves (they give 0)
            areas[i] = np.inf
            areas[j] = np.inf
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _hull_area(pts):
    """Compute convex hull area using scipy or manual."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: shoelace on sorted points (approximate)
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


def _make_symmetric(params):
    """
    Build 13 points from symmetric parameters.
    
    Design: Regular hexagon (6 vertices) + regular heptagon (7 vertices) inside,
    with radial scaling and angular offset as free parameters.
    
    params = [r_hex, r_hep, angle_hex, angle_hep, center_x, center_y]
    """
    r_hex = params[0]
    r_hep = params[1]
    angle_hex = params[2]
    angle_hep = params[3]
    cx = params[4]
    cy = params[5]
    
    # 6 points: regular hexagon
    angles_hex = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_hex
    pts_hex = np.column_stack([
        cx + r_hex * np.cos(angles_hex),
        cy + r_hex * np.sin(angles_hex)
    ])
    
    # 7 points: regular heptagon
    angles_hep = np.linspace(0, 2 * np.pi, 7, endpoint=False) + angle_hep
    pts_hep = np.column_stack([
        cx + r_hep * np.cos(angles_hep),
        cy + r_hep * np.sin(angles_hep)
    ])
    
    return np.vstack([pts_hex, pts_hep])


def _score_symmetric(params):
    """Negative score for minimization."""
    pts = _make_symmetric(params)
    return -_score(pts)


def _make_hex7(params):
    """
    Alternative: 7 outer (heptagon) + 6 inner (hexagon), both centered.
    params = [r_outer, r_inner, angle_outer, angle_inner, cx, cy]
    """
    r_out = params[0]
    r_in = params[1]
    ang_out = params[2]
    ang_in = params[3]
    cx = params[4]
    cy = params[5]
    
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + ang_out
    pts_out = np.column_stack([
        cx + r_out * np.cos(angles_out),
        cy + r_out * np.sin(angles_out)
    ])
    
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + ang_in
    pts_in = np.column_stack([
        cx + r_in * np.cos(angles_in),
        cy + r_in * np.sin(angles_in)
    ])
    
    return np.vstack([pts_out, pts_in])


def _score_hex7(params):
    pts = _make_hex7(params)
    return -_score(pts)


def construct():
    t_start = time.time()
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Symmetric hexagon + heptagon (6+7)
    # Bounds: radii in [0.1, 1.5], angles in [0, 2pi], center in [-0.5, 0.5]
    bounds1 = [(0.05, 1.5), (0.05, 1.5), (0, 2 * np.pi), (0, 2 * np.pi),
               (-0.5, 0.5), (-0.5, 0.5)]
    
    # Strategy 2: 7 outer + 6 inner
    bounds2 = [(0.5, 1.5), (0.05, 1.0), (0, 2 * np.pi), (0, 2 * np.pi),
               (-0.5, 0.5), (-0.5, 0.5)]
    
    strategies = [
        (_score_symmetric, bounds1, _make_symmetric),
        (_score_hex7, bounds2, _make_hex7),
    ]
    
    for score_fn, bounds, make_fn in strategies:
        if time.time() - t_start > DEADLINE:
            break
        
        try:
            result = differential_evolution(
                score_fn, bounds,
                seed=42, maxiter=80, popsize=15,
                tol=1e-8, mutation=(0.3, 1.2), recombination=0.8,
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
    
    # Strategy 3: Full 26D optimization from best symmetric solution
    if best_pts is not None and time.time() - t_start < DEADLINE - 30:
        try:
            from scipy.optimize import minimize
            
            def neg_score_flat(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            x0 = best_pts.flatten()
            
            # Use Nelder-Mead for refinement
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
    
    # Strategy 4: Multi-start Nelder-Mead from various symmetric configs
    if time.time() - t_start < DEADLINE - 20:
        from scipy.optimize import minimize
        
        def neg_score_flat(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        # Generate diverse starting points
        np.random.seed(123)
        starts = []
        
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        starts.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
        
        # Various hex+hep configs
        for r_h in [0.3, 0.5, 0.7]:
            for r_he in [0.4, 0.6, 0.8]:
                for a_h in [0, np.pi/6]:
                    for a_he in [0, np.pi/7]:
                        p = _make_symmetric([r_h, r_he, a_h, a_he, 0, 0])
                        starts.append(p.flatten())
        
        # Random perturbations of best
        if best_pts is not None:
            for _ in range(5):
                perturbed = best_pts + np.random.randn(13, 2) * 0.1
                starts.append(perturbed.flatten())
        
        for x0 in starts:
            if time.time() - t_start > DEADLINE:
                break
            try:
                res = minimize(
                    neg_score_flat, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                pts = res.x.reshape(-1, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                continue
    
    # Final fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts


# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points:\n{pts}")
