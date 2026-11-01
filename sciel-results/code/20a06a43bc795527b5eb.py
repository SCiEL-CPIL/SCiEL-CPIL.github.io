import numpy as np
import scipy.spatial as spat
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 14

def _hull_area(points):
    """Calculate area of convex hull."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    n = len(points)
    i, j, k = np.array(list(itertools.combinations(range(n), 3))).T
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = N
    idx_arr = np.array(list(itertools.combinations(range(n), 3)))
    i_idx = idx_arr[:, 0]
    j_idx = idx_arr[:, 1]
    k_idx = idx_arr[:, 2]

    def fast_score(points):
        """Fast score computation for a given set of points."""
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas) / hull_area

    def two_heptagon_config(params):
        """
        Generate 14 points from two heptagons.
        params: [r, phi, t1, t2, t3, t4, t5, t6, t7]
        r: scale of inner heptagon (0 to 1)
        phi: angular offset of inner heptagon
        t1..t7: edge positions for inner heptagon vertices (0 to 1, position along edge)
        
        Actually, let's use a cleaner parameterization:
        Outer heptagon: fixed at unit circle, angles 0, 2pi/7, ..., 12pi/7
        Inner heptagon: scaled by r, rotated by phi
        But we also allow the inner points to move along the edges of the outer heptagon.
        
        Better approach: 
        - 7 outer points on a regular heptagon (unit circle)
        - 7 inner points, each constrained to lie on one of the 7 edges of the outer heptagon
        - Parameters: r (scale), phi (rotation), and 7 edge-position parameters
        
        Let's use: outer heptagon fixed, inner heptagon with scale r and rotation phi.
        This gives 2 parameters. We can add more by allowing individual radial adjustments.
        
        For now, let's use the simple two-heptagon model with r and phi.
        """
        r, phi = params[0], params[1]
        
        base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        inner = r * np.column_stack((np.cos(base_angles + phi), np.sin(base_angles + phi)))
        
        return np.vstack((outer, inner))

    def two_heptagon_score(params):
        """Score for two heptagon configuration (negative for minimization)."""
        points = two_heptagon_config(params)
        return -fast_score(points)

    # Initial best from regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    best_points = pts_reg.copy()
    best_score = fast_score(pts_reg)
    
    # Try two-heptagon configurations with differential evolution
    if time.time() < deadline - 60:
        try:
            bounds = [(0.3, 1.0), (0, 2*np.pi)]
            result_de = differential_evolution(
                two_heptagon_score, 
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True
            )
            
            if result_de.success or result_de.fun < -best_score:
                params_opt = result_de.x
                pts_opt = two_heptagon_config(params_opt)
                s_opt = fast_score(pts_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = pts_opt.copy()
        except Exception:
            pass

    # Also try a three-ring configuration: outer heptagon, middle heptagon, inner heptagon
    # But we only have 14 points, so maybe 7+7 is the way to go.
    # Let's also try 5+5+4 or other splits.
    
    # Try a grid-like configuration
    if time.time() < deadline - 40:
        # 4x4 grid minus 2 points? Or 3x5 minus 1?
        # Let's try a 4x4 grid with 2 points removed
        grid_x = np.linspace(-1, 1, 4)
        grid_y = np.linspace(-1, 1, 4)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_pts = np.column_stack((gx.ravel(), gy.ravel()))
        # Remove 2 points (corners to keep convexity)
        # Actually, let's just use the first 14 points
        grid_pts_14 = grid_pts[:14]
        s_grid = fast_score(grid_pts_14)
        if s_grid > best_score:
            best_score = s_grid
            best_points = grid_pts_14.copy()
    
    # Try random configurations with local optimization
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random points in unit disk
        r = np.sqrt(np.random.uniform(0, 1, N))
        theta = np.random.uniform(0, 2*np.pi, N)
        pts_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        s_rand = fast_score(pts_rand)
        if s_rand > best_score:
            best_score = s_rand
            best_points = pts_rand.copy()
        
        # Local optimization from this point
        if time.time() > deadline - 10:
            continue
        
        def neg_score_local(x):
            pts = x.reshape(N, 2)
            return -fast_score(pts)
        
        try:
            res = minimize(neg_score_local, pts_rand.flatten(), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < -best_score:
                pts_opt = res.x.reshape(N, 2)
                s_opt = fast_score(pts_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = pts_opt.copy()
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead from best point
    if time.time() < deadline - 5 and best_points is not None:
        def neg_score_final(x):
            pts = x.reshape(N, 2)
            return -fast_score(pts)
        
        try:
            res = minimize(neg_score_final, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < -best_score:
                pts_final = res.x.reshape(N, 2)
                s_final = fast_score(pts_final)
                if s_final > best_score:
                    best_score = s_final
                    best_points = pts_final.copy()
        except Exception:
            pass
    
    return best_points
