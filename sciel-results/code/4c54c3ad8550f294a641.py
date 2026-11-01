import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time
import itertools

N = 14

def calc_score(points):
    """Compute the exact score: min triangle area / convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    min_area = float('inf')
    n = len(points)
    # Vectorized triangle area calculation for speed
    # Indices for all unique triples
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    pi = points[i_flat]
    pj = points[j_flat]
    pk = points[k_flat]
    
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    if len(areas) == 0:
        return 0.0
        
    min_area = np.min(areas)
    return min_area / hull_area

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _two_heptagon_points(rho, phi):
    """
    Construct 14 points from two nested regular heptagons.
    Outer: radius 1.0, angle offset 0.
    Inner: radius rho, angle offset phi.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_pts = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    
    angles_inner = angles_outer + phi
    inner_pts = rho * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    
    return np.vstack([outer_pts, inner_pts])

def _score_two_heptagon(params):
    rho, phi = params
    if rho < 0.01 or rho > 0.99:
        return -1.0
    if phi < -np.pi or phi > np.pi:
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
    
    points = _two_heptagon_points(rho, phi)
    return calc_score(points)

def _local_refine(points, max_iters=1000):
    """
    Local Nelder-Mead refinement on the full 28D space.
    Uses a custom objective that returns negative score.
    """
    def neg_score(x):
        pts = x.reshape(-1, 2)
        # Center the points to avoid drift
        pts -= pts.mean(axis=0)
        return -calc_score(pts)
    
    x0 = points.flatten()
    try:
        res = opt.minimize(
            neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        best_pts = res.x.reshape(-1, 2)
        best_pts -= best_pts.mean(axis=0)
        return best_pts
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_points = None
    best_score = -1.0

    # Phase 1: Coarse Grid Search over (rho, phi)
    # rho in [0.3, 0.9], phi in [0, pi/7] (symmetry of heptagon)
    # We can exploit symmetry: rotating inner by 2*pi/7 is equivalent to rotating outer.
    # So phi in [0, 2*pi/7] covers all unique configurations relative to outer.
    
    rho_vals = np.linspace(0.3, 0.9, 40)
    phi_vals = np.linspace(0, 2 * np.pi / 7, 40)
    
    for rho in rho_vals:
        if time.time() > deadline - 30:
            break
        for phi in phi_vals:
            points = _two_heptagon_points(rho, phi)
            score = calc_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

    # Phase 2: Fine Grid Search around the best found
    if best_points is not None:
        # Recover best rho, phi approximately
        # We just saved the points, so we can't directly recover rho, phi easily without fitting.
        # Instead, let's do a local Nelder-Mead on the 2-parameter space first.
        
        # We need to estimate rho and phi from the saved points.
        # Outer points are the 7 farthest from center.
        center = best_points.mean(axis=0)
        rel_points = best_points - center
        dists = np.linalg.norm(rel_points, axis=1)
        sorted_idx = np.argsort(dists)[::-1]
        outer_pts = rel_points[sorted_idx[:7]]
        inner_pts = rel_points[sorted_idx[7:]]
        
        # Estimate rho
        rho_est = np.mean(np.linalg.norm(inner_pts, axis=1)) / np.mean(np.linalg.norm(outer_pts, axis=1))
        
        # Estimate phi
        # Project outer and inner onto a common basis? 
        # Simpler: Just use the saved points as the start for full-space optimization.
        # But the prompt suggests high-precision 1D/2D search. 
        # Let's refine rho and phi using Nelder-Mead on the 2D parameter space, 
        # initializing with the estimated values.
        
        # To get phi, we can align the first outer point to (1,0) and see the angle of the closest inner point.
        # This is tricky. Let's just run Nelder-Mead on the 2D space with a few random starts 
        # or the estimated rho and a few phi candidates.
        
        # Actually, a simpler and more robust approach:
        # Run Nelder-Mead on the 2D parameter space (rho, phi) starting from the estimated rho
        # and a few phi values, or just use the points from Phase 1 directly for full-space refinement.
        
        # Given the budget, let's do a quick 2D Nelder-Mead refinement on (rho, phi) 
        # using the estimated rho and a scan of phi near the best grid point.
        
        # Find the best grid point's rho and phi
        # We didn't store them, so let's re-run a very fine grid around the best score region?
        # No, let's just use the full-space Nelder-Mead on the best_points from Phase 1.
        # This is a structural change from the previous gradient ascent.
        
        if time.time() < deadline - 20:
            refined = _local_refine(best_points, max_iters=500)
            score = calc_score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()

    # Phase 3: If time permits, try a few other symmetric structures or random starts
    if time.time() < deadline - 10:
        # Try a regular 14-gon
        angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        score = calc_score(reg_pts)
        if score > best_score:
            best_score = score
            best_points = reg_pts.copy()
            
        # Try random starts with local refinement
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Random points in a disk
            r = np.sqrt(np.random.rand(14))
            theta = np.random.rand(14) * 2 * np.pi
            rand_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            
            refined = _local_refine(rand_pts, max_iters=200)
            score = calc_score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
