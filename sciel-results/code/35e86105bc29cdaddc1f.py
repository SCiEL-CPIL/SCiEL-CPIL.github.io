import numpy as np
import time
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize

N = 26

def _voronoi_inradii(points, box=(0.0, 1.0, 0.0, 1.0)):
    """
    For each point, compute the inradius of its Voronoi cell clipped to the box.
    Returns array of radii (one per point).
    """
    xmin, xmax, ymin, ymax = box
    n = len(points)
    radii = np.zeros(n)
    
    # Add boundary points to make Voronoi cells bounded
    # We'll use a trick: add points just outside the box to clip cells
    # Actually, scipy's Voronoi doesn't clip to a box easily.
    # Alternative: compute distance from each point to the nearest edge of its Voronoi cell.
    # For Voronoi cells, the inradius is the minimum distance from the seed to any edge of the cell.
    
    # A more robust approach: for each point, the inradius is the minimum over:
    # 1. Distance to box boundaries
    # 2. Half the distance to each neighbor in the Delaunay triangulation
    # This is exact for Voronoi cells: the cell edges are perpendicular bisectors of Delaunay edges.
    
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback: just use boundary distances
        d_left = points[:, 0] - xmin
        d_right = xmax - points[:, 0]
        d_bottom = points[:, 1] - ymin
        d_top = ymax - points[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        return radii
    
    for i in range(n):
        # Distance to box boundaries
        d_box = min(points[i, 0] - xmin, xmax - points[i, 0],
                    points[i, 1] - ymin, ymax - points[i, 1])
        r_i = d_box
        
        # For each Delaunay neighbor, the Voronoi edge is at half the distance
        # Find all simplices containing point i
        # tri.simplices: shape (m, 3)
        # For each simplex, find neighbors
        for simplex in tri.simplices:
            if i not in simplex:
                continue
            for j in simplex:
                if j == i:
                    continue
                d = np.linalg.norm(points[i] - points[j])
                r_i = min(r_i, d / 2.0)
        
        radii[i] = r_i
    
    return radii


def _voronoi_inradii_fast(points, box=(0.0, 1.0, 0.0, 1.0)):
    """
    Vectorized version: for each point, inradius = min(boundary_dist, min over Delaunay neighbors of dist/2)
    """
    xmin, xmax, ymin, ymax = box
    n = len(points)
    
    # Boundary distances
    d_left = points[:, 0] - xmin
    d_right = xmax - points[:, 0]
    d_bottom = points[:, 1] - ymin
    d_top = ymax - points[:, 1]
    d_box = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    try:
        tri = Delaunay(points)
    except Exception:
        return d_box
    
    # Build neighbor list from Delaunay triangulation
    # For each point, find all other points in simplices containing it
    # This is O(n * simplices_per_point)
    
    # More efficient: iterate over all simplices
    # For each simplex (a, b, c), the three pairs (a,b), (b,c), (c,a) are Delaunay edges
    # For each edge (i, j), the Voronoi constraint is r_i <= d(i,j)/2 and r_j <= d(i,j)/2
    
    # Initialize with boundary distances
    r = d_box.copy()
    
    simplices = tri.simplices  # shape (m, 3)
    for simplex in simplices:
        a, b, c = simplex
        # Edge a-b
        d_ab = np.linalg.norm(points[a] - points[b])
        half = d_ab / 2.0
        if half < r[a]:
            r[a] = half
        if half < r[b]:
            r[b] = half
        # Edge b-c
        d_bc = np.linalg.norm(points[b] - points[c])
        half = d_bc / 2.0
        if half < r[b]:
            r[b] = half
        if half < r[c]:
            r[c] = half
        # Edge c-a
        d_ca = np.linalg.norm(points[c] - points[a])
        half = d_ca / 2.0
        if half < r[c]:
            r[c] = half
        if half < r[a]:
            r[a] = half
    
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    best_radii = None
    
    num_starts = 15
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 5.0)
    
    for k in range(num_starts):
        if time.time() > deadline - 3.0:
            break
        
        start_k = time.time()
        
        # Initialize points
        if k == 0:
            # Hexagonal-like grid
            side = int(np.ceil(np.sqrt(N)))
            pts = np.zeros((N, 2))
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        pts[count, 0] = (i + 0.5) / side
                        pts[count, 1] = (j + 0.5) / side
                        count += 1
        elif k % 2 == 1:
            # Random uniform
            pts = np.random.rand(N, 2) * 0.8 + 0.1
        else:
            # Random with some structure
            pts = np.random.rand(N, 2) * 0.7 + 0.15
            pts[:, 0] = np.clip(pts[:, 0], 0.05, 0.95)
            pts[:, 1] = np.clip(pts[:, 1], 0.05, 0.95)
        
        # Ensure points are well-separated initially to avoid degenerate Delaunay
        # Add small jitter
        pts += np.random.normal(0, 0.005, (N, 2))
        pts = np.clip(pts, 0.01, 0.99)
        
        # Compute initial inradii
        radii = _voronoi_inradii_fast(pts)
        
        # Local optimization: maximize sum of inradii by moving points
        # The inradius function is piecewise smooth (min of linear/affine functions)
        # We use a subgradient-like approach or smooth approximation
        
        # Define objective: negative sum of inradii
        # We'll use a smooth approximation: replace min with a soft-min
        def smooth_min(vals, temperature=0.01):
            """Soft minimum approximation"""
            return -temperature * np.log(np.sum(np.exp(-vals / temperature)))
        
        def objective(z):
            p = z.reshape(N, 2)
            p = np.clip(p, 0.01, 0.99)
            # Compute inradii using Delaunay
            try:
                tri = Delaunay(p)
            except Exception:
                # Degenerate: return large penalty
                return 100.0
            
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
            d_left = p[:, 0] - xmin
            d_right = xmax - p[:, 0]
            d_bottom = p[:, 1] - ymin
            d_top = ymax - p[:, 1]
            d_box = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
            
            # For each point, find minimum over Delaunay neighbors
            r = d_box.copy()
            for simplex in tri.simplices:
                a, b, c = simplex
                d_ab = np.sqrt((p[a,0]-p[b,0])**2 + (p[a,1]-p[b,1])**2)
                half = d_ab / 2.0
                if half < r[a]:
                    r[a] = half
                if half < r[b]:
                    r[b] = half
                d_bc = np.sqrt((p[b,0]-p[c,0])**2 + (p[b,1]-p[c,1])**2)
                half = d_bc / 2.0
                if half < r[b]:
                    r[b] = half
                if half < r[c]:
                    r[c] = half
                d_ca = np.sqrt((p[c,0]-p[a,0])**2 + (p[c,1]-p[a,1])**2)
                half = d_ca / 2.0
                if half < r[c]:
                    r[c] = half
                if half < r[a]:
                    r[a] = half
            
            # Use smooth minimum for differentiability
            r_smooth = smooth_min(r, temperature=0.005)
            return -np.sum(r_smooth)
        
        # Optimize
        z0 = pts.flatten()
        bounds = [(0.01, 0.99)] * (2 * N)
        
        try:
            res = minimize(
                objective,
                z0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success or res.fun < objective(z0):
                pts_opt = res.x.reshape(N, 2)
                pts_opt = np.clip(pts_opt, 0.01, 0.99)
                radii_opt = _voronoi_inradii_fast(pts_opt)
                score_opt = np.sum(radii_opt)
            else:
                pts_opt = pts
                radii_opt = radii
                score_opt = np.sum(radii)
        except Exception:
            pts_opt = pts
            radii_opt = radii
            score_opt = np.sum(radii)
        
        # Second round of optimization with smaller temperature
        def objective2(z):
            p = z.reshape(N, 2)
            p = np.clip(p, 0.01, 0.99)
            try:
                tri = Delaunay(p)
            except Exception:
                return 100.0
            
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
            d_left = p[:, 0] - xmin
            d_right = xmax - p[:, 0]
            d_bottom = p[:, 1] - ymin
            d_top = ymax - p[:, 1]
            d_box = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
            
            r = d_box.copy()
            for simplex in tri.simplices:
                a, b, c = simplex
                d_ab = np.sqrt((p[a,0]-p[b,0])**2 + (p[a,1]-p[b,1])**2)
                half = d_ab / 2.0
                if half < r[a]:
                    r[a] = half
                if half < r[b]:
                    r[b] = half
                d_bc = np.sqrt((p[b,0]-p[c,0])**2 + (p[b,1]-p[c,1])**2)
                half = d_bc / 2.0
                if half < r[b]:
                    r[b] = half
                if half < r[c]:
                    r[c] = half
                d_ca = np.sqrt((p[c,0]-p[a,0])**2 + (p[c,1]-p[a,1])**2)
                half = d_ca / 2.0
                if half < r[c]:
                    r[c] = half
                if half < r[a]:
                    r[a] = half
            
            r_smooth = smooth_min(r, temperature=0.001)
            return -np.sum(r_smooth)
        
        try:
            res2 = minimize(
                objective2,
                pts_opt.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'maxfun': 100, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if res2.success or res2.fun < objective2(pts_opt.flatten()):
                pts_opt2 = res2.x.reshape(N, 2)
                pts_opt2 = np.clip(pts_opt2, 0.01, 0.99)
                radii_opt2 = _voronoi_inradii_fast(pts_opt2)
                score_opt2 = np.sum(radii_opt2)
                if score_opt2 > score_opt:
                    pts_opt = pts_opt2
                    radii_opt = radii_opt2
                    score_opt = score_opt2
        except Exception:
            pass
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
            best_radii = radii_opt.copy()
    
    # Fallback
    if best_points is None:
        best_points = np.random.rand(N, 2) * 0.8 + 0.1
        best_radii = _voronoi_inradii_fast(best_points)
    
    # Final safety: verify and project
    # Ensure all circles are inside the box and non-overlapping
    # The Voronoi inradius guarantees this by construction, but let's be safe
    for i in range(N):
        # Boundary check
        max_r = min(best_points[i, 0], 1.0 - best_points[i, 0],
                    best_points[i, 1], 1.0 - best_points[i, 1])
        best_radii[i] = min(best_radii[i], max_r)
    
    # Pairwise check: if any overlap, shrink radii
    for _ in range(20):
        max_violation = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                overlap = best_radii[i] + best_radii[j] - d
                if overlap > max_violation:
                    max_violation = overlap
        if max_violation < 1e-9:
            break
        # Shrink all radii slightly
        best_radii *= 0.98
        # Re-clip to boundary
        for i in range(N):
            max_r = min(best_points[i, 0], 1.0 - best_points[i, 0],
                        best_points[i, 1], 1.0 - best_points[i, 1])
            best_radii[i] = min(best_radii[i], max_r)
    
    final_state = np.column_stack([best_points[:, 0], best_points[:, 1], best_radii])
    return final_state
