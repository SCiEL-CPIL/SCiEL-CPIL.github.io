import numpy as np
from scipy.optimize import minimize

N = 13

def construct():
    # Initial guess: regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.cos(theta0)
    y0 = np.sin(theta0)
    pts0 = np.column_stack((x0, y0))
    p0 = pts0.flatten()

    def score(params):
        pts = params.reshape(N, 2)
        
        # Center at origin
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        # Check for degenerate (all same point)
        if np.allclose(pts, pts[0]):
            return 1e10

        # Sort by angle around origin
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        idx = np.argsort(angles)
        pts_sorted = pts[idx]
        
        # Check convexity
        # Calculate cross products of consecutive edge vectors
        # Edge i is pts_sorted[i+1] - pts_sorted[i]
        # We need pts_sorted[N] to be pts_sorted[0]
        pts_closed = np.vstack([pts_sorted, pts_sorted[:1]])
        edges = pts_closed[1:] - pts_closed[:-1]
        
        # Cross product z-component: x1*y2 - y1*x2
        cross_z = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # For a convex polygon oriented counter-clockwise, all cross products should be positive.
        # If any is <= 0, it's not strictly convex (or clockwise).
        # We assume CCW. If sum is negative, flip? No, just check signs.
        if np.any(cross_z <= 1e-10):
            # Not convex. Penalize.
            # We can return a large value, or try to calculate hull area roughly.
            # Returning large penalty guides optimizer to convex region.
            return 1e10 + np.sum(pts**2) # Add quadratic term to avoid zero
            
        # Calculate Area via Shoelace
        x = pts_sorted[:, 0]
        y = pts_sorted[:, 1]
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        
        if area < 1e-10:
            return 1e10

        # Normalize to unit area
        scale = 1.0 / np.sqrt(area)
        pts_scaled = pts_sorted * scale
        
        # Calculate min triangle area
        min_area = np.inf
        for i in range(N):
            pi = pts_scaled[i]
            for j in range(i + 1, N):
                pj = pts_scaled[j]
                for k in range(j + 1, N):
                    pk = pts_scaled[k]
                    v1 = pj - pi
                    v2 = pk - pi
                    area_tri = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area_tri < min_area:
                        min_area = area_tri
        
        return -min_area

    # Optimize
    # Nelder-Mead is good for this
    res = minimize(score, p0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Extract best points
    pts_opt = res.x.reshape(N, 2)
    
    # Final normalization
    centroid = np.mean(pts_opt, axis=0)
    pts_opt = pts_opt - centroid
    
    # Ensure convex and calculate area for final scaling
    angles = np.arctan2(pts_opt[:, 1], pts_opt[:, 0])
    idx = np.argsort(angles)
    pts_sorted = pts_opt[idx]
    pts_closed = np.vstack([pts_sorted, pts_sorted[:1]])
    edges = pts_closed[1:] - pts_closed[:-1]
    cross_z = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    if np.any(cross_z <= 1e-10):
        # Fallback to ConvexHull if it fell out of convexity
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts_opt)
            area = hull.volume
            pts_opt = pts_opt / np.sqrt(area)
        except:
            pass
    else:
        x = pts_sorted[:, 0]
        y = pts_sorted[:, 1]
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if area > 1e-10:
            pts_opt = pts_opt / np.sqrt(area)
            
    return pts_opt
