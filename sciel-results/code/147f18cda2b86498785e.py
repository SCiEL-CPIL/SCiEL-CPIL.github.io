import numpy as np
from scipy.spatial import ConvexHull

def construct():
    N = 50
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Run local optimization
    # We will use a steepest descent approach on the covering radius.
    # The gradient of the covering radius with respect to a point p_i is non-zero
    # only if p_i is part of a facet with the maximum circumradius.
    
    for it in range(2000):
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, jitter points slightly
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        facets = hull.simplices
        radii = np.zeros(len(facets))
        centers = np.zeros((len(facets), 3))
        
        for k, tri in enumerate(facets):
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                radii[k] = 0
                centers[k] = a
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                u = -n
            else:
                u = n
            centers[k] = u
            dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
            radii[k] = np.arccos(dot_val)
            
        current_max_rad = np.max(radii)
        
        if current_max_rad < best_radius:
            best_radius = current_max_rad
            best_points = points.copy()
            
        # Identify critical facets (those close to the max)
        # We want to reduce the radius of the worst facets.
        # For each point, calculate the "push" direction.
        # If a point is in a facet with radius R, and R is the max, 
        # moving the point towards the facet center reduces R.
        
        push = np.zeros((N, 3))
        weight = np.zeros(N)
        
        # Threshold for "critical" facets to avoid moving points in small facets unnecessarily
        # which might increase their radius.
        threshold = current_max_rad - 0.1 * np.pi # 0.1 rad ~ 5.7 degrees
        # Actually, let's just use the max facets to be safe.
        # Or use a soft threshold.
        
        for k, tri in enumerate(facets):
            if radii[k] > threshold:
                # Weight by how bad this facet is
                w = radii[k] 
                for idx in tri:
                    # Move towards the center of this facet
                    push[idx] += w * centers[k]
                    weight[idx] += w
        
        # Normalize push directions
        moved = False
        for i in range(N):
            if weight[i] > 1e-10:
                target = push[i] / weight[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target /= norm_t
                    # Adaptive step size: smaller if we are close to convergence
                    step = 0.05 if it < 500 else 0.01
                    # Ensure we don't flip
                    new_p = points[i] + step * (target - points[i])
                    new_p /= np.linalg.norm(new_p)
                    points[i] = new_p
                    moved = True
                    
        if not moved:
            break
            
        # Periodic random restarts to escape local minima
        if it % 500 == 499 and it > 0:
            # Small random perturbation
            points += np.random.normal(0, 0.05, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)

    # Final polish: Use the best points found and run a few more iterations with smaller steps
    points = best_points
    for it in range(500):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        facets = hull.simplices
        radii = np.zeros(len(facets))
        centers = np.zeros((len(facets), 3))
        for k, tri in enumerate(facets):
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                u = -n
            else:
                u = n
            centers[k] = u
            dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
            radii[k] = np.arccos(dot_val)
            
        current_max_rad = np.max(radii)
        
        push = np.zeros((N, 3))
        weight = np.zeros(N)
        threshold = current_max_rad - 0.05 * np.pi
        
        for k, tri in enumerate(facets):
            if radii[k] > threshold:
                w = radii[k]
                for idx in tri:
                    push[idx] += w * centers[k]
                    weight[idx] += w
                    
        for i in range(N):
            if weight[i] > 1e-10:
                target = push[i] / weight[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target /= norm_t
                    step = 0.01
                    new_p = points[i] + step * (target - points[i])
                    new_p /= np.linalg.norm(new_p)
                    points[i] = new_p

    return points
