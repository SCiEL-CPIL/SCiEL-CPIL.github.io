import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    pts = fibonacci_spiral(N)
    
    best_pts = pts.copy()
    best_radius = np.inf

    # Helper to compute covering radius
    def compute_covering_radius(points):
        try:
            hull = ConvexHull(points)
            # Facets are indices of points
            facets = hull.simplices
            # Normals are not directly provided in a way that maps to vertices easily for Voronoi vertices?
            # Actually, we can compute the normal for each simplex.
            # The normal of a triangle (a,b,c) on the unit sphere is proportional to a x b + b x c + c x a? 
            # Or simply normalize(a+b+c) is the circumcenter if the triangle is acute? 
            # The true circumcenter of a spherical triangle is the normalized vector sum of the vertices? 
            # No, the circumcenter is the intersection of the perpendicular bisectors. 
            # For a triangle on the sphere, the circumcenter is the normalized vector $n = a \times b + b \times c + c \times a$? 
            # Let's use the fact that the normal to the plane containing a,b,c is $n = \text{normalize}((b-a) \times (c-a))$.
            # This normal is the circumcenter of the spherical triangle ONLY if the triangle is "small" or we take the correct hemisphere.
            # Since points are on the unit sphere, the normal to the face of the convex hull is indeed the vertex of the spherical Voronoi diagram.
            
            normals = np.zeros((len(facets), 3))
            for i, simplex in enumerate(facets):
                a = points[simplex[0]]
                b = points[simplex[1]]
                c = points[simplex[2]]
                # Cross product of two edges
                v1 = b - a
                v2 = c - a
                n = np.cross(v1, v2)
                norm = np.linalg.norm(n)
                if norm > 1e-10:
                    n = n / norm
                else:
                    n = a # fallback
                normals[i] = n
            
            # For each point, find the max distance to any normal of a facet it belongs to.
            # We need to map point index to list of facet indices.
            point_facets = [[] for _ in range(N)]
            for i, simplex in enumerate(facets):
                for p_idx in simplex:
                    point_facets[p_idx].append(i)
            
            max_radius = 0.0
            worst_normals = [None] * N
            for i in range(N):
                if not point_facets[i]:
                    # Point is inside hull, not a vertex. This is bad for covering.
                    # Assign a large penalty or random normal?
                    # For now, let's say radius is pi/2 (worst case) or just skip movement?
                    # Actually, if a point is inside, it's redundant. We should move it out.
                    # Let's pick a random direction away from center?
                    # Better: assign the normal of the nearest facet?
                    # For simplicity in this loop, if no facets, we can't compute local max.
                    # Let's assume this is rare.
                    continue
                
                # Get distances to all normals of incident facets
                facet_indices = point_facets[i]
                incident_normals = normals[facet_indices]
                # Dot products
                dots = np.dot(incident_normals, points[i])
                # Clamp dots to [-1, 1]
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                
                max_angle = np.max(angles)
                if max_angle > max_radius:
                    max_radius = max_angle
                    worst_idx = np.argmax(angles)
                    worst_normals[i] = incident_normals[worst_idx]
                else:
                    worst_idx = np.argmax(angles)
                    worst_normals[i] = incident_normals[worst_idx]
            
            return max_radius, worst_normals, point_facets, facets, normals

        except Exception as e:
            # If hull fails (e.g. coplanar points), return a bad score
            return np.pi, [None]*N, [[] for _ in range(N)], [], np.zeros((0,3))

    # Main Optimization Loop
    step_size = 0.05
    for it in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        radius, worst_normals, point_facets, facets, normals = compute_covering_radius(pts)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
            
        # Check convergence
        if radius < 18.2 * np.pi / 180.0: # Slightly better than known best? Unlikely to jump that fast, but let's aim for 18.3
             # Actually 18.3 deg is 0.319 rad.
             pass

        # Update points
        new_pts = pts.copy()
        for i in range(N):
            if worst_normals[i] is not None:
                target = worst_normals[i]
                # Move towards target
                # Geodesic step
                # new = normalize(p + alpha * (target - p))
                # We need to be careful not to overshoot.
                # A small step is safer.
                new_pts[i] = pts[i] + step_size * (target - pts[i])
                new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
            else:
                # If point is inside hull, move it randomly to get out
                new_pts[i] = pts[i] + 0.01 * np.random.randn(3)
                new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
                
        pts = new_pts
        
        # Optional: Add small noise to escape local minima
        # if it % 50 == 0:
        #     pts = pts + 0.001 * np.random.randn(N, 3)
        #     pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # Final check and return best found
    return best_pts
