import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
            # 1. Initialization: Fibonacci Spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
            # Normalize
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            start_time = time.time()
            time_limit = 140.0 # Safety margin
            
            best_points = points.copy()
            best_radius = np.inf
            
            # Helper to calculate covering radius
            def get_covering_radius(pts):
                try:
                    hull = ConvexHull(pts)
                    max_rad = 0.0
                    for simplex in hull.simplices:
                        # simplex contains indices of points forming the facet
                        facet_pts = pts[simplex]
                        # Compute circumcenter of the spherical facet
                        # For a triangle on the unit sphere, the circumcenter is the normalized cross product? No.
                        # The circumcenter of a spherical triangle is the intersection of the great circles bisecting the arcs.
                        # Alternatively, the circumcenter of the planar triangle formed by the vertices, projected to the sphere?
                        # Actually, for the convex hull, the "cap" is defined by the plane passing through the facet vertices.
                        # The center of the cap is the normal to this plane.
                        # The angular radius is the angle between the normal and any vertex.
                        
                        # Compute normal to the facet
                        # Cross product of (v1-v0) and (v2-v0)
                        v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                        n = np.cross(v1 - v0, v2 - v0)
                        n_norm = np.linalg.norm(n)
                        if n_norm < 1e-10:
                            continue
                        n /= n_norm
                        # The center of the cap is n (or -n, check which one is "outward")
                        # For a convex hull of points on the unit sphere, the origin is inside.
                        # The outward normal points away from the origin.
                        # Since points are on the unit sphere, the normal n is a unit vector.
                        # The angular distance from n to v0 is arccos(n dot v0).
                        dist = np.arccos(np.clip(n @ v0, -1, 1))
                        # Note: if the facet is "large", the center might be the antipode?
                        # No, the covering radius is the max distance to the *nearest* point.
                        # The farthest point on the sphere from the set of points is the center of the largest empty circle.
                        # For a convex hull facet, the empty circle is centered at the normal vector n.
                        # The radius is the angle between n and the vertices.
                        if dist > max_rad:
                            max_rad = dist
                    return max_rad
                except Exception:
                    return np.pi # Invalid hull, return large value
            
            # Main Loop
            step_size = 0.01
            repulsion_strength = 0.001
            
            while time.time() - start_time < time_limit:
                # 1. Find worst facet
                try:
                    hull = ConvexHull(points)
                    worst_rad = 0.0
                    worst_simplex = None
                    worst_center = None
                    
                    for simplex in hull.simplices:
                        facet_pts = points[simplex]
                        v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                        n = np.cross(v1 - v0, v2 - v0)
                        n_norm = np.linalg.norm(n)
                        if n_norm < 1e-10:
                            continue
                        n /= n_norm
                        dist = np.arccos(np.clip(n @ v0, -1, 1))
                        if dist > worst_rad:
                            worst_rad = dist
                            worst_simplex = simplex
                            worst_center = n
                except Exception:
                    break
                
                if worst_simplex is None:
                    break
                
                # Update best if improved
                if worst_rad < best_radius:
                    best_radius = worst_rad
                    best_points = points.copy()
                
                # 2. Move vertices of worst facet towards the center
                # This reduces the radius of this specific cap.
                for idx in worst_simplex:
                    points[idx] = points[idx] + step_size * (worst_center - points[idx])
                
                # 3. Apply global repulsion to prevent clustering
                # This is O(N^2) which is fine for N=130.
                for i in range(N):
                    for j in range(i + 1, N):
                        diff = points[i] - points[j]
                        dist_sq = np.dot(diff, diff)
                        # If points are close, repel
                        if dist_sq < 1e-6:
                            dist_sq = 1e-6
                        dist = np.sqrt(dist_sq)
                        # Force is proportional to 1/dist^2 or similar
                        # We want to push them apart.
                        force = repulsion_strength / dist_sq
                        points[i] += force * (diff / dist)
                        points[j] -= force * (diff / dist)
                
                # 4. Normalize
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                # 5. Adaptive Step Size
                # Calculate new radius
                current_rad = get_covering_radius(points)
                if current_rad < worst_rad:
                    # Improved, maybe increase step slightly?
                    # step_size *= 1.01
                    pass
                else:
                    # Degraded, reduce step
                    step_size *= 0.9
                    # Revert to best? Or just keep current and hope it recovers?
                    # Reverting is safer but slower. Let's just reduce step.
                
                # If step size becomes too small, break
                if step_size < 1e-6:
                    break

            # Final check: return the best found
            # Ensure the returned points are valid and normalized
            best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
            
            # Final sanity check: are they distinct?
            # The repulsion should handle this.
            
            return best_points
