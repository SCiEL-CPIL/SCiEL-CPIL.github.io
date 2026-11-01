from scipy.optimize import differential_evolution
import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
        start_time = time.time()
        deadline = start_time + 140.0  # Safety margin before 150s limit

        def hexagon_vertices(cx, cy, side, angle_rad):
            """Vertices of a regular hexagon (flat-top when angle=0)."""
            angles = angle_rad + np.radians(np.arange(0, 360, 60))
            return np.column_stack([
                cx + side * np.cos(angles),
                cy + side * np.sin(angles)
            ])

        def polygon_disjoint_sat(v1, v2):
            """Check if two convex polygons are disjoint using SAT."""
            axes = []
            for v in [v1, v2]:
                for i in range(len(v)):
                    j = (i + 1) % len(v)
                    edge = v[j] - v[i]
                    normal = np.array([-edge[1], edge[0]])
                    if np.linalg.norm(normal) > 1e-12:
                        normal /= np.linalg.norm(normal)
                        axes.append(normal)
            
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                    return True
            
            return False

        def hexagons_disjoint(positions, angles):
            """Check if all hexagons are pairwise disjoint."""
            n = len(positions)
            for i in range(n):
                vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                for j in range(i + 1, n):
                    vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                    if not polygon_disjoint_sat(vi, vj):
                        return False
            return True

        def calc_min_outer_side(all_verts, theta):
            """Calculate minimum outer hexagon side length for a given orientation theta."""
            angles = theta + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            apothem = max_width / 2.0
            side = 2 * apothem / np.sqrt(3)
            return side

        def find_best_outer_orientation(all_verts):
            """Find orientation and side length for minimum enclosing hexagon."""
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(lambda th: calc_min_outer_side(all_verts, th), 
                                  bounds=(0, np.pi/3), method='bounded', 
                                  options={'xatol': 1e-4})
            return res.fun, res.x

        def full_objective(params):
            """Objective: outer hexagon side length, with penalty for overlap."""
            if time.time() > deadline:
                return 1e10
                
            n = len(params) // 3
            positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
            angles = np.array([params[i*3+2] for i in range(n)])
            
            if not hexagons_disjoint(positions, angles):
                return 1e10
        
            all_verts = []
            for i in range(n):
                v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            
            side, _ = find_best_outer_orientation(all_verts)
            return side

        # Initial honeycomb configuration
        # Use d > 2 to ensure disjointness for any orientation (max diameter is 2)
        d = 2.1
        positions = []
        angles = []
        
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        for k in range(6):
            ang = np.radians(60 * k)
            positions.append([d * np.cos(ang), d * np.sin(ang)])
            angles.append(0.0)
        
        for k in range(5):
            ang = np.radians(60 * k)
            positions.append([2 * d * np.cos(ang), 2 * d * np.sin(ang)])
            angles.append(0.0)
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        if not hexagons_disjoint(positions, angles):
             positions = positions * 1.2
             if not hexagons_disjoint(positions, angles):
                 positions = positions * 1.5

        best_obj = full_objective(np.concatenate([positions.flatten(), angles]))
        best_params = np.concatenate([positions.flatten(), angles]).copy()
        
        bounds = []
        for i in range(12):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
            bounds.append((0.0, np.pi))
        
        # Reduce popsize and maxiter to ensure we finish within time limit
        # and have a valid fallback
        try:
            result = differential_evolution(
                full_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                polish=False,
                callback=lambda x, convergence: None
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
    
        except Exception as e:
            pass
    
        # Extract best solution
        if best_params is not None and len(best_params) == 36:
            best_positions = np.array([[best_params[i*3], best_params[i*3+1]] for i in range(12)])
            best_angles = np.array([best_params[i*3+2] for i in range(12)])
            
            if hexagons_disjoint(best_positions, best_angles):
                all_verts = []
                for i in range(12):
                    v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
                    all_verts.append(v)
                all_verts = np.vstack(all_verts)
                
                outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
                # Center is the centroid of the projections? 
                # For the returned center, we need to calculate it properly.
                # The center that minimizes the max distance is the midpoint of the bounding box in each normal direction?
                # Actually, for a regular hexagon, the center is unique for a given orientation and size.
                # We can find it by averaging the midpoints of the projections.
                angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
                normals = np.column_stack([np.cos(angles), np.sin(angles)])
                dots = all_verts @ normals.T
                mins = dots.min(axis=0)
                maxs = dots.max(axis=0)
                mids = (mins + maxs) / 2.0
                # The center c satisfies dot(c, n_k) = mid_k for all k?
                # This is an overdetermined system. We can use least squares.
                # A c = b, where A is normals.T (6x2), b is mids (6x1)
                c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
                outer_center = tuple(c)
                outer_angle_deg = np.degrees(outer_angle_rad)
            else:
                outer_center = (0.0, 0.0)
                outer_side = 10.0
                outer_angle_deg = 0.0
                best_positions = positions
                best_angles = angles
        else:
            outer_center = (0.0, 0.0)
            outer_side = 10.0
            outer_angle_deg = 0.0

        inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
        
        return inner, outer_center, outer_side, outer_angle_deg
    # EVOLVE-BLOCK-END
