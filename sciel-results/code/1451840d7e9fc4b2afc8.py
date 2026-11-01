from scipy.optimize import differential_evolution, minimize_scalar
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

    def approx_best_outer_side(all_verts):
        """Fast approximation of the minimum outer hexagon side length.
        Checks a discrete set of orientations."""
        thetas = np.linspace(0, np.pi/3, 12)
        min_side = np.inf
        for th in thetas:
            side = calc_min_outer_side(all_verts, th)
            if side < min_side:
                min_side = side
        return min_side

    def find_best_outer_orientation(all_verts):
        """Find orientation and side length for minimum enclosing hexagon."""
        res = minimize_scalar(lambda th: calc_min_outer_side(all_verts, th), 
                              bounds=(0, np.pi/3), method='bounded', 
                              options={'xatol': 1e-6})
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
        
        # Use fast approximation for the optimization loop
        side = approx_best_outer_side(all_verts)
        return side

    # Initial honeycomb configuration
    # Use d > 2 to ensure disjointness for any orientation (max diameter is 2)
    d = 2.05
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
        # Second ring: 6 positions, we take 5.
        # To minimize spread, we might want to adjust angles or positions later.
        # For now, standard hexagonal lattice.
        positions.append([2 * d * np.cos(ang), 2 * d * np.sin(ang)])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    if not hexagons_disjoint(positions, angles):
         positions = positions * 1.1
         if not hexagons_disjoint(positions, angles):
             positions = positions * 1.2

    best_obj = full_objective(np.concatenate([positions.flatten(), angles]))
    best_params = np.concatenate([positions.flatten(), angles]).copy()
    
    bounds = []
    for i in range(12):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, np.pi))
    
    # Increase popsize and maxiter since evaluation is faster now
    try:
        result = differential_evolution(
            full_objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-5,
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
            
            # Use precise optimization for the final answer
            outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
            
            # Calculate center
            angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            mins = dots.min(axis=0)
            maxs = dots.max(axis=0)
            mids = (mins + maxs) / 2.0
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
