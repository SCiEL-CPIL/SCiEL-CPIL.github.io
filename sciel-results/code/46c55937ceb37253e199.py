"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull, distance as scipy_distance

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def sat_distance(v1, v2):
        """
        Fast SAT distance check.
        Returns negative if overlapping (penetration depth), positive if separated.
        """
        min_overlap = float('inf')
        
        # Check all 12 axes (6 from v1, 6 from v2)
        for poly in [v1, v2]:
            for i in range(6):
                p1 = poly[i]
                p2 = poly[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Separation check
                if max1 < min2 or max2 < min1:
                    return float('inf') # Separated
                
                # Overlap amount on this axis
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_overlap:
                    min_overlap = overlap
        
        return -min_overlap

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length."""
        max_required = 0.0
        theta = outer_angle_rad
        normals = np.column_stack([
            np.cos(theta + np.radians(np.arange(0, 360, 60))),
            np.sin(theta + np.radians(np.arange(0, 360, 60)))
        ])
        
        # Vectorized projection check
        for hex_verts in inner_hexes:
            rel = hex_verts - np.array([outer_cx, outer_cy])
            projections = np.dot(rel, normals.T) # shape (6, 6)
            # For each normal k, we need max projection <= side * sqrt(3)/2
            # side = max_proj * 2 / sqrt(3)
            max_projs = np.max(projections, axis=0)
            required_sides = max_projs * 2.0 / np.sqrt(3.0)
            current_max = np.max(required_sides)
            if current_max > max_required:
                max_required = current_max
        return max_required

    def objective(params):
        """
        Objective: minimize outer side length + penalty for overlaps.
        params: [x0, y0, a0, ..., x11, y11, a11]
        Outer hexagon is fixed at (0,0) with angle 0 for optimization simplicity,
        but we will optimize the outer angle separately at the end.
        """
        if time.time() > deadline:
            return 1e6
        
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        
        # Compute smooth overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = sat_distance(inner_hexes[i], inner_hexes[j])
                if d < 0:
                    # Overlapping: quadratic penalty
                    penalty += 100.0 * (-d) ** 2
                elif d < 0.05:
                    # Small gap: encourage separation
                    penalty += 10.0 * (0.05 - d) ** 2
        
        # Compute outer side (centered at origin, angle 0)
        side = compute_outer_side(inner_hexes, 0.0, 0.0, 0.0)
        
        return side + penalty

    # Initial Configuration: Honeycomb-like grid
    def init_config():
        hx = []
        hy = []
        ha = []
        sqrt3 = np.sqrt(3)
        # 3 rows of 4
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                hx.append(x)
                hy.append(y)
                ha.append(0.0)
        
        hx = np.array(hx)
        hy = np.array(hy)
        ha = np.array(ha)
        # Center
        hx -= np.mean(hx)
        hy -= np.mean(hy)
        return np.concatenate([hx, hy, ha])

    best_params = init_config()
    best_score = objective(best_params)

    # Simple Local Search / Gradient Descent
    # Use Nelder-Mead for robustness on small dimensions, or manual gradient steps
    # Given 36 dimensions, Nelder-Mead might be slow. Let's use a simple coordinate descent or random perturbation.
    
    # We will use scipy's minimize with L-BFGS-B if gradients were available, but they aren't easily derived for SAT.
    # So we use a heuristic local search.
    
    step_size = 0.1
    max_iters = 1000
    no_improve_count = 0
    patience = 100
    
    current_params = best_params.copy()
    
    for i in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Random perturbation
        rng = np.random.RandomState(i)
        perturbation = rng.normal(0, step_size, 36)
        new_params = current_params + perturbation
        
        new_score = objective(new_params)
        
        if new_score < best_score:
            best_score = new_score
            best_params = new_params
            current_params = new_params
            no_improve_count = 0
            # Reduce step size slowly
            step_size = max(0.01, step_size * 0.99)
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                # Reset to best and increase step size slightly to escape local minima
                current_params = best_params.copy()
                step_size = min(0.5, step_size * 1.5)
                no_improve_count = 0
                if step_size > 0.5:
                    break

    # Final Optimization of Outer Hexagon
    inner_x = best_params[0:12]
    inner_y = best_params[12:24]
    inner_a = best_params[24:36]
    
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    cx = np.mean(inner_x)
    cy = np.mean(inner_y)
    
    # Find best outer angle
    best_side = np.inf
    best_outer_angle = 0.0
    for angle_deg in np.arange(0, 60, 1.0):
        angle_rad = np.radians(angle_deg)
        side = compute_outer_side(inner_hexes, cx, cy, angle_rad)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_rad
    
    # Fine-tune outer angle
    for angle_deg in np.arange(best_outer_angle - 1.0, best_outer_angle + 1.0, 0.1):
        angle_rad = np.radians(angle_deg)
        side = compute_outer_side(inner_hexes, cx, cy, angle_rad)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_rad

    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            d = sat_distance(inner_hexes[i], inner_hexes[j])
            if d < -1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: spread hexagons far apart
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    outer_side = best_side
    
    return inner, (cx, cy), outer_side, np.degrees(best_outer_angle)
# EVOLVE-BLOCK-END
