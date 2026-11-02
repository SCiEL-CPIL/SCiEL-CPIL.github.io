"""
Strategy: Symmetry-Constrained Gradient Descent.
Instead of searching a 36-dimensional space with discrete steps, we exploit the
high symmetry of the problem (packing hexagons in a hexagon). We assume the
optimal configuration consists of:
1. One central hexagon at the origin.
2. A ring of 6 hexagons at radius r1 and angle offset a1.
3. A ring of 5 hexagons at radius r2 and angle offset a2.
This reduces the problem to 6 parameters: [r1, a1, r2, a2, outer_angle, outer_scale].
Wait, the outer hexagon is determined by the inner ones. We can just optimize the
inner positions and compute the minimal outer hexagon.
Actually, we can optimize the 6 parameters of the inner rings and let the outer
hexagon be the minimal enclosing hexagon.
To make the objective smooth for gradient descent, we use a soft penalty for
overlaps instead of a hard constraint.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def min_outer_side(inner_positions, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center (0,0) that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
        
        all_verts = np.vstack(all_verts)
        
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = np.cos(na)
            ny = np.sin(na)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        return max_proj / (np.sqrt(3) / 2)

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        """
        best_s = float('inf')
        best_theta = 0.0
        
        # Scan theta
        for theta in np.arange(0, 60, 1.0):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Refine theta around best
        for theta in np.arange(best_theta - 1.0, best_theta + 1.0, 0.1):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
                
        return best_s, best_theta

    def compute_inner_positions(params):
        """
        params: [r1, a1, r2, a2]
        r1, r2: radii for ring 1 (6 hexes) and ring 2 (5 hexes)
        a1, a2: angle offsets for the rings
        Returns list of (x, y, angle) for 12 hexagons.
        """
        r1 = params[0]
        a1 = params[1]
        r2 = params[2]
        a2 = params[3]
        
        positions = []
        
        # Central hexagon
        positions.append((0.0, 0.0, 0.0))
        
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = np.radians(60 * k + a1)
            x = r1 * np.cos(angle)
            y = r1 * np.sin(angle)
            # Rotate the hexagon to align with the radial direction? 
            # Or keep them flat? 
            # In a hexagonal lattice, they are usually flat or pointy relative to center.
            # Let's try aligning the "point" towards the center or away.
            # Actually, for minimal packing, we might want to rotate them.
            # Let's assume they are rotated by the same angle as their position for symmetry.
            rot_angle = a1
            positions.append((x, y, rot_angle))
            
        # Ring 2: 5 hexagons
        # These are placed in the "valleys" or further out.
        # Symmetry: 5-fold? No, the outer container is 6-fold.
        # Wait, 1 + 6 + 5 = 12.
        # The 5th ring in a hexagonal lattice usually has 6 hexagons.
        # If we only have 5, we break the 6-fold symmetry of the second ring.
        # However, the problem states "12 disjoint regular hexagons".
        # It doesn't enforce symmetry. But assuming symmetry is a strong prior.
        # Maybe the 5 hexagons are arranged with 5-fold symmetry? Unlikely in a hexagon.
        # More likely, they are placed in 5 of the 6 spots of the second ring, or
        # they are arranged differently.
        # Let's assume the 5 hexagons are placed at angles 0, 72, 144, 216, 288? No, that's 5-fold.
        # Or maybe they are at 0, 60, 120, 180, 240? (5 of the 6 spots).
        # Let's try placing them at 0, 60, 120, 180, 240 degrees with offset a2.
        for k in range(5):
            angle = np.radians(60 * k + a2)
            x = r2 * np.cos(angle)
            y = r2 * np.sin(angle)
            rot_angle = a2
            positions.append((x, y, rot_angle))
            
        return positions

    def objective(params):
        """
        Objective function to minimize.
        Returns outer side length + penalty for overlaps.
        """
        positions = compute_inner_positions(params)
        
        # Compute outer side length
        s, theta = get_min_outer_config(positions)
        
        # Compute overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, a1 = positions[i]
                cx2, cy2, a2 = positions[j]
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                
                # Simple distance check for early exit
                dist = math.hypot(cx1-cx2, cy1-cy2)
                if dist > 4.0: # Max distance between centers for touching unit hexes is ~3.46
                    continue
                
                # SAT check
                axes = []
                for i_v in range(6):
                    p1 = v1[i_v]
                    p2 = v1[(i_v + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                for i_v in range(6):
                    p1 = v2[i_v]
                    p2 = v2[(i_v + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                
                overlap = False
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                        break
                else:
                    overlap = True
                
                if overlap:
                    # Smooth penalty based on distance
                    # Ideal distance for touching hexes is 2*sqrt(3) ~ 3.464
                    # Penalty grows as distance decreases
                    ideal_dist = 2 * math.sqrt(3)
                    if dist < ideal_dist:
                        penalty += (ideal_dist - dist) * 10.0
        
        return s + penalty

    # Initial guess
    # Ring 1: 6 hexes around center. Distance ~ 2*sqrt(3) ~ 3.46
    # Ring 2: 5 hexes further out.
    
    best_params = None
    best_score = float('inf')
    
    # Multi-start with different initial guesses
    starts = [
        [3.46, 0, 5.0, 0],
        [3.46, 30, 5.0, 30],
        [3.0, 0, 5.5, 0],
        [4.0, 0, 6.0, 0],
        [3.46, 0, 6.0, 0],
        [3.46, 0, 5.0, 10],
        [3.46, 10, 5.0, 10],
        [3.46, 0, 4.5, 0],
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
            
        # Use BFGS for optimization
        try:
            res = minimize(objective, start, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback
        best_params = np.array([3.46, 0, 5.0, 0])
        best_score = objective(best_params)
        
    # Final refinement
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract best inner positions
    inner = compute_inner_positions(best_params)
    
    # Compute final outer side and angle
    s, theta = get_min_outer_config(inner)
    
    # Center is (0,0)
    center = (0.0, 0.0)
    side = s
    angle = theta
    
    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: expand side slightly
        side *= 1.01
        
    return inner, center, side, angle

# EVOLVE-BLOCK-END
