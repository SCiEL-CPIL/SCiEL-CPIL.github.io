# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    def get_hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def hex_separation(h1, h2):
        """Check if two unit hexagons are separated (non-intersecting).
        h1, h2: (cx, cy, angle_deg)
        Returns True if separated by more than 1e-6 on at least one of 12 edge normals.
        """
        cx1, cy1, a1 = h1
        cx2, cy2, a2 = h2
        v1 = get_hex_vertices(cx1, cy1, 1.0, a1)
        v2 = get_hex_vertices(cx2, cy2, 1.0, a2)
        
        # Edge normals for hex1
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            # Normal pointing outward
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            # Project both hexagons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Check if intervals are apart
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
        
        # Edge normals for hex2
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
        
        return False

    def min_separation(h1, h2):
        """Return the minimum gap between two hexagons (negative if overlapping)."""
        cx1, cy1, a1 = h1
        cx2, cy2, a2 = h2
        v1 = get_hex_vertices(cx1, cy1, 1.0, a1)
        v2 = get_hex_vertices(cx2, cy2, 1.0, a2)
        
        min_gap = np.inf
        # Check all 12 edge normals
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                # Gap on this normal
                gap = min(max1 - max2, max2 - max1)
                # If projections overlap, gap is negative
                if max1 < min2:
                    gap = min1 - max1
                elif max2 < min1:
                    gap = min2 - max2
                else:
                    gap = -1.0  # overlapping on this normal
                min_gap = min(min_gap, gap)
        
        return min_gap

    def compute_min_side(inner_hex):
        """Given inner hexagons, find minimum outer hexagon side length containing all."""
        # Outer hexagon centered at origin with some angle. We'll try angle 0 first.
        # The constraint: every inner vertex must be inside the outer hexagon.
        # For a regular hexagon centered at origin with side length s and angle a,
        # the half-width in each of the 6 normal directions is s.
        # A point (x,y) is inside if for all 6 normals n_i, dot(point, n_i) <= s.
        
        best_side = np.inf
        # Try a few angles for the outer hexagon
        for outer_angle in [0, 30, 15, 45]:
            a_rad = np.deg2rad(outer_angle)
            normals = []
            for i in range(6):
                angle = a_rad + np.deg2rad(i * 60)
                normals.append(np.array([np.cos(angle), np.sin(angle)]))
            
            # For each inner hexagon, compute max projection onto each normal
            required = np.zeros(6)
            for h in inner_hex:
                cx, cy, a_deg = h
                v = get_hex_vertices(cx, cy, 1.0, a_deg)
                for j in range(6):
                    proj = np.dot(v, normals[j])
                    required[j] = max(required[j], np.max(proj))
            
            side = np.max(required)
            best_side = min(best_side, side)
        
        return best_side

    def objective(x):
        """Objective: minimize outer hexagon side length."""
        # x: 33 values = 11 * (cx, cy, angle)
        inner = x.reshape(11, 3)
        # Ensure finite
        if not np.all(np.isfinite(inner)):
            return 1e10
        
        # Check separation between all pairs
        for i in range(11):
            for j in range(i + 1, 11):
                gap = min_separation(inner[i], inner[j])
                if gap < -1e-6:  # overlapping
                    return 1e10 + abs(gap) * 1000
        
        # Compute minimum outer side
        side = compute_min_side(inner)
        if not np.isfinite(side):
            return 1e10
        return side

    # Initial guess: compact hexagonal arrangement
    # 7 hexagons in a flower pattern + 4 more
    # Center at (0,0)
    # 6 surrounding at distance 2, angles 0, 60, 120, 180, 240, 300
    # Then 4 more at the "corners"
    
    r = 2.0  # distance between centers of touching hexagons
    base_positions = [
        [0, 0, 0],  # center
        [r, 0, 0],
        [r * 0.5, r * np.sqrt(3) / 2, 0],
        [-r * 0.5, r * np.sqrt(3) / 2, 0],
        [-r, 0, 0],
        [-r * 0.5, -r * np.sqrt(3) / 2, 0],
        [r * 0.5, -r * np.sqrt(3) / 2, 0],
    ]
    
    # Add 4 more hexagons at the outer corners
    # The flower has extent: max x = r + 1 = 3, max y = r*sqrt(3)/2 + sqrt(3)/2 = sqrt(3) * (r/2 + 1/2)
    # For r=2: max x = 3, max y = sqrt(3) * 1.5 = 2.598
    # Place 4 more at positions that extend the pattern
    # Try placing them at angles 30, 90, 150, 210, 270, 330 but only 4
    
    # Better: place 4 at the "gaps" between the 6 surrounding
    # Actually, let's just use a reasonable initial guess and let optimization handle it
    
    extra_positions = [
        [r * 1.5, r * np.sqrt(3) / 2, 0],  # top-right gap
        [-r * 1.5, r * np.sqrt(3) / 2, 0],  # top-left gap
        [-r * 1.5, -r * np.sqrt(3) / 2, 0],  # bottom-left gap
        [r * 1.5, -r * np.sqrt(3) / 2, 0],  # bottom-right gap
    ]
    
    initial = np.array(base_positions + extra_positions)
    x0 = initial.flatten()
    
    # Try optimization
    best_x = x0.copy()
    best_side = compute_min_side(initial)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if result.fun < best_side:
                best_x = result.x
                best_side = result.fun
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        # Try another optimization from a different starting point
        # Rotate all hexagons slightly
        x1 = x0.copy()
        x1[2::3] += 5  # add 5 degrees to all angles
        try:
            result = minimize(objective, x1, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if result.fun < best_side:
                best_x = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Final result
    inner_hex_data = best_x.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
