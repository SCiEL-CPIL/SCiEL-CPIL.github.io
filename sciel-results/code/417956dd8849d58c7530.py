import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def get_gap(v1, v2):
        """
        Calculate the maximum gap between two polygons.
        Positive gap means they are disjoint (separated by that distance).
        Negative gap means they overlap (penetration depth).
        Uses the Separating Axis Theorem (SAT).
        """
        max_gap = -float('inf')
        # We only need to check normals for 3 edges, as opposite edges are parallel
        for i in range(3):
            # Edge 1 from polygon 1
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    # Overlapping on this axis
                    gap = - (min(max1, max2) - max(min1, min2))
                
                if gap > max_gap:
                    max_gap = gap
            
            # Edge 2 from polygon 2
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = - (min(max1, max2) - max(min1, min2))
                
                if gap > max_gap:
                    max_gap = gap
        
        return max_gap

    def check_disjoint(poly1, poly2):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint, False if overlapping.
        """
        # Quick bounding box check
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return True

        # Use SAT gap for robustness and smoothness if needed later
        gap = get_gap(poly1, poly2)
        return gap >= -1e-9

    # Strategy:
    # 1. Generate candidate configurations using a structured search (Honeycomb lattice).
    # 2. For each configuration, verify validity and calculate score.
    # 3. Keep the best valid configuration.
    # 4. Perform a local refinement on the best configuration using Nelder-Mead
    #    with a smooth penalty function based on SAT gap.

    best_score = float('inf')
    best_positions = None
    best_angles = None

    sqrt3 = math.sqrt(3)
    
    # Generate a dense set of candidate positions on a hexagonal lattice
    # Spacing sqrt(3) (touching flat-to-flat)
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Candidate 1: The 12 closest points to the origin on this lattice.
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # Check validity
    polys = [get_hex_verts(init_positions[i][0], init_positions[i][1], 1.0, 0.0) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not check_disjoint(polys[i], polys[j]):
                valid = False
                break
        if not valid: break
    
    if valid:
        score = calc_min_outer_side(init_positions, init_angles, 0.0)
        if score < best_score:
            best_score = score
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()
            
    # Let's try to refine the best configuration found so far.
    if best_positions is not None:
        # Local refinement using Nelder-Mead
        # We will optimize positions and angles.
        # To ensure validity, we use a penalty function based on SAT gap.
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
                
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if penalty > 10.0:
                        break
                    gap = get_gap(polys[i], polys[j])
                    if gap < 0:
                        # Penalize overlap. The penalty should be smooth.
                        # gap is negative. We want to push them apart.
                        # penalty += -gap * 100.0 might be too strong or weak.
                        # Let's use a quadratic penalty for overlap depth.
                        penalty += (-gap) ** 2 * 10.0
                if penalty > 10.0:
                    break
            
            base_cost = calc_min_outer_side(positions, angles, 0.0)
            
            if penalty > 0:
                return base_cost + penalty
            else:
                return base_cost
            
        # Initial guess
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_score:
                # Verify validity of the result
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_score = res.fun
                    best_positions = new_positions
                    best_angles = new_angles
        except Exception as e:
            pass

    # Fallback if no valid configuration was found (should not happen)
    if best_positions is None:
        # Use the simple honeycomb fallback
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_positions = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        best_angles = np.zeros(N)
        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
