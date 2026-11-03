import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300
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

    def check_disjoint(poly1, poly2, eps=1e-9):
        """
        Fast check for disjointness of two convex hexagons using Separating Axis Theorem (SAT).
        Returns True if disjoint, False if overlapping.
        """
        # Collect all axes (normals to edges of both polygons)
        axes = []
        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
        
        for i in range(6):
            p1 = poly2[i]
            p2 = poly2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
        
        # Check projection on each axis
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            proj2 = np.dot(poly2, axis)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they are disjoint
            if max1 + eps < min2 or max2 + eps < min1:
                return True
        
        return False

    # Strategy:
    # 1. Define a symmetric initial configuration.
    #    The outer hexagon has 6-fold symmetry. We can place 3 hexagons in one 60-degree sector
    #    and rotate them to fill the rest.
    #    However, simply rotating 3 hexagons by 60 degrees might cause overlaps if the sector
    #    is not wide enough or if the hexagons are too close to the boundary/center.
    #    A simpler symmetric pattern: 1 in center, 5 in inner ring, 6 in outer ring?
    #    Or just 12 in a honeycomb cluster.
    #    
    #    Let's try a configuration that respects D6 symmetry:
    #    - 1 hex at center (0,0).
    #    - 5 hexes in a ring around it? No, 6 is better for symmetry.
    #    - If we use 1 center + 5 others, we break 6-fold symmetry.
    #    
    #    Let's use a "cluster" of 12 that is roughly hexagonal.
    #    Honeycomb lattice of 19 hexes (center + 6 + 12). We need 12.
    #    Take center (1) + inner ring (6) + 5 from outer ring?
    #    This breaks symmetry.
    #    
    #    Alternative: Use 12 hexes in a 3x4 rectangular-ish block? No, regular hex container.
    #    
    #    Let's stick to the "Sector" idea but allow the optimizer to break symmetry slightly
    #    by optimizing all 12 positions, but initialize with a symmetric guess.
    #    Actually, the prompt suggests reducing DOF.
    #    If we enforce D6 symmetry, we optimize 3 hexes (x,y,angle) and generate the other 9.
    #    Total variables: 3 * 3 = 9.
    #    This is a huge reduction from 36.
    
    # Let's define the symmetric construction.
    # Sector 0: Hex 0, Hex 1, Hex 2
    # Sector 1: Hex 3, Hex 4, Hex 5 (rotated 60 deg)
    # ...
    # Hex i in sector k has angle = base_angle + k*60.
    # Position is rotated by k*60.
    
    def expand_symmetric(params):
        """
        params: array of length 9.
        [x0, y0, a0, x1, y1, a1, x2, y2, a2]
        Returns positions (12, 2) and angles (12,)
        """
        positions = np.zeros((12, 2))
        angles = np.zeros(12)
        
        for k in range(6):
            rot_angle = k * 60.0
            rot_rad = math.radians(rot_angle)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            for j in range(3):
                idx = k * 3 + j
                x = params[j*3]
                y = params[j*3 + 1]
                a = params[j*3 + 2]
                
                pos = np.array([x, y])
                rotated_pos = rot_mat @ pos
                positions[idx] = rotated_pos
                angles[idx] = a + rot_angle
        
        return positions, angles

    # Objective function for symmetric optimization
    def objective_symmetric(params):
        if time.time() > deadline - 5:
            return 1e6
            
        positions, angles = expand_symmetric(params)
        
        # Calculate outer side
        outer_side = calc_min_outer_side(positions, angles, 0.0)
        
        # Check disjointness
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(polys[i], polys[j]):
                    # Estimate penalty based on how much they overlap?
                    # For simplicity, large fixed penalty.
                    penalty += 100.0
                    if penalty > 1000:
                        break
            if penalty > 1000:
                break
        
        return outer_side + penalty

    # Initial guess for the symmetric sector (3 hexes)
    # Place them in a way that they are likely disjoint.
    # Hex 0: Center (0,0), Angle 0
    # Hex 1: (sqrt(3), 0), Angle 0  (Touching flat-to-flat? No, dist sqrt(3) is touching flats)
    # Hex 2: (sqrt(3)/2, 1.5), Angle 0 (Another neighbor)
    
    # Let's refine this initial guess to be safer.
    # In a honeycomb, neighbors are at distance sqrt(3).
    # Let's place:
    # H0: (0,0)
    # H1: (1.732, 0)  # sqrt(3)
    # H2: (0.866, 1.5) # sqrt(3)/2, 1.5 -> dist from H0 is sqrt(0.75+2.25)=sqrt(3)
    
    init_params = np.array([
        0.0, 0.0, 0.0,
        1.732, 0.0, 0.0,
        0.866, 1.5, 0.0
    ])
    
    best_score = float('inf')
    best_params = init_params.copy()
    
    # Optimize using SLSQP
    try:
        res = minimize(
            objective_symmetric,
            init_params,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.fun < best_score:
            # Verify validity
            positions, angles = expand_symmetric(res.x)
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_score = calc_min_outer_side(positions, angles, 0.0)
                best_params = res.x.copy()
    except Exception as e:
        pass

    # If symmetric optimization failed or didn't find a valid solution, 
    # try a free-floating Nelder-Mead from the symmetric initial guess (expanded).
    if best_params is None or not np.any(best_params != init_params) or best_score > 10.0:
         # Expand initial guess to full 12
         init_positions, init_angles = expand_symmetric(init_params)
         
         def objective_free(params):
             if time.time() > deadline - 5:
                 return 1e6
             positions = params[:2*N].reshape(N, 2)
             angles = params[2*N:]
             
             outer_side = calc_min_outer_side(positions, angles, 0.0)
             
             polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
             penalty = 0.0
             for i in range(N):
                 for j in range(i+1, N):
                     if not check_disjoint(polys[i], polys[j]):
                         penalty += 100.0
                         if penalty > 1000: break
                 if penalty > 1000: break
             
             return outer_side + penalty

         x0 = np.concatenate([init_positions.flatten(), init_angles])
         
         try:
             res_free = minimize(
                 objective_free,
                 x0,
                 method='Nelder-Mead',
                 options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
             )
             
             if res_free.fun < best_score:
                 new_positions = res_free.x[:2*N].reshape(N, 2)
                 new_angles = res_free.x[2*N:]
                 
                 polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                 valid = True
                 for i in range(N):
                     for j in range(i+1, N):
                         if not check_disjoint(polys[i], polys[j]):
                             valid = False
                             break
                     if not valid: break
                 
                 if valid:
                     best_score = res_free.fun
                     best_positions = new_positions
                     best_angles = new_angles
         except Exception as e:
             pass
    else:
         # We have a good symmetric solution, expand it for final output
         best_positions, best_angles = expand_symmetric(best_params)

    # Fallback if everything fails
    if 'best_positions' not in locals() or best_positions is None:
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

