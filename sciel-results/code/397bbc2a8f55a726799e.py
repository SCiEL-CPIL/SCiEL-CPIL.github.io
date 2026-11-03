import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle_rad
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def min_outer_side_for_vertices(verts, outer_angle):
    # verts: (M, 2)
    # outer_angle: radians
    phi = outer_angle
    normals = np.column_stack([np.cos(np.arange(6)*np.pi/3 + phi), 
                               np.sin(np.arange(6)*np.pi/3 + phi)])
    proj = verts @ normals.T
    max_proj = np.max(proj)
    return 2.0 / np.sqrt(3.0) * max_proj

def overlap_penalty(inner, eps=1e-9):
    # inner: list of (x, y, angle_deg)
    # Returns sum of squared penetration depths
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            v1 = get_hex_vertices(x1, y1, a1)
            v2 = get_hex_vertices(x2, y2, a2)
            
            axes = []
            for v in [v1, v2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-10:
                        normal = normal / norm
                        axes.append(normal)
            
            min_dist = np.inf
            for ax in axes:
                proj1 = v1 @ ax
                proj2 = v2 @ ax
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = max1 - min2
                gap2 = max2 - min1
                gap = min(gap1, gap2)
                
                if gap < -eps:
                    min_dist = min(min_dist, gap)
                else:
                    min_dist = 0.0
                    break
            
            if min_dist < 0:
                penalty += min_dist**2
                
    return penalty

def construct():
    start_time = time.time()
    deadline = 145.0
    
    sqrt3 = np.sqrt(3)
    
    # Corrected 3x4 Topology (Valid non-overlapping)
    # Row 0 (y=0): x = -3, -1, 1, 3
    # Row 1 (y=sqrt3): x = -2, 0, 2, 4
    # Row 2 (y=2*sqrt3): x = -3, -1, 1, 3
    # Center this cluster?
    # X range: -3 to 4. Center X = 0.5. Shift left by 0.5.
    # Row 0: -3.5, -1.5, 0.5, 2.5
    # Row 1: -2.5, -0.5, 1.5, 3.5
    # Row 2: -3.5, -1.5, 0.5, 2.5
    
    centers_3x4 = []
    for k in range(3):
        y = k * sqrt3
        if k % 2 == 0:
            xs = [-3.5, -1.5, 0.5, 2.5]
        else:
            xs = [-2.5, -0.5, 1.5, 3.5]
        for x in xs:
            centers_3x4.append((x, y, 0.0))
            
    # Corrected 2x6 Topology
    # Row 0 (y=0): 6 hexagons. x = -5, -3, -1, 1, 3, 5
    # Row 1 (y=sqrt3): 6 hexagons. x = -4, -2, 0, 2, 4, 6
    # Center X: Row 0 center 0. Row 1 center 1.
    # Shift Row 1 left by 1? Or average?
    # Let's shift both to center at 0.
    # Row 0: -5, -3, -1, 1, 3, 5. Center 0.
    # Row 1: -4, -2, 0, 2, 4, 6. Center 1.
    # Shift Row 1 by -1: -5, -3, -1, 1, 3, 5.
    # But then they are vertically aligned.
    # Distance between (0,0) and (0, sqrt3) is sqrt3.
    # Do they overlap?
    # H1 top edge y=sqrt3/2. H2 bottom edge y=sqrt3 - sqrt3/2 = sqrt3/2.
    # They touch.
    # So 2x6 can be stacked directly?
    # Yes, if aligned.
    
    centers_2x6 = []
    for k in range(2):
        y = k * sqrt3
        xs = [-5, -3, -1, 1, 3, 5]
        for x in xs:
            centers_2x6.append((x, y, 0.0))

    best_score = 1e10
    best_inner = None
    best_outer_angle = 0.0
    
    topologies = [("3x4", centers_3x4), ("2x6", centers_2x6)]
    
    for name, centers in topologies:
        if time.time() > deadline - 10:
            break
            
        # Initial check
        init_pen = overlap_penalty(centers)
        if init_pen > 1e-6:
            continue # Skip if initial is invalid
            
        # Optimize
        x0_local = []
        for x, y, _ in centers:
            x0_local.extend([x, y, 0.0])
        x0_local.append(0.0) # outer angle
        
        def obj_local(params):
            if time.time() > deadline - 5:
                return 1e10
            inner = []
            for k in range(12):
                x = params[3*k]
                y = params[3*k+1]
                th = params[3*k+2]
                inner.append((x, y, th))
            outer_angle = params[36]
            
            pen = overlap_penalty(inner)
            if pen > 1e-4:
                return 1e8 + pen # Strong penalty for invalid
                
            verts = []
            for x, y, ang in inner:
                verts.append(get_hex_vertices(x, y, ang))
            verts = np.vstack(verts)
            side = min_outer_side_for_vertices(verts, np.deg2rad(outer_angle))
            
            return side
            
        try:
            res_local = minimize(obj_local, x0_local, method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            
            # Check validity of result
            inner_final = []
            for k in range(12):
                x = res_local.x[3*k]
                y = res_local.x[3*k+1]
                th = res_local.x[3*k+2]
                inner_final.append([x, y, th])
            outer_angle_final = res_local.x[36]
            
            pen = overlap_penalty(inner_final)
            if pen < 1e-6:
                verts = []
                for x, y, ang in inner_final:
                    verts.append(get_hex_vertices(x, y, ang))
                verts = np.vstack(verts)
                side_final = min_outer_side_for_vertices(verts, np.deg2rad(outer_angle_final))
                
                if side_final < best_score:
                    best_score = side_final
                    best_inner = inner_final
                    best_outer_angle = outer_angle_final
                    
        except Exception:
            pass
            
    # Fallback
    if best_inner is None:
        # Use 3x4 initial
        best_inner = [[x, y, th] for x, y, th in centers_3x4]
        verts = []
        for x, y, ang in best_inner:
            verts.append(get_hex_vertices(x, y, ang))
        verts = np.vstack(verts)
        best_score = min_outer_side_for_vertices(verts, 0.0)
        best_outer_angle = 0.0

    return best_inner, (0.0, 0.0), best_score, best_outer_angle
