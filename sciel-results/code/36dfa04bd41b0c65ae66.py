import numpy as np
import time
import math

N = 11

def _hex_vertices(center, angle, side=1.0):
    cx, cy = center
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _check_overlap(poly1, poly2, tol=1e-6):
    polys = [poly1, poly2]
    for p in polys:
        for i in range(len(p)):
            p1 = p[i]
            p2 = p[(i + 1) % len(p)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def _min_enclosing_hexagon_side(centers, angles):
    all_verts = []
    for c, a in zip(centers, angles):
        v = _hex_vertices(c, a)
        all_verts.extend(v)
    all_verts = np.array(all_verts)
    
    min_side = float('inf')
    for theta_deg in range(0, 31, 1):
        theta = np.deg2rad(theta_deg)
        max_proj = 0
        for k in range(6):
            n_theta = theta + k * np.pi / 3
            nx = np.cos(n_theta)
            ny = np.sin(n_theta)
            projs = np.abs(all_verts[:, 0] * nx + all_verts[:, 1] * ny)
            max_proj = max(max_proj, np.max(projs))
        side = 2 * max_proj / np.sqrt(3)
        if side < min_side:
            min_side = side
    return min_side

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    sp = np.sqrt(3)
    centers = []
    angles = [0.0] * N
    
    # 7 center
    centers.append((0.0, 0.0))
    for i in range(6):
        theta = i * np.pi / 3
        centers.append((sp * np.cos(theta), sp * np.sin(theta)))
        
    # 4 from next layer (valid lattice points)
    centers.append((2*sp, 0.0))
    centers.append((1.5*sp, 1.5)) # 3*sp/2, 3/2
    centers.append((-1.5*sp, 1.5))
    centers.append((-2*sp, 0.0))
    
    pos = [list(c) for c in centers]
    
    best_score = _min_enclosing_hexagon_side(pos, angles)
    best_pos = [p[:] for p in pos]
    best_angles = angles[:]
    
    current_pos = [p[:] for p in best_pos]
    current_angles = best_angles[:]
    current_score = best_score
    
    def get_polygons(pos, angles):
        polys = []
        for c, a in zip(pos, angles):
            polys.append(_hex_vertices(c, a))
        return polys
    
    current_polys = get_polygons(current_pos, current_angles)
    
    while time.time() < deadline:
        i = np.random.randint(N)
        
        old_pos_i = current_pos[i][:]
        old_angle_i = current_angles[i]
        old_poly_i = current_polys[i].copy()
        
        dx = np.random.normal(0, 0.2)
        dy = np.random.normal(0, 0.2)
        dangle = np.random.normal(0, 0.1)
        
        new_pos_i = [old_pos_i[0] + dx, old_pos_i[1] + dy]
        new_angle_i = old_angle_i + dangle
        
        current_pos[i] = new_pos_i
        current_angles[i] = new_angle_i
        
        new_poly_i = _hex_vertices(new_pos_i, new_angle_i)
        
        overlap = False
        for j in range(N):
            if i == j: continue
            if _check_overlap(new_poly_i, current_polys[j]):
                overlap = True
                break
        
        if overlap:
            current_pos[i] = old_pos_i
            current_angles[i] = old_angle_i
            current_polys[i] = old_poly_i
            continue
        
        current_polys[i] = new_poly_i
        
        new_score = _min_enclosing_hexagon_side(current_pos, current_angles)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_pos = [p[:] for p in current_pos]
                best_angles = current_angles[:]
        else:
            current_pos[i] = old_pos_i
            current_angles[i] = old_angle_i
            current_polys[i] = old_poly_i
            
    all_verts_best = []
    for c, a in zip(best_pos, best_angles):
        v = _hex_vertices(c, a)
        all_verts_best.extend(v)
    all_verts_best = np.array(all_verts_best)
    centroid = np.mean(all_verts_best, axis=0)
    
    for i in range(N):
        best_pos[i][0] -= centroid[0]
        best_pos[i][1] -= centroid[1]
        
    best_score = _min_enclosing_hexagon_side(best_pos, best_angles)
    
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        a_deg = np.rad2deg(best_angles[i])
        inner.append([x, y, a_deg])
    
    return inner, (0.0, 0.0), best_score, 0.0
