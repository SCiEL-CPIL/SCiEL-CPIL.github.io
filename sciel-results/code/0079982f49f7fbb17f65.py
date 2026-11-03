import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * s1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
        
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s_val = eval_phi(phi_deg)
            if s_val < best_side:
                best_side = s_val
                best_phi = phi_deg
        
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    def generate_honeycomb(seed_offset=0):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        patterns = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(-2,0),(1,-1),(-1,1),(0,2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,-1),(-2,-1),(1,-1),(-1,-1),(0,2)],
        ]
        idx = (seed_offset) % len(patterns)
        lattice_coords = patterns[idx]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (li, lj) in enumerate(lattice_coords):
            positions[i] = li * a1 + lj * a2
            angles[i] = 0.0
        return positions, angles

    def generate_random_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = random.uniform(0, 60)
        
        for i in range(1, N):
            for attempt in range(50):
                ref_idx = random.randint(0, i-1)
                dx = random.uniform(-3.5, 3.5)
                dy = random.uniform(-3.5, 3.5)
                positions[i] = positions[ref_idx] + [dx, dy]
                angles[i] = random.uniform(0, 60)
                
                valid = True
                angles_rad = np.radians(angles[:i+1])
                h_new = (positions[i, 0], positions[i, 1], angles_rad[i])
                for j in range(i):
                    h_other = (positions[j, 0], positions[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h_new, h_other):
                        valid = False
                        break
                if valid:
                    break
            
            if not valid:
                positions[i] = [random.uniform(-5, 5), random.uniform(-5, 5)]
                angles[i] = random.uniform(0, 60)
        
        return positions, angles

    def smooth_objective(params):
        # Unpack
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        angles_rad = np.radians(ang)
        
        # Compute outer side length
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + pos[:, 0:1]
        verts_y = ry + pos[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
        
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        # Optimal orientation is roughly 0 for honeycomb, but we can just take max projection
        # To make it smooth, we use a soft-max or just evaluate at 0 and 30 degrees?
        # Actually, the outer hexagon orientation is a free variable in the return, but fixed here for optimization speed.
        # We assume the outer hexagon aligns with the "average" orientation or we just maximize projection at 0.
        # Better: The outer hexagon side length is determined by the max projection.
        # Let's evaluate at a few angles and take the max to be safe, or just 0.
        # For speed, we evaluate at 0.
        normal_angles = np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_len = np.max(side_vals)
        
        # Compute overlap penalty
        # For each pair, check separation along 12 normals
        penalty = 0.0
        beta = 10.0 # Temperature for log-sum-exp
        
        v1x = pos[:, 0:1] + _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        v1y = pos[:, 1:2] + _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        en1x = _enx[None, :] * c[:, None] - _eny[None, :] * s[:, None]
        en1y = _enx[None, :] * s[:, None] + _eny[None, :] * c[:, None]
        
        # v1x shape (N, 6), en1x shape (N, 6)
        
        for i in range(N):
            for j in range(i+1, N):
                # Normals for i and j
                n1x = en1x[i]
                n1y = en1y[i]
                n2x = en1x[j]
                n2y = en1y[j]
                
                p1x = v1x[i]
                p1y = v1y[i]
                p2x = v1x[j]
                p2y = v1y[j]
                
                # Projections of i on normals of i
                proj1_i = p1x * n1x + p1y * n1y
                proj2_i = p2x * n1x + p2y * n1y
                
                # Projections of j on normals of j
                proj1_j = p1x * n2x + p1y * n2y
                proj2_j = p2x * n2x + p2y * n2y
                
                # Separation distance for each normal
                # If max(A) < min(B) or max(B) < min(A), they are separated.
                # Overlap amount is max(0, min(maxA, maxB) - max(minA, minB))
                # Wait, the condition for disjointness is:
                # For ANY normal n: max(dot(v1, n)) < min(dot(v2, n)) OR max(dot(v2, n)) < min(dot(v1, n))
                # So if for ALL normals, the intervals [min1, max1] and [min2, max2] overlap, they intersect.
                
                # Let's compute the "overlap depth" for each of the 12 normals.
                # Overlap depth d_k = max(0, min(max1_k, max2_k) - max(min1_k, min2_k))
                # If sum of d_k > 0, they overlap.
                
                min1_i = np.min(proj1_i)
                max1_i = np.max(proj1_i)
                min2_i = np.min(proj2_i)
                max2_i = np.max(proj2_i)
                
                min1_j = np.min(proj1_j)
                max1_j = np.max(proj1_j)
                min2_j = np.min(proj2_j)
                max2_j = np.max(proj2_j)
                
                overlaps = []
                for k in range(6):
                    # Normal k of hexagon i
                    m1 = proj1_i[k]
                    m2 = proj2_i[k]
                    # Intervals are [m1 - R, m1 + R] ? No, vertices are given.
                    # proj1_i[k] is the projection of vertex k of hexagon i? No.
                    # v1x[i] is the 6 vertices of hexagon i.
                    # n1x[i] are the 6 normals of hexagon i.
                    # proj1_i[k] = dot(vertex_k, normal_k). This is NOT the projection of the whole hexagon on normal_k.
                    # The projection of the whole hexagon on normal_k is [min(dot(v, n_k)), max(dot(v, n_k))].
                    
                    # We need to compute min and max of dot(v1, n_k) for all vertices of v1.
                    # This is expensive to do inside the loop for every pair.
                    
                # Optimization: Precompute vertex projections?
                # This function is called many times. We need it to be fast.
                
                # Let's use a simpler penalty: distance between centers minus radius?
                # No, that's not accurate for hexagons.
                
                # Alternative: Use the fact that if they are disjoint, there is a separating axis.
                # If they overlap, the minimum separation distance is negative.
                # We can compute the minimum of the "separation" over the 12 normals.
                # Separation s_k = min(max1_k, max2_k) - max(min1_k, min2_k)
                # If s_k < 0 for all k, they overlap.
                # Penalty = log-sum-exp(s_k)
                
                # To compute min/max projections efficiently:
                # For hexagon i and normal n:
                # min_proj = min(dot(v1x, n_x) + dot(v1y, n_y))
                
                # Let's do it properly but vectorized per pair if possible, or just loop.
                # Given N=12, pairs=66. 12 normals per pair. 792 operations.
                # Inside minimize, this might be slow if not vectorized well.
                
                # Let's try to vectorize the projection calculation for the pair.
                # v1: (6,2), v2: (6,2)
                # n1: (6,2), n2: (6,2)
                
                # Projections of v1 on n1:
                # We need min and max of dot(v1, n1[k]) for k=0..5
                # This is equivalent to:
                # P1 = v1 @ n1.T -> (6,6) matrix where P1[i,k] = dot(v1[i], n1[k])
                # min1_k = min(P1[:, k])
                # max1_k = max(P1[:, k])
                
                P1 = np.dot(v1x[i], n1x[i]) + np.dot(v1y[i], n1y[i]) # (6,6)
                min1 = np.min(P1, axis=0)
                max1 = np.max(P1, axis=0)
                
                P2 = np.dot(v1x[j], n1x[i]) + np.dot(v1y[j], n1y[i]) # (6,6)
                min2 = np.min(P2, axis=0)
                max2 = np.max(P2, axis=0)
                
                # Separation for normals of i
                sep1 = np.minimum(max1, max2) - np.maximum(min1, min2)
                
                P1_j = np.dot(v1x[i], n2x[j]) + np.dot(v1y[i], n2y[j]) # (6,6)
                min1_j = np.min(P1_j, axis=0)
                max1_j = np.max(P1_j, axis=0)
                
                P2_j = np.dot(v1x[j], n2x[j]) + np.dot(v1y[j], n2y[j]) # (6,6)
                min2_j = np.min(P2_j, axis=0)
                max2_j = np.max(P2_j, axis=0)
                
                sep2 = np.minimum(max1_j, max2_j) - np.maximum(min1_j, min2_j)
                
                all_seps = np.concatenate([sep1, sep2])
                
                # If all seps are negative, they overlap.
                # Penalty is based on how much they overlap.
                # We want to push them apart, so we penalize negative separation.
                # soft_min = -log-sum-exp(-sep) ?
                # If sep is large positive (separated), -sep is large negative, exp is 0, penalty 0.
                # If sep is negative (overlap), -sep is positive, exp is large, penalty large.
                
                # Use log-sum-exp for numerical stability
                max_neg_sep = np.max(-all_seps)
                if max_neg_sep > 100:
                    penalty += 100.0
                else:
                    penalty += -max_neg_sep - np.log(np.sum(np.exp(-all_seps - max_neg_sep)))
        
        return side_len + 10.0 * penalty

    # Portfolio approach
    candidates = []
    for offset in range(3):
        pos, ang = generate_honeycomb(offset)
        candidates.append((pos, ang))
    
    for seed in range(5):
        random.seed(seed * 12345)
        pos, ang = generate_random_cluster()
        candidates.append((pos, ang))
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate initial candidates
    for i, (pos, ang) in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 20:
            break
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        if side < best_side:
            best_side = side
            best_positions = pos
            best_angles = ang
            best_phi = phi
            best_cx = cx
            best_cy = cy
            
    # Optimize top candidates
    candidates.sort(key=lambda x: compute_min_outer_side(x[0], x[1])[0])
    
    time_per_opt = (DEADLINE - (time.time() - t0) - 10) / len(candidates)
    
    for i, (pos, ang) in enumerate(candidates[:5]): # Only optimize top 5
        if time.time() - t0 > DEADLINE - 10:
            break
            
        init_params = np.zeros(3*N)
        for j in range(N):
            init_params[3*j] = pos[j, 0]
            init_params[3*j+1] = pos[j, 1]
            init_params[3*j+2] = ang[j]
            
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 1000})
            
            if res.fun < 100: # Check if it's a reasonable solution (not huge penalty)
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for j in range(N):
                    new_pos[j, 0] = res.x[3*j]
                    new_pos[j, 1] = res.x[3*j+1]
                    new_ang[j] = res.x[3*j+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass
    
    # Fallback
    if best_positions is None or not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

