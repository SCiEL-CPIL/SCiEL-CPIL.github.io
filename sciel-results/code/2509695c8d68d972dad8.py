"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s

# Pre-compute base hexagon vertices (side=1, angle=0)
_BASE_ANGLES = np.deg2rad(np.arange(0, 360, 60))
_BASE_VERTS = np.column_stack((np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)))

def _hex_vertices_array(cx, cy, angle_deg):
    """Returns (6, 2) array of vertices for a hexagon."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ _BASE_VERTS.T).T + np.array([cx, cy])

def _hex_edge_normals(angle_deg):
    """Returns (6, 2) array of edge normals for a unit hexagon with given rotation."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    base_normals = np.column_stack((np.cos(_BASE_ANGLES + np.pi/6), np.sin(_BASE_ANGLES + np.pi/6)))
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ base_normals.T).T

def _pairwise_overlap_penalty(verts1, verts2, normals1, normals2, tol=1e-7):
    """
    Smooth overlap penalty using SAT projections.
    Returns a non-negative value: 0 if separated, >0 if overlapping.
    Uses the maximum penetration depth across all separating axes.
    """
    all_normals = np.vstack((normals1, normals2))
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    
    min1 = np.min(proj1, axis=1)
    max1 = np.max(proj1, axis=1)
    min2 = np.min(proj2, axis=1)
    max2 = np.max(proj2, axis=1)
    
    # Overlap amount on each axis: positive means overlap
    overlaps = np.minimum(max1 - min2, max2 - min1)
    # The minimum overlap across all axes is the penetration depth
    # If any axis has negative overlap, polygons are separated
    min_overlap = np.min(overlaps)
    if min_overlap <= tol:
        return 0.0
    return min_overlap

def _total_overlap_penalty(cfg, beta=50.0):
    """
    Compute smooth total overlap penalty for all pairs.
    cfg: (N, 3) array of [x, y, angle_deg]
    Returns a scalar penalty (0 if no overlaps).
    """
    verts_list = []
    normals_list = []
    for i in range(N):
        x, y, ang = cfg[i]
        verts_list.append(_hex_vertices_array(x, y, ang))
        normals_list.append(_hex_edge_normals(ang))
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cfg[i][0] - cfg[j][0]
            dy = cfg[i][1] - cfg[j][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.5:  # Hexagons can't overlap if centers > 2*sqrt(3)/sqrt(3) apart
                continue
            penalty = _pairwise_overlap_penalty(verts_list[i], verts_list[j], 
                                                 normals_list[i], normals_list[j])
            # Use smooth exponential penalty
            total_penalty += math.exp(beta * penalty) - 1.0
    return total_penalty

def _outer_hex_side(cfg, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = np.deg2rad(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    
    normals_angles = a + np.deg2rad(90 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
    
    cx, cy = center
    
    max_need = 0.0
    for i in range(N):
        x, y, ang = cfg[i]
        verts = _hex_vertices_array(x, y, ang)
        rel_verts = verts - np.array([cx, cy])
        projections = rel_verts @ normals.T
        max_proj = np.max(projections)
        if max_proj > max_need:
            max_need = max_proj
            
    return max_need / apothem_unit

def _smooth_objective(v, beta=50.0):
    """
    Smooth objective function for optimization.
    v = [x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle]
    Returns outer_side + large_penalty * overlap_penalty
    """
    cfg = v[:3*N].reshape(N, 3)
    center = v[3*N:3*N+2]
    cangle = v[3*N+2]
    
    overlap_pen = _total_overlap_penalty(cfg, beta)
    if overlap_pen > 0.1:
        return 100.0 + overlap_pen
    
    side = _outer_hex_side(cfg, center, cangle)
    return side + 10.0 * overlap_pen

def _lattice_topologies():
    """Generate hexagonal lattice topologies for 12 unit hexagons."""
    topologies = []
    sqrt3 = math.sqrt(3)
    
    # Topology A: 4-3-4-1
    offsets_A = [
        (0, 0), (3, 0), (6, 0), (9, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (4.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_A, [0.0]*12))
    
    # Topology B: 2-3-4-3
    offsets_B = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_B, [0.0]*12))
    
    # Topology C: 3-4-3-2
    offsets_C = [
        (0, 0), (3, 0), (6, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3), (7.5, sqrt3),
        (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_C, [0.0]*12))
    
    # Topology D: 3-3-3-3
    offsets_D = [
        (0, 0), (3, 0), (6, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_D, [0.0]*12))
    
    # Topology E: 4-2-4-2
    offsets_E = [
        (0, 0), (3, 0), (6, 0), (9, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_E, [0.0]*12))
    
    # Topology F: 2-4-2-4
    offsets_F = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3),
        (0, 3*sqrt3/2), (3, 3*sqrt3/2), (6, 3*sqrt3/2), (9, 3*sqrt3/2),
    ]
    topologies.append((offsets_F, [0.0]*12))
    
    # Topology G: 1-4-4-3
    offsets_G = [
        (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_G, [0.0]*12))
    
    # Topology H: 3-4-4-1
    offsets_H = [
        (0, 0), (3, 0), (6, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (4.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_H, [0.0]*12))
    
    # Topology I: 3-3-4-2
    offsets_I = [
        (0, 0), (3, 0), (6, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    topologies.append((offsets_I, [0.0]*12))
    
    # Topology J: 2-4-3-3
    offsets_J = [
        (1.5, 0), (4.5, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    topologies.append((offsets_J, [0.0]*12))
    
    return topologies

def _topology_to_vector(offsets, angles, global_rotation_deg=0.0, center=(0.0, 0.0)):
    """Convert topology to optimization vector."""
    a_rad = np.deg2rad(global_rotation_deg)
    ca, sa = math.cos(a_rad), math.sin(a_rad)
    
    v = np.zeros(3*N + 3)
    for i, (dx, dy) in enumerate(offsets):
        rx = ca * dx - sa * dy
        ry = sa * dx + ca * dy
        ang = angles[i] + global_rotation_deg
        v[3*i] = rx
        v[3*i+1] = ry
        v[3*i+2] = ang
    
    v[3*N] = center[0]
    v[3*N+1] = center[1]
    v[3*N+2] = global_rotation_deg
    return v

def _vector_to_cfg(v):
    """Convert optimization vector to (cfg, center, angle)."""
    cfg = v[:3*N].reshape(N, 3)
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    return cfg, center, angle

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_cfg = None
    best_score = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    topologies = _lattice_topologies()
    
    # Phase 1: Local refinement from each topology using L-BFGS-B
    for offsets, angles in topologies:
        if time.time() - t0 > TIME_LIMIT - 30.0:
            break
        
        # Try multiple initial rotations
        for init_rot in [0, 15, 30, 45]:
            if time.time() - t0 > TIME_LIMIT - 30.0:
                break
            
            v0 = _topology_to_vector(offsets, angles, init_rot, (0.0, 0.0))
            
            # Define bounds
            bounds = []
            for i in range(N):
                bounds.extend([(-15, 15), (-15, 15), (-180, 180)])
            bounds.extend([(-5, 5), (-5, 5), (-90, 90)])
            
            try:
                result = minimize(
                    _smooth_objective,
                    v0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                v_opt = result.x
                cfg_opt, center_opt, angle_opt = _vector_to_cfg(v_opt)
                
                # Check if valid (no overlaps)
                overlap_pen = _total_overlap_penalty(cfg_opt, beta=50.0)
                if overlap_pen < 0.01:
                    side_opt = _outer_hex_side(cfg_opt, center_opt, angle_opt)
                    if side_opt < best_score:
                        best_score = side_opt
                        best_cfg = cfg_opt.copy()
                        best_center = center_opt
                        best_angle = angle_opt
                        
            except Exception:
                pass
    
    # Phase 2: Differential evolution for global search (if time permits)
    if time.time() - t0 < TIME_LIMIT - 40.0 and best_cfg is not None:
        # Use best solution as starting point for DE
        # Define bounds for DE
        de_bounds = []
        for i in range(N):
            de_bounds.extend([(-12, 12), (-12, 12), (-180, 180)])
        de_bounds.extend([(-3, 3), (-3, 3), (-90, 90)])
        
        # Create initial population centered around best solution
        x0 = np.zeros(3*N + 3)
        for i in range(N):
            x0[3*i] = best_cfg[i][0]
            x0[3*i+1] = best_cfg[i][1]
            x0[3*i+2] = best_cfg[i][2]
        x0[3*N] = best_center[0]
        x0[3*N+1] = best_center[1]
        x0[3*N+2] = best_angle
        
        # Create multiple starting points from topologies
        init_points = [x0]
        for offsets, angles in topologies[:4]:
            for rot in [0, 30]:
                v = _topology_to_vector(offsets, angles, rot, (0.0, 0.0))
                init_points.append(v)
        
        def de_objective(v):
            return _smooth_objective(v, beta=40.0)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='latinhypercube'
            )
            
            v_de = de_result.x
            cfg_de, center_de, angle_de = _vector_to_cfg(v_de)
            
            overlap_pen = _total_overlap_penalty(cfg_de, beta=50.0)
            if overlap_pen < 0.01:
                side_de = _outer_hex_side(cfg_de, center_de, angle_de)
                if side_de < best_score:
                    best_score = side_de
                    best_cfg = cfg_de.copy()
                    best_center = center_de
                    best_angle = angle_de
                    
        except Exception:
            pass
    
    # Phase 3: Final polish with simulated annealing (if time permits)
    if time.time() - t0 < TIME_LIMIT - 15.0 and best_cfg is not None:
        cfg = [list(best_cfg[i]) for i in range(N)]
        center = list(best_center)
        angle = best_angle
        cur_score = best_score
        
        T = 0.3
        T_end = 0.002
        alpha_sa = 0.9998
        
        nm_time_budget = TIME_LIMIT - 5.0
        
        while time.time() - t0 < nm_time_budget:
            r = random.random()
            
            if r < 0.5:
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                step = 0.15 * (1.0 + T)
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
            elif r < 0.7:
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
            elif r < 0.8:
                old_cx, old_cy = center[0], center[1]
                step = 0.08 * (1.0 + T)
                center[0] += random.gauss(0, step)
                center[1] += random.gauss(0, step)
            elif r < 0.9:
                old_angle = angle
                angle += random.gauss(0, 2.0 + 2.0 * T)
            else:
                idx1, idx2 = random.sample(range(N), 2)
                old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
                old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
                step = 0.1 * (1.0 + T)
                cfg[idx1][0] += random.gauss(0, step)
                cfg[idx1][1] += random.gauss(0, step)
                cfg[idx2][0] += random.gauss(0, step)
                cfg[idx2][1] += random.gauss(0, step)
            
            # Check validity using smooth penalty
            cfg_array = np.array(cfg)
            overlap_pen = _total_overlap_penalty(cfg_array, beta=50.0)
            
            if overlap_pen > 0.05:
                # Revert
                if r < 0.5:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.7:
                    cfg[idx][2] = old_ang
                elif r < 0.8:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.9:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                T = max(T_end, T * alpha_sa)
                continue
            
            new_score = _outer_hex_side(cfg_array, tuple(center), angle)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = cfg_array.copy()
                    best_center = tuple(center)
                    best_angle = angle
            else:
                if r < 0.5:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.7:
                    cfg[idx][2] = old_ang
                elif r < 0.8:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.9:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            
            T = max(T_end, T * alpha_sa)
    
    # Fallback if no valid configuration found
    if best_cfg is None:
        cfg = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 10.0
            cfg.append([r * math.cos(theta), r * math.sin(theta), 0.0])
        best_cfg = np.array(cfg)
        best_score = 20.0
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    inner = [[float(best_cfg[i][0]), float(best_cfg[i][1]), float(best_cfg[i][2])] for i in range(N)]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
