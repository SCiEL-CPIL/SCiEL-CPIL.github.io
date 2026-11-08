import numpy as np
import time
from itertools import combinations

N = 11

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    angles = np.deg2rad(angle_deg) + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point is inside hexagon using half-plane method."""
    dx = px - cx
    dy = py - cy
    theta = np.deg2rad(angle_deg)
    rx = dx * np.cos(theta) + dy * np.sin(theta)
    ry = -dx * np.sin(theta) + dy * np.cos(theta)
    
    apothem = side * np.sqrt(3) / 2
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees in local frame
    for na in [30, 90, 150, 210, 270, 330]:
        na_rad = np.deg2rad(na)
        dist = rx * np.cos(na_rad) + ry * np.sin(na_rad)
        if dist > apothem + 1e-9:
            return False
    return True

def _min_enclosing_hex(centers, angles, side=1.0):
    """Find minimum enclosing regular hexagon. Returns (side, center, rotation_deg)."""
    all_verts = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], side, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = np.inf
    best_center = (0.0, 0.0)
    best_rot = 0.0
    
    # Try rotations from 0 to 60 degrees
    for rot_deg in np.arange(0, 60.01, 2.0):
        theta = np.deg2rad(rot_deg)
        # Edge normals of outer hexagon
        normal_angles = np.deg2rad(rot_deg + np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto each normal
        projections = all_verts @ normals.T  # shape (n_verts, 6)
        M = np.max(projections, axis=0)  # max projection on each normal
        m = np.min(projections, axis=0)  # min projection on each normal
        
        # Required side: s such that s*sqrt(3)/2 >= max(M_i - c.n_i) and s*sqrt(3)/2 >= max(c.n_i - m_i)
        # Optimal: s*sqrt(3)/2 = max_i ((M_i - m_i)/2) ... no
        # We need: M_i - c.n_i <= s*sqrt(3)/2 and c.n_i - m_i <= s*sqrt(3)/2
        # => c.n_i >= M_i - s*sqrt(3)/2 and c.n_i <= m_i + s*sqrt(3)/2
        # Feasibility: M_i - s*sqrt(3)/2 <= m_i + s*sqrt(3)/2 => s >= (M_i - m_i)/sqrt(3)
        # So s_min = max_i (M_i - m_i) / sqrt(3)
        
        s_min = np.max(M - m) / np.sqrt(3)
        
        if s_min < best_side:
            # Find center: c.n_i should be in [M_i - s*sqrt(3)/2, m_i + s*sqrt(3)/2]
            # Target: (M_i + m_i)/2 for each i
            target = (M + m) / 2
            # Solve normals @ c = target (least squares, 6 equations, 2 unknowns)
            c, _, _, _ = np.linalg.lstsq(normals, target, rcond=None)
            best_side = s_min
            best_center = (float(c[0]), float(c[1]))
            best_rot = float(rot_deg)
    
    return best_side, best_center, best_rot

def _check_validity(centers, angles, outer_center, outer_side, outer_rot, inner_side=1.0):
    """Check all inner hexagons are inside outer and pairwise disjoint."""
    # Containment
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], inner_side, angles[i])
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_rot):
                return False
    
    # Disjointness using SAT
    def hex_overlap(c1, a1, c2, a2, s=1.0):
        v1 = _hex_vertices(c1[0], c1[1], s, a1)
        v2 = _hex_vertices(c2[0], c2[1], s, a2)
        
        def get_normals(verts):
            ns = []
            for i in range(6):
                edge = verts[(i+1)%6] - verts[i]
                n = np.array([-edge[1], edge[0]])
                n = n / (np.linalg.norm(n) + 1e-15)
                ns.append(n)
            return np.array(ns)
        
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        
        for n in np.vstack([n1, n2]):
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                return False  # Separated on this axis
        return True  # Overlapping
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            if hex_overlap(centers[i], angles[i], centers[j], angles[j]):
                return False
    
    return True

def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin
    
    best_side = np.inf
    best_result = None
    
    d = np.sqrt(3)  # Distance between adjacent hexagon centers in honeycomb
    
    # Generate multiple initial configurations
    configs = []
    
    # Honeycomb flower: 1 + 6 + 4
    center = np.array([0.0, 0.0])
    ring1 = [center + d * np.array([np.cos(np.deg2rad(k*60)), np.sin(np.deg2rad(k*60))]) for k in range(6)]
    ring2 = [center + 2*d * np.array([np.cos(np.deg2rad(k*60)), np.sin(np.deg2rad(k*60))]) for k in range(6)]
    
    for subset in combinations(range(6), 4):
        pos = [center] + ring1 + [ring2[i] for i in subset]
        configs.append((np.array(pos), np.zeros(11)))
    
    # Staggered rows
    for y_off in [0.0, d*0.866]:
        pos = []
        for i in range(6):
            pos.append(np.array([(i-2.5)*d, 0.0]))
        for i in range(5):
            pos.append(np.array([(i-2)*d, d*0.866]))
        configs.append((np.array(pos), np.zeros(11)))
    
    # Linear
    pos = [np.array([(i-5)*d, 0.0]) for i in range(11)]
    configs.append((np.array(pos), np.zeros(11)))
    
    for cfg_idx, (pos0, ang0) in enumerate(configs):
        if time.time() > deadline - 5:
            break
        
        for inner_rot in [0.0, 30.0, 15.0]:
            if time.time() > deadline - 5:
                break
            
            pos = pos0.copy()
            angles = np.full(11, inner_rot)
            
            # Force-directed relaxation
            for step in range(300):
                if time.time() > deadline - 3:
                    break
                forces = np.zeros_like(pos)
                for i in range(N):
                    for j in range(i+1, N):
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff)
                        if dist < 1e-10:
                            continue
                        min_d = d
                        if dist < min_d:
                            f = (min_d - dist) * 3.0
                            forces[i] += f * diff / dist
                            forces[j] -= f * diff / dist
                pos += 0.05 * forces
                pos -= np.mean(pos, axis=0)
            
            side, center_c, rot = _min_enclosing_hex(pos, angles)
            if side < best_side:
                if _check_validity(pos, angles, center_c, side, rot):
                    best_side = side
                    best_result = (pos, angles, center_c, side, rot)
    
    if best_result is None:
        # Fallback
        pos = np.array([(i-5)*d for i in range(11)], dtype=float).reshape(-1, 1)
        pos = np.hstack([pos, np.zeros((11, 1))])
        angles = np.zeros(11)
        side, center_c, rot = _min_enclosing_hex(pos, angles)
        side *= 1.05
        best_result = (pos, angles, center_c, side, rot)
    
    pos, angles, center_c, side, rot = best_result
    inner = [[float(pos[i][0]), float(pos[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(center_c[0]), float(center_c[1])), float(side), float(rot)

