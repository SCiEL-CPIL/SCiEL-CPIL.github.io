"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import random

N = 12

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy)."""
    angle = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside a regular hexagon."""
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * np.cos(-angle) - dy * np.sin(-angle)
    ry = dx * np.sin(-angle) + dy * np.cos(-angle)
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem = side * np.sqrt(3) / 2
    projections = rx * normals[:, 0] + ry * normals[:, 1]
    return np.all(projections <= apothem + 1e-9)

def hexagon_separation(p1, a1, p2, a2):
    """Minimum separation between two unit hexagons (positive if disjoint)."""
    v1 = hex_vertices(p1[0], p1[1], 1.0, a1)
    v2 = hex_vertices(p2[0], p2[1], 1.0, a2)
    
    def get_normals(verts):
        normals = []
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            normals.append(n)
        return normals
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = n1 + n2
    
    min_sep = np.inf
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() < proj2.min():
            sep = proj2.min() - proj1.max()
        elif proj2.max() < proj1.min():
            sep = proj1.min() - proj2.max()
        else:
            sep = 0.0
        min_sep = min(min_sep, sep)
    
    if min_sep < 1e-9:
        d = np.linalg.norm(np.array(p1) - np.array(p2))
        return -max(0.5 - d) * 10
    return min_sep

def outer_penalty(px, py, pangle, outer_side, outer_angle, outer_cx, outer_cy):
    """Penalty for inner hexagon extending outside outer hexagon."""
    inner_verts = hex_vertices(px, py, 1.0, pangle)
    penalty = 0.0
    angle = np.radians(outer_angle)
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem = outer_side * np.sqrt(3) / 2
    
    for v in inner_verts:
        dx = v[0] - outer_cx
        dy = v[1] - outer_cy
        rx = dx * np.cos(-angle) - dy * np.sin(-angle)
        ry = dx * np.sin(-angle) + dy * np.cos(-angle)
        projections = rx * normals[:, 0] + ry * normals[:, 1]
        excess = np.maximum(0, projections - apothem)
        penalty += np.sum(excess)
    return penalty

def objective(params, outer_side, outer_angle, outer_cx, outer_cy):
    """Total penalty: pairwise collisions + outer boundary violations."""
    total = 0.0
    positions = []
    for i in range(N):
        positions.append((params[3*i], params[3*i+1], params[3*i+2]))
    
    for i in range(N):
        for j in range(i+1, N):
            sep = hexagon_separation((positions[i][0], positions[i][1]), positions[i][2],
                                     (positions[j][0], positions[j][1]), positions[j][2])
            if sep < 0:
                total += sep * sep * 10
    
    for i in range(N):
        pen = outer_penalty(positions[i][0], positions[i][1], positions[i][2],
                           outer_side, outer_angle, outer_cx, outer_cy)
        total += pen * pen * 5
    
    return total

def construct():
    """Construct 12 unit hexagons inside a minimal regular hexagon."""
    random.seed(42)
    
    # Initial configuration: hexagons spread in a circle
    params0 = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 5.0
        params0.extend([r * np.cos(angle), r * np.sin(angle), 0.0])
    
    best_params = None
    best_side = np.inf
    
    for trial in range(3):
        params = list(params0)
        if trial > 0:
            for i in range(len(params)):
                if i % 3 == 0:
                    params[i] += random.uniform(-2, 2)
                elif i % 3 == 1:
                    params[i] += random.uniform(-2, 2)
                else:
                    params[i] = random.uniform(0, 60)
        
        # Start with large outer side
        outer_side = 8.0
        outer_angle = 0.0
        outer_cx = 0.0
        outer_cy = 0.0
        
        # Optimize inner positions
        res = minimize(objective, params, args=(outer_side, outer_angle, outer_cx, outer_cy),
                      method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-6})
        
        # Binary search on outer side
        lo, hi = 3.0, outer_side
        for _ in range(8):
            mid = (lo + hi) / 2
            test = res.x.copy()
            r = minimize(objective, test, args=(mid, outer_angle, outer_cx, outer_cy),
                        method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-6})
            if r.fun < 1e-4:
                hi = mid
                res = r
            else:
                lo = mid
        
        if hi < best_side:
            best_side = hi
            best_params = res.x.copy()
    
    # Final polish
    res = minimize(objective, best_params, args=(best_side, 0.0, 0.0, 0.0),
                  method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
    best_params = res.x
    
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
# EVOLVE-BLOCK-END
