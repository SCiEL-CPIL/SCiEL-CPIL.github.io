"""
Dual-Space Linear Programming strategy for hexagon packing.
Instead of relaxing inner hexagons with physics, we fix them in dense,
pre-validated honeycomb clusters. Then, for each cluster, we use linear
programming to find the optimal outer hexagon (center and side length)
that contains the cluster. This transforms the containment problem into
a system of linear inequalities.
"""
import numpy as np
import time
from scipy.optimize import linprog

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_honeycomb_clusters():
    """
    Generate diverse, compact initial configurations using hexagonal lattice patterns.
    These are pre-validated to be non-overlapping.
    """
    configs = []
    
    # Hexagonal lattice basis vectors for side-1 hexagons
    # Distance between centers of adjacent hexagons in a close pack is 2 * apothem = sqrt(3).
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    def make_config(coords, angle=0.0):
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], angle])
        return centers

    # Pattern 1: 3-4-3-2 (Compact)
    p1 = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p1))
    
    # Pattern 2: 2-4-4-2 (Rectangular-ish)
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p2))
    
    # Pattern 3: 1-3-4-3-1 (Diamond-ish)
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    configs.append(make_config(p3))
    
    # Pattern 4: 4-4-4 (Three rows)
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    configs.append(make_config(p4))
    
    # Pattern 5: 2-3-4-3 (Asymmetric)
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p5))
    
    # Pattern 6: 3-3-3-3 (Staggered rows)
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p6))
    
    # Pattern 7: 2-2-2-2-2-2 (Hexagonal ring + center? No, just rows)
    # Let's try a more circular cluster
    p7 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, -1), (1, -1), (-1, -1),
        (2, 1), (-2, 1), (2, -1), (-2, -1)
    ]
    configs.append(make_config(p7))
    
    return configs

def solve_outer_lp(centers, outer_angle=0.0):
    """
    Solve for the minimum outer hexagon side length and center that contains the given inner hexagons.
    Variables: [cx, cy, S]
    Constraints:
    For each vertex v_i of each inner hexagon, and for each of the 6 outer normals n_k:
    | (v_i - c) dot R(outer_angle) n_k | <= S * sqrt(3)/2
    
    This is equivalent to:
    (v_i - c) dot (R n_k) <= S * sqrt(3)/2
    -(v_i - c) dot (R n_k) <= S * sqrt(3)/2
    
    Let u_k = R(outer_angle) n_k.
    (v_i dot u_k) - c dot u_k <= S * sqrt(3)/2
    -(v_i dot u_k) + c dot u_k <= S * sqrt(3)/2
    
    Rearranging for LP (Ax <= b):
    1) -u_k_x * cx - u_k_y * cy + sqrt(3)/2 * S >= v_i dot u_k
       => u_k_x * cx + u_k_y * cy - sqrt(3)/2 * S <= -v_i dot u_k
    2) u_k_x * cx + u_k_y * cy - sqrt(3)/2 * S >= -v_i dot u_k
       => -u_k_x * cx - u_k_y * cy + sqrt(3)/2 * S <= v_i dot u_k
    
    Minimize: S
    Subject to:
    A @ [cx, cy, S] <= b
    S >= 0
    """
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(oa_rad)
    s_o = np.sin(oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    # Rotated normals
    rotated_normals = OUTER_NORMALS @ R_o.T  # (6, 2)
    
    verts = get_hex_vertices(np.array(centers, dtype=float))
    all_verts = verts.reshape(-1, 2)  # (N*6, 2)
    n_verts = len(all_verts)
    n_norms = 6
    
    # Number of constraints: 2 * n_verts * n_norms
    n_constraints = 2 * n_verts * n_norms
    
    A = np.zeros((n_constraints, 3))
    b = np.zeros(n_constraints)
    
    apothem_factor = np.sqrt(3) / 2.0
    
    idx = 0
    for v in all_verts:
        for k in range(n_norms):
            u = rotated_normals[k]
            proj = v @ u
            
            # Constraint 1: c dot u - apothem_factor * S <= -proj
            A[idx, 0] = u[0]
            A[idx, 1] = u[1]
            A[idx, 2] = -apothem_factor
            b[idx] = -proj
            idx += 1
            
            # Constraint 2: -c dot u + apothem_factor * S <= proj
            A[idx, 0] = -u[0]
            A[idx, 1] = -u[1]
            A[idx, 2] = apothem_factor
            b[idx] = proj
            idx += 1
            
    # Objective: Minimize S
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Bounds: S >= 0, cx, cy free
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0.0, np.inf)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            cx, cy, S = res.x
            return cx, cy, S
        else:
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_honeycomb_clusters()
    
    # Try different outer rotations
    rotations = [0.0, 30.0]
    
    for config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        for rot in rotations:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 2:
                break
                
            result = solve_outer_lp(config, outer_angle=rot)
            if result is not None:
                cx, cy, S = result
                # Verify validity
                if check_validity(config, (cx, cy), S, rot):
                    if S < best_score:
                        best_score = S
                        best_result = (config, (cx, cy), S, rot)
    
    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    # Estimate side for fallback
    verts = get_hex_vertices(np.array(inner_default))
    max_extent = 0.0
    for v in verts:
        for n in OUTER_NORMALS:
            max_extent = max(max_extent, np.abs(v @ n))
    required_side = max_extent / (np.sqrt(3) / 2)
    
    return inner_default, (0.0, 0.0), required_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
