import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def compute_outer_side_and_center(positions, angles_deg):
        """
        Compute the minimum side length and center of a regular hexagon
        (at angle 0, flat-top) containing all unit hexagons.
        Returns (side_length, center_x, center_y).
        """
        # For a regular hexagon centered at (cx, cy) with side length s and angle 0 (flat-top),
        # the support functions in the 6 normal directions are:
        # n0 = (0, 1) -> cy + s
        # n1 = (sqrt(3)/2, -1/2) -> cx*sqrt(3)/2 - cy/2 + s
        # n2 = (sqrt(3)/2, 1/2) -> cx*sqrt(3)/2 + cy/2 + s
        # n3 = (0, -1) -> -cy + s
        # n4 = (-sqrt(3)/2, -1/2) -> -cx*sqrt(3)/2 - cy/2 + s
        # n5 = (-sqrt(3)/2, 1/2) -> -cx*sqrt(3)/2 + cy/2 + s
        #
        # We need s >= max_i (dot(p_i, n_j) - dot(c, n_j)) for all vertices p_i and all 6 directions j
        # => s >= max over all (p_i, n_j) of (p_i · n_j) - max over all (q_j) of (c · n_j) ... wait
        #
        # The condition is: for all vertices v of all hexagons, and for all 6 outward normals n_j of the outer hex,
        # v · n_j <= c · n_j + s
        # => s >= v · n_j - c · n_j = (v - c) · n_j
        # => s >= max_{v, j} (v - c) · n_j
        #
        # We want to minimize s over c. For a fixed s, the feasible region for c is the intersection of half-planes.
        # But we can do better: the minimum s is achieved when the outer hexagon is the smallest one containing all points.
        #
        # Actually, let's use the fact that for a given set of points, the minimum enclosing regular hexagon (fixed orientation)
        # can be found by considering the 6 support functions.
        #
        # Let's collect all 72 vertices (12 hexagons * 6 vertices each).
        # For each of the 6 normal directions, compute the max projection of all vertices.
        # Let M_j = max_v (v · n_j)
        # Then we need c · n_j + s >= M_j for all j
        # => s >= M_j - c · n_j
        # => s >= max_j (M_j - c · n_j)
        #
        # We want to minimize s(c) = max_j (M_j - c · n_j)
        # This is a convex optimization in c (minimize max of linear functions).
        #
        # Let's set up the problem: minimize s subject to s + c · n_j >= M_j for all j.
        # Variables: c = (cx, cy), s
        # Constraints: s + cx*n_jx + cy*n_jy - M_j >= 0 for j=0..5

        # Get all vertices
        all_vertices = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            all_vertices.append(verts)
        all_verts = np.vstack(all_vertices)  # shape (72, 2)

        # Normal directions for flat-top hexagon (angle 0)
        normals = np.array([
            [0.0, 1.0],
            [SQRT3 / 2.0, -0.5],
            [SQRT3 / 2.0, 0.5],
            [0.0, -1.0],
            [-SQRT3 / 2.0, -0.5],
            [-SQRT3 / 2.0, 0.5]
        ])  # shape (6, 2)

        # M_j = max over all vertices of dot(v, n_j)
        # all_verts @ normals.T gives shape (72, 6)
        projections = all_verts @ normals.T
        M = np.max(projections, axis=0)  # shape (6,)

        # Now solve: minimize s subject to s + c · n_j >= M_j
        # This is a linear program!
        # Variables: cx, cy, s (3 variables)
        # Constraints: -cx*n_jx - cy*n_jy - s <= -M_j for j=0..5
        #              s >= 0
        #
        # Let's use scipy.optimize.linprog or just solve it directly.
        # Actually, since it's a small LP, let's use scipy.optimize.linprog.
        
        from scipy.optimize import linprog
        
        # c: minimize f^T x where x = [cx, cy, s]
        # f = [0, 0, 1] (minimize s)
        # A_ub * x <= b_ub
        # For each j: -n_jx * cx - n_jy * cy - s <= -M_j
        A_ub = -np.hstack([normals, -np.ones((6, 1))])  # shape (6, 3)
        b_ub = -M  # shape (6,)
        
        # s >= 0 => -s <= 0
        A_ub = np.vstack([A_ub, [[0, 0, -1]]])
        b_ub = np.append(b_ub, 0)
        
        # Bounds: cx, cy are free, s >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c=np.array([0, 0, 1.0]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            cx, cy, s = res.x
            return s, cx, cy
        else:
            # Fallback: use origin and compute s
            s = np.max(M)
            return s, 0.0, 0.0

    def check_validity(positions, angles_deg):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, angles_deg[i])
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    # Simulated Annealing with gradient-based refinement
    def simulated_annealing(init_positions, init_angles, max_iters=80000, T_start=3.0, T_end=0.005):
        current_pos = init_positions.copy()
        current_angles = init_angles.copy()
        
        # Compute initial side
        current_side, _, _ = compute_outer_side_and_center(current_pos, current_angles)
        
        if not check_validity(current_pos, current_angles):
            # Try to fix initial config by spreading out
            for scale in [1.5, 2.0, 2.5, 3.0]:
                current_pos *= scale
                current_side, _, _ = compute_outer_side_and_center(current_pos, current_angles)
                if check_validity(current_pos, current_angles):
                    break
            else:
                return init_positions, init_angles, init_positions, init_angles, float('inf')
        
        best_pos = current_pos.copy()
        best_angles = current_angles.copy()
        best_side = current_side
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Random perturbation: move one hexagon and/or rotate it
            idx = np.random.randint(N)
            new_pos = current_pos.copy()
            new_angles = current_angles.copy()
            
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            new_pos[idx] += np.array([dx, dy])
            
            # Sometimes rotate
            if np.random.random() < 0.3:
                dangle = np.random.normal(0, 10.0)
                new_angles[idx] += dangle
            
            # Check validity of new config
            if not check_validity(new_pos, new_angles):
                continue
            
            new_side, _, _ = compute_outer_side_and_center(new_pos, new_angles)
            delta = new_side - current_side
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-9)):
                current_pos = new_pos
                current_angles = new_angles
                current_side = new_side
                if current_side < best_side:
                    best_pos = current_pos.copy()
                    best_angles = current_angles.copy()
                    best_side = current_side
        
        return best_pos, best_angles, current_pos, current_angles, best_side

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1_pos = []
    init1_angles = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init1_pos.append([x, y])
        init1_angles.append(0.0)
    configs.append((np.array(init1_pos), np.array(init1_angles)))
    
    # 2. Circular arrangement
    init2_pos = np.zeros((N, 2))
    init2_angles = np.zeros(N)
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2_pos[i, 0] = r * np.cos(angle)
        init2_pos[i, 1] = r * np.sin(angle)
        init2_angles[i] = np.degrees(angle)  # Rotate to face outward
    configs.append((init2_pos, init2_angles))
    
    # 3. Random
    np.random.seed(42)
    init3_pos = np.random.uniform(-4, 4, (N, 2))
    init3_angles = np.random.uniform(0, 60, N)
    configs.append((init3_pos, init3_angles))
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    for init_pos, init_angles in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        if not check_validity(init_pos, init_angles):
            valid = False
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled_pos = init_pos * scale
                if check_validity(scaled_pos, init_angles):
                    init_pos = scaled_pos
                    valid = True
                    break
            if not valid:
                continue
        
        result_pos, result_angles, current_pos, current_angles, result_side = simulated_annealing(init_pos, init_angles)
        
        if result_side < best_side:
            best_side = result_side
            best_positions = result_pos
            best_angles = result_angles
    
    # Fallback
    if best_positions is None:
        best_positions = configs[0][0]
        best_angles = configs[0][1]
        best_side, _, _ = compute_outer_side_and_center(best_positions, best_angles)
    
    # Final validation
    if not check_validity(best_positions, best_angles):
        # Try to find a valid config
        for init_pos, init_angles in configs:
            pos = init_pos.copy()
            angles = init_angles.copy()
            for scale in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = pos * scale
                if check_validity(scaled, angles):
                    best_positions = scaled
                    best_angles = angles.copy()
                    best_side, _, _ = compute_outer_side_and_center(best_positions, best_angles)
                    break
            if check_validity(best_positions, best_angles):
                break
        else:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_angles = np.zeros(N)
            best_side, _, _ = compute_outer_side_and_center(best_positions, best_angles)
    
    # Calculate final exact side and center
    final_side, center_x, center_y = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    return inner, (float(center_x), float(center_y)), float(final_side), 0.0
