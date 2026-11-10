# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_centers, outer_angle_deg):
    """
    Given inner hexagon centers (all unit hexagons, angle 0), find the minimum side length s
    of an outer regular hexagon with given angle that contains all inner hexagons.
    The outer hexagon is centered at origin.
    """
    all_vertices = []
    for cx, cy in inner_centers:
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # shape (72, 2)

    apothem_factor = np.sqrt(3) / 2
    normal_angles = np.deg2rad(outer_angle_deg) + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    dots = all_vertices @ normals.T
    max_dot = np.max(dots)
    # Ensure s is positive
    s = max(max_dot / apothem_factor, 1e-9)
    return s


def _check_separation_sat(inner_centers):
    """
    Robust Separating Axis Test for unit hexagons at angle 0.
    Returns True if all pairs are separated by > 1e-6.
    """
    n = len(inner_centers)
    # Normals for angle 0 hexagon
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Precompute vertices for a unit hexagon at origin
    verts = _hex_vertices(0.0, 0.0, 1.0, 0.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            c_i = inner_centers[i]
            c_j = inner_centers[j]
            
            separated = False
            for k in range(6):
                n_vec = normals[k]
                
                # Project vertices of hex i
                verts_i = verts + c_i
                proj_i_verts = verts_i @ n_vec
                min_i = np.min(proj_i_verts)
                max_i = np.max(proj_i_verts)
                
                # Project vertices of hex j
                verts_j = verts + c_j
                proj_j_verts = verts_j @ n_vec
                min_j = np.min(proj_j_verts)
                max_j = np.max(proj_j_verts)
                
                # Check gap
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    return True


def _optimize_positions(centers, max_iter=500, tol=1e-9):
    """
    Local optimization to minimize the max support function (outer hex side length).
    Uses a simple gradient-free approach or subgradient method.
    Since the function is convex in positions (max of linear functions), we can use
    a variant of the Frank-Wolfe or simply a subgradient descent.
    However, the constraints (non-overlap) are non-convex.
    We assume the initial configuration is feasible and we just optimize the objective.
    We must ensure we don't violate non-overlap.
    """
    n = len(centers)
    centers = centers.copy()
    
    # Precompute vertices for unit hexagon
    base_verts = _hex_vertices(0.0, 0.0, 1.0, 0.0)
    
    # Objective function: min s such that all vertices are inside outer hex.
    # s(outer_angle) = max_v (v . n) / apothem_factor
    # We want to minimize max over outer_angles? No, we choose outer_angle too.
    # The current program iterates over outer angles.
    # Here, we can jointly optimize centers and outer_angle.
    # But the objective is max_{a in [0,60)} s(a, centers).
    # This is a max-min problem.
    
    # Let's stick to the current structure: iterate over outer angles, and for each,
    # optimize centers to minimize s.
    # But the current program doesn't optimize centers for a fixed angle.
    # It just uses a fixed lattice.
    # We can add a local search step for each outer angle.
    
    # However, the non-overlap constraint is hard.
    # If we move centers, we might overlap.
    # A safe approach: only allow moves that keep the SAT check passing.
    # This is a constrained optimization.
    
    # Given the time budget and complexity, let's try a simpler refinement:
    # For a fixed outer angle, we can try to shift the entire cluster (translation)
    # to minimize s. But s is defined relative to the outer center (origin).
    # Shifting the cluster by d changes s.
    # s = max_v ((v + d) . n) / apothem = (max_v v.n + d.n) / apothem.
    # To minimize this for all 6 normals, we need to choose d such that the max
    # of (max_i v_i.n_k + d.n_k) is minimized.
    # This is a linear programming problem in d.
    
    # Let M_k = max_v v.n_k for k=0..5.
    # We want to minimize max_k (M_k + d.n_k).
    # Let t be the max. Minimize t subject to M_k + d.n_k <= t for all k.
    # This can be solved by linear programming.
    
    # However, the current code centers the cluster at the centroid.
    # If we shift the cluster, the centroid moves.
    # The outer hexagon is centered at origin.
    # So shifting the cluster is equivalent to moving the outer hexagon center.
    # But the problem statement says "outer a (3,) array (x, y, angle_deg)".
    # And "after shifting all centers by minus the outer center...".
    # So we can place the outer center anywhere.
    # The current code assumes outer center is (0,0) and centers the cluster at origin.
    # This is a valid strategy, but maybe not optimal.
    # We can optimize the outer center position.
    
    # Let's implement a function to find the optimal outer center for a given set of inner centers and outer angle.
    # Actually, the outer center can be anywhere.
    # Let c_out be the outer center.
    # The inner vertices are v_i.
    # The constraint is that v_i is inside the hexagon centered at c_out with side s and angle a.
    # This is equivalent to (v_i - c_out) being inside the hexagon centered at origin.
    # So we want to minimize s such that all (v_i - c_out) are inside.
    # s = max_k (max_i (v_i - c_out).n_k) / apothem
    # = max_k (max_i v_i.n_k - c_out.n_k) / apothem
    # Let M_k = max_i v_i.n_k.
    # s = max_k (M_k - c_out.n_k) / apothem.
    # We want to minimize this over c_out.
    # Let t = s * apothem. Minimize t subject to M_k - c_out.n_k <= t for all k.
    # This is: c_out.n_k + t >= M_k for all k.
    # This is a linear programming problem in variables c_out (2 vars) and t (1 var).
    
    # We can solve this using scipy.optimize.linprog.
    
    # However, we must also ensure that the inner hexagons don't overlap.
    # The positions of inner hexagons are fixed relative to each other (rigid body assumption? No, the current code uses a fixed lattice, so they are rigid).
    # If we treat the cluster as rigid, we only optimize c_out and outer_angle.
    # This is what the current code does implicitly (by centering at origin).
    # But we can do better by choosing the optimal c_out.
    
    # Let's try to solve the LP for c_out for each outer angle.
    
    # But wait, the inner centers are relative to the cluster centroid.
    # If we move c_out, we are effectively moving the outer hexagon.
    # The inner hexagons stay fixed in space.
    # So we just need to find the best position for the outer hexagon.
    
    # Let's modify the main loop to solve for the optimal outer center.
    
    pass # Placeholder


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580

    sqrt3 = np.sqrt(3)
    SCALE = 1.0001
    a1 = np.array([2.0 * SCALE, 0.0])
    a2 = np.array([1.0 * SCALE, sqrt3 * SCALE])
    
    arrangements = []

    # Arrangement A: 4x3
    arrA = [(i, j) for j in range(3) for i in range(4)]
    arrangements.append(arrA)

    # Arrangement B: Hex cluster 1+6+5
    arrB = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2)]
    arrangements.append(arrB)

    # Arrangement C: 3-4-3-2
    arrC = []
    for j in range(4):
        if j == 0:
            for i in range(3): arrC.append((i, j))
        elif j == 1:
            for i in range(4): arrC.append((i, j))
        elif j == 2:
            for i in range(3): arrC.append((i, j))
        elif j == 3:
            for i in range(2): arrC.append((i, j))
    arrangements.append(arrC)

    # Arrangement D: 2-4-4-2
    arrD = []
    for j in range(4):
        if j == 0:
            for i in range(2): arrD.append((i, j))
        elif j in (1, 2):
            for i in range(4): arrD.append((i, j))
        elif j == 3:
            for i in range(2): arrD.append((i, j))
    arrangements.append(arrD)

    # Arrangement E: 3x4
    arrE = [(i, j) for j in range(4) for i in range(3)]
    arrangements.append(arrE)

    # Arrangement F: 4-3-3-2
    arrF = []
    for j in range(4):
        if j == 0:
            for i in range(4): arrF.append((i, j))
        elif j in (1, 2):
            for i in range(3): arrF.append((i, j))
        elif j == 3:
            for i in range(2): arrF.append((i, j))
    arrangements.append(arrF)

    # Arrangement G: 2-3-4-3
    arrG = []
    for j in range(4):
        if j == 0:
            for i in range(2): arrG.append((i, j))
        elif j == 1:
            for i in range(3): arrG.append((i, j))
        elif j == 2:
            for i in range(4): arrG.append((i, j))
        elif j == 3:
            for i in range(3): arrG.append((i, j))
    arrangements.append(arrG)
    
    # Arrangement H: 1+6+5 variant
    arrH = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)]
    arrangements.append(arrH)

    best_s = float('inf')
    best_arrangement = None
    best_outer_angle = 0.0
    best_outer_center = np.array([0.0, 0.0])

    # We will iterate over outer angles with a finer step
    outer_angles_to_try = np.arange(0, 60, 0.5)

    for arr in arrangements:
        centers = []
        for (i, j) in arr:
            x = i * a1[0] + j * a2[0]
            y = i * a1[1] + j * a2[1]
            centers.append([x, y])
        centers = np.array(centers)

        centroid = np.mean(centers, axis=0)
        centers = centers - centroid

        if not _check_separation_sat(centers):
            continue

        # Compute all vertices of the cluster
        all_vertices = []
        for cx, cy in centers:
            verts = _hex_vertices(cx, cy, 1.0, 0.0)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices) # (72, 2)

        for oa in outer_angles_to_try:
            # Normals for the outer hexagon
            normal_angles = np.deg2rad(oa) + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60)
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
            
            # M_k = max_i v_i . n_k
            dots = all_vertices @ normals.T # (72, 6)
            M = np.max(dots, axis=0) # (6,)
            
            # We want to minimize t = max_k (M_k - c_out . n_k)
            # subject to M_k - c_out . n_k <= t for all k.
            # This is equivalent to:
            # -n_k . c_out + t >= M_k
            # Let x = [c_out_x, c_out_y, t].
            # Constraints: -n_k[0] x[0] - n_k[1] x[1] + 1 x[2] >= M_k
            # Rewrite for linprog (A_ub x <= b_ub):
            # n_k[0] x[0] + n_k[1] x[1] - 1 x[2] <= -M_k
            
            # Objective: minimize x[2] (which is t)
            # c = [0, 0, 1]
            
            # A_ub: 6x3 matrix
            A_ub = np.zeros((6, 3))
            A_ub[:, 0] = normals[:, 0]
            A_ub[:, 1] = normals[:, 1]
            A_ub[:, 2] = -1.0
            
            b_ub = -M
            
            c_obj = np.array([0.0, 0.0, 1.0])
            
            # Use scipy.optimize.linprog
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
            
            if res.success:
                t_opt = res.x[2]
                s_opt = t_opt / (np.sqrt(3) / 2)
                if s_opt < best_s:
                    best_s = s_opt
                    best_arrangement = centers.copy()
                    best_outer_angle = oa
                    best_outer_center = np.array([res.x[0], res.x[1]])
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    if best_arrangement is None:
        # Fallback
        centers = []
        for (i, j) in arrangements[0]:
            x = i * a1[0] + j * a2[0]
            y = i * a1[1] + j * a2[1]
            centers.append([x, y])
        centers = np.array(centers)
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        best_arrangement = centers
        best_s = _min_containing_hex_side(centers, 0.0)
        best_outer_angle = 0.0
        best_outer_center = np.array([0.0, 0.0])

    inner_hex_data = np.zeros((12, 3))
    for idx in range(12):
        inner_hex_data[idx, 0] = best_arrangement[idx, 0]
        inner_hex_data[idx, 1] = best_arrangement[idx, 1]
        inner_hex_data[idx, 2] = 0.0

    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], float(best_outer_angle)])

    return inner_hex_data, outer_hex_data, float(best_s)


# EVOLVE-BLOCK-END
