import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute base vertices for a unit hexagon (flat-top)
# Vertices at angles 0, 60, 120, 180, 240, 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_vertices(pos, ang):
    """
    Get vertices for a hexagon at pos (2,) with angle ang (scalar).
    Returns (6, 2) array.
    """
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (R @ BASE_VERTS.T).T + pos

def compute_outer_side_and_grad(pos_2d, angles):
    """
    Compute the side length of the smallest regular hexagon (centered at origin, angle 0)
    containing all hexagons, and its gradient w.r.t positions and angles.
    
    pos_2d: (N, 2)
    angles: (N,)
    """
    # The outer hexagon is defined by 6 half-spaces.
    # For a regular hexagon of side s centered at origin with "pointy-top" or "flat-top"?
    # The task says "regular hexagon". The seed used specific linear forms.
    # Let's stick to the seed's definition of the outer hexagon to be consistent with scoring.
    # Seed used:
    # L1 = x + y / SQRT3
    # L2 = y * 2.0 / SQRT3
    # L3 = -x + y / SQRT3
    # L4 = -x - y / SQRT3
    # L5 = -y * 2.0 / SQRT3
    # L6 = x - y / SQRT3
    # Side length = max(L_i)
    
    # We need to minimize S = max_i L_i(x, y)
    # This is a convex function of (x, y) for fixed hexagons, but non-convex in angles.
    # We will use a smooth approximation: S_smooth = logsumexp(L_i) / log_base? 
    # Or just use the max and handle subgradients?
    # Better: Minimize S such that L_i <= S for all i.
    # In the penalty method, we can just add S to the objective and penalize violations?
    # No, S is the objective.
    
    # To make it differentiable, we can use the "soft max" approximation:
    # S_approx = (1/k) * log( sum( exp(k * L_i) ) )
    # As k -> inf, this approaches max(L_i).
    
    # Let's compute all vertices for all hexagons
    all_verts = np.zeros((N * 6, 2))
    grads_pos = np.zeros((N, 2))
    grads_ang = np.zeros(N)
    
    # We will accumulate gradients for the max constraint.
    # Since S = max(L), the gradient is the gradient of the active L.
    # In the soft max approach, the gradient is a weighted average.
    
    k = 50.0 # Sharpness of soft max
    
    # L forms matrix
    # L1: [1, 1/SQRT3]
    # L2: [0, 2/SQRT3]
    # L3: [-1, 1/SQRT3]
    # L4: [-1, -1/SQRT3]
    # L5: [0, -2/SQRT3]
    # L6: [1, -1/SQRT3]
    
    normals = np.array([
        [1.0, 1.0/SQRT3],
        [0.0, 2.0/SQRT3],
        [-1.0, 1.0/SQRT3],
        [-1.0, -1.0/SQRT3],
        [0.0, -2.0/SQRT3],
        [1.0, -1.0/SQRT3]
    ])
    
    # For each hexagon, we project its vertices onto these normals.
    # The max projection for a given normal direction determines the constraint.
    # For a convex polygon, max projection is max over vertices.
    
    # To get gradients, we need to know which vertex is active for which normal.
    # This is discrete. However, for the outer hexagon side length,
    # the "active" constraint is the one with the max value among all 6*6*6 combinations?
    # Actually, for a fixed normal n, the support function h(n) = max_{v in union} n . v.
    # h(n) is convex in the set positions.
    # S = max_{n in N_outer} h(n).
    
    # Let's compute h(n_j) for j=1..6.
    # h(n_j) = max_{i=1..N} max_{k=1..6} n_j . (R(a_i) v_k + p_i)
    
    # Gradient of h(n_j) w.r.t p_i is n_j if hexagon i has the active vertex for n_j.
    # Gradient w.r.t a_i is n_j . (d/d a_i (R(a_i) v_k)) = n_j . (v_k rotated 90 deg * sign?)
    # d/da (R(a)v) = R'(a)v = [[-sin, -cos], [cos, -sin]] v = perp(R(a)v)
    # where perp(x,y) = (-y, x).
    # So grad_a h(n_j) = n_j . perp(vertex_k) if vertex k of hex i is active.
    
    vals = np.zeros(6)
    active_hex = np.zeros(6, dtype=int)
    active_vert = np.zeros(6, dtype=int)
    
    for j in range(6):
        n = normals[j]
        max_val = -np.inf
        max_i = 0
        max_k = 0
        for i in range(N):
            verts = get_vertices(pos_2d[i], angles[i])
            projs = verts.dot(n)
            idx = np.argmax(projs)
            if projs[idx] > max_val:
                max_val = projs[idx]
                max_i = i
                max_k = idx
        vals[j] = max_val
        active_hex[j] = max_i
        active_vert[j] = max_k
        
    # Soft max over vals
    # S = soft_max(vals)
    # grad_S = softmax(vals) @ vals? No.
    # d/du logsumexp(u) = softmax(u)
    # d/du (1/k logsumexp(k u)) = softmax(k u)
    
    # Let's just use the hard max for the objective value, but for gradient,
    # we can use the soft max gradient to smooth the optimization landscape.
    # Actually, L-BFGS-B works well with non-smooth objectives if we provide a subgradient.
    # The subgradient of max is the average of the gradients of the active terms.
    # If we use soft max, we get a smooth gradient.
    
    k_sm = 20.0
    exp_vals = np.exp(k_sm * (vals - np.max(vals)))
    weights = exp_vals / np.sum(exp_vals)
    
    S_val = np.max(vals) # Use hard max for actual score
    # Or S_val = np.log(np.sum(exp_vals)) / k_sm + np.max(vals)
    
    # Gradient accumulation
    # grad_p = sum_j weights[j] * n_j * (1 if i == active_hex[j] else 0)
    # grad_a = sum_j weights[j] * (n_j . perp(vert_k)) * (1 if i == active_hex[j] else 0)
    
    for j in range(6):
        w = weights[j]
        if w < 1e-6: continue
        i = active_hex[j]
        k_idx = active_vert[j]
        
        n = normals[j]
        grads_pos[i] += w * n
        
        # Get the active vertex vector
        verts_i = get_vertices(pos_2d[i], angles[i])
        v = verts_i[k_idx]
        # perp(v) = (-v[1], v[0])
        perp_v = np.array([-v[1], v[0]])
        grads_ang[i] += w * np.dot(n, perp_v)
        
    return S_val, grads_pos, grads_ang

def compute_overlap_penalty_and_grad(pos_2d, angles):
    """
    Compute a smooth penalty for overlaps.
    Penalty = sum_{i<j} max(0, 1 - dist(i,j))^2 ?
    Or use the support function of Minkowski difference.
    
    For two convex sets A, B, they are disjoint iff 0 not in A-B.
    Distance between A and B is min_{a in A, b in B} ||a-b||.
    If we use the support function h_{A-B}(u) = h_A(u) + h_B(-u).
    The distance is roughly related to min_u h_{A-B}(u)? No.
    
    A simpler smooth approximation for "non-overlap" of two polygons P, Q:
    Penalty = max(0, max_{u in U} (h_P(u) + h_Q(-u) - dist_min))^2 ?
    
    Let's use a simpler heuristic:
    If the distance between centers is less than the sum of circumradii,
    there is potential overlap.
    But we need exactness.
    
    Alternative: Use the "witness point" method?
    
    Let's stick to a robust penalty:
    For each pair (i,j), compute the minimum distance between their boundaries?
    That's expensive.
    
    Let's use the fact that for regular hexagons, if we fix the relative angle,
    the Minkowski difference is a convex polygon.
    0 is outside A-B iff there exists a separating axis.
    The separating axis is one of the edge normals of A or B.
    
    Let D = A - B.
    If 0 is in D, they overlap.
    The distance of 0 to D is min_{d in D} ||d||.
    This is a convex optimization problem.
    
    For the penalty, we can use the squared distance of the origin to the Minkowski difference polygon.
    If 0 is outside, dist > 0. If inside, dist = 0.
    We want to penalize when dist is small (i.e., they are close or overlapping).
    Actually, we want to penalize when they OVERLAP.
    So we need a measure of "how much" they overlap.
    If 0 is inside D, the overlap is positive.
    
    Let's use the following:
    For each pair, check the 12 separating axes (6 from i, 6 from j).
    For each axis, project both hexagons.
    If projections overlap, the axis is not separating.
    If ALL axes show overlap, they are intersecting.
    
    To make it smooth, we can sum the "overlap depths" on the axes?
    Overlap depth on axis n = min(max_A, max_B) - max(min_A, min_B).
    If > 0, they overlap on this axis.
    If any axis has < 0, they are separated.
    
    So, Separation = max_{axis} (-Overlap_Depth(axis)).
    If Separation > 0, they are disjoint.
    If Separation <= 0, they overlap.
    
    Penalty = max(0, -Separation)^2.
    
    This is smooth enough for gradient descent if we use soft-max for the max.
    """
    penalty = 0.0
    grads_pos = np.zeros_like(pos_2d)
    grads_ang = np.zeros_like(angles)
    
    # Precompute normals for each hexagon
    # Normal for edge k of hexagon i
    # Edge vector e_k = v_{k+1} - v_k
    # Normal n_k = perp(e_k) / ||e_k||
    
    for i in range(N):
        verts_i = get_vertices(pos_2d[i], angles[i])
        # Normals for i
        norms_i = np.zeros((6, 2))
        for k in range(6):
            e = verts_i[(k+1)%6] - verts_i[k]
            n = np.array([-e[1], e[0]])
            n /= np.linalg.norm(n)
            norms_i[k] = n
            
    for j in range(N):
        if i == j: continue
        verts_j = get_vertices(pos_2d[j], angles[j])
        norms_j = np.zeros((6, 2))
        for k in range(6):
            e = verts_j[(k+1)%6] - verts_j[k]
            n = np.array([-e[1], e[0]])
            n /= np.linalg.norm(n)
            norms_j[k] = n
            
        # Check all 12 axes
        # We want max(-overlap_depth)
        # overlap_depth(n) = min(max_i, max_j) - max(min_i, min_j)
        
        depths = np.zeros(12)
        active_axis_idx = np.zeros(12, dtype=int) # 0-5 for i, 6-11 for j
        active_i = np.zeros(12, dtype=int) # index of vertex in i
        active_j = np.zeros(12, dtype=int) # index of vertex in j
        
        # Project i
        proj_i = verts_i.dot(norms_i.T) # (6, 6) ? No.
        # We need projection of verts_i onto norms_j too.
        
        # Let's just loop for clarity and gradient tracking
        max_neg_depth = -np.inf
        best_axis_type = 0 # 0 for i, 1 for j
        best_axis_idx = 0
        best_vi_idx = 0
        best_vj_idx = 0
        
        # Axes from i
        for k in range(6):
            n = norms_i[k]
            min_i, max_i = np.min(verts_i.dot(n)), np.max(verts_i.dot(n))
            min_j, max_j = np.min(verts_j.dot(n)), np.max(verts_j.dot(n))
            
            depth = min(max_i, max_j) - max(min_i, min_j)
            # We want to maximize -depth (separation)
            sep = -depth
            if sep > max_neg_depth:
                max_neg_depth = sep
                best_axis_type = 0
                best_axis_idx = k
                # Find active vertices
                # min_i is from some vertex, max_i from some vertex
                # For gradient, we need to know which vertices contribute to min/max
                vi_min_idx = np.argmin(verts_i.dot(n))
                vi_max_idx = np.argmax(verts_i.dot(n))
                vj_min_idx = np.argmin(verts_j.dot(n))
                vj_max_idx = np.argmax(verts_j.dot(n))
                
                # The "active" constraint for the max separation is this axis.
                # The gradient of sep w.r.t pos_i is:
                # d/dp_i ( - (min(max_i, max_j) - max(min_i, min_j)) )
                # This is tricky.
                # Let's assume the active case is where i is the "limiting" factor.
                # Actually, let's just use the subgradient of the max function.
                # If this axis gives the max separation, its gradient is used.
                
                # Gradient of sep = -depth
                # depth = min(max_i, max_j) - max(min_i, min_j)
                
                # Case 1: max_i < max_j and min_i > min_j
                # depth = max_i - min_i
                # sep = min_i - max_i
                # grad_p_i = n (from min_i) - n (from max_i) = 0?
                # Wait. min_i = min_k n . (v_k + p_i) = n . p_i + min_k n . v_k
                # max_i = n . p_i + max_k n . v_k
                # sep = (n . p_i + c_min) - (n . p_i + c_max) = c_min - c_max
                # Independent of p_i!
                # This makes sense: translation doesn't change the width of the hexagon.
                # But it changes the position relative to the other hexagon.
                
                # Let's re-evaluate.
                # sep = - (min(max_i, max_j) - max(min_i, min_j))
                
                # If max_i < max_j: min(max_i, max_j) = max_i
                # If min_i > min_j: max(min_i, min_j) = min_i
                # Then sep = - (max_i - min_i) = min_i - max_i.
                # This is the negative width of i. Constant.
                
                # If max_i < max_j and min_i < min_j:
                # sep = - (max_i - min_j) = min_j - max_i.
                # min_j = n . p_j + c_j_min
                # max_i = n . p_i + c_i_max
                # sep = n . p_j - n . p_i + const
                # grad_p_i = -n
                # grad_p_j = n
                
                # We need to determine which case we are in.
                
                if max_i < max_j and min_i < min_j:
                    # sep = min_j - max_i
                    grad_i = -n
                    grad_j = n
                    # Angle gradients
                    # d/d a_i max_i = n . perp(v_i_max)
                    # d/d a_j min_j = n . perp(v_j_min)
                    grad_ai = -n.dot(np.array([-verts_i[vi_max_idx,1], verts_i[vi_max_idx,0]]))
                    grad_aj = n.dot(np.array([-verts_j[vj_min_idx,1], verts_j[vj_min_idx,0]]))
                    
                elif max_i > max_j and min_i > min_j:
                    # sep = min_i - max_j
                    grad_i = n
                    grad_j = -n
                    grad_ai = n.dot(np.array([-verts_i[vi_min_idx,1], verts_i[vi_min_idx,0]]))
                    grad_aj = -n.dot(np.array([-verts_j[vj_max_idx,1], verts_j[vj_max_idx,0]]))
                    
                elif max_i < max_j and min_i > min_j:
                    # sep = min_i - max_i (Width of i, constant)
                    grad_i = np.zeros(2)
                    grad_j = np.zeros(2)
                    grad_ai = 0
                    grad_aj = 0
                    
                else: # max_i > max_j and min_i < min_j
                    # sep = min_j - max_j (Width of j, constant)
                    grad_i = np.zeros(2)
                    grad_j = np.zeros(2)
                    grad_ai = 0
                    grad_aj = 0
                    
                # Store these for the "best" axis
                if sep > max_neg_depth:
                    max_neg_depth = sep
                    best_grad_i = grad_i
                    best_grad_j = grad_j
                    best_grad_ai = grad_ai
                    best_grad_aj = grad_aj
                    best_axis_type = 0
                    best_axis_idx = k

        # Axes from j
        for k in range(6):
            n = norms_j[k]
            min_i, max_i = np.min(verts_i.dot(n)), np.max(verts_i.dot(n))
            min_j, max_j = np.min(verts_j.dot(n)), np.max(verts_j.dot(n))
            
            depth = min(max_i, max_j) - max(min_i, min_j)
            sep = -depth
            
            if sep > max_neg_depth:
                # Similar logic
                vi_min_idx = np.argmin(verts_i.dot(n))
                vi_max_idx = np.argmax(verts_i.dot(n))
                vj_min_idx = np.argmin(verts_j.dot(n))
                vj_max_idx = np.argmax(verts_j.dot(n))
                
                if max_i < max_j and min_i < min_j:
                    # sep = min_j - max_i
                    grad_i = -n
                    grad_j = n
                    grad_ai = -n.dot(np.array([-verts_i[vi_max_idx,1], verts_i[vi_max_idx,0]]))
                    grad_aj = n.dot(np.array([-verts_j[vj_min_idx,1], verts_j[vj_min_idx,0]]))
                    
                elif max_i > max_j and min_i > min_j:
                    # sep = min_i - max_j
                    grad_i = n
                    grad_j = -n
                    grad_ai = n.dot(np.array([-verts_i[vi_min_idx,1], verts_i[vi_min_idx,0]]))
                    grad_aj = -n.dot(np.array([-verts_j[vj_max_idx,1], verts_j[vj_max_idx,0]]))
                    
                elif max_i < max_j and min_i > min_j:
                    # Width of i
                    grad_i = np.zeros(2)
                    grad_j = np.zeros(2)
                    grad_ai = 0
                    grad_aj = 0
                    
                else:
                    # Width of j
                    grad_i = np.zeros(2)
                    grad_j = np.zeros(2)
                    grad_ai = 0
                    grad_aj = 0
                    
                max_neg_depth = sep
                best_grad_i = grad_i
                best_grad_j = grad_j
                best_grad_ai = grad_ai
                best_grad_aj = grad_aj
                best_axis_type = 1
                best_axis_idx = k
                
        # Apply penalty if overlapping (sep <= 0)
        if max_neg_depth < 0:
            # Penalty = sep^2
            # Grad = 2 * sep * grad_sep
            factor = 2 * max_neg_depth
            penalty += max_neg_depth * max_neg_depth
            grads_pos[i] += factor * best_grad_i
            grads_pos[j] += factor * best_grad_j
            grads_ang[i] += factor * best_grad_ai
            grads_ang[j] += factor * best_grad_aj
            
    return penalty, grads_pos, grads_ang

def objective_and_grad(x):
    """
    x: (N*3,) array [x1, y1, a1, x2, y2, a2, ...]
    """
    pos_2d = x[0::3].reshape(N, 2)
    angles = x[1::3].reshape(N)
    
    S, grad_S_pos, grad_S_ang = compute_outer_side_and_grad(pos_2d, angles)
    P, grad_P_pos, grad_P_ang = compute_overlap_penalty_and_grad(pos_2d, angles)
    
    # Weight for penalty
    # We want to keep P small.
    # If P is large, we are overlapping.
    # We can use a large weight if P > 0, else small.
    weight = 100.0 if P > 1e-3 else 1.0
    
    val = S + weight * P
    grad = np.zeros_like(x)
    
    grad[0::3] = grad_S_pos[:, 0] + weight * grad_P_pos[:, 0]
    grad[1::3] = grad_S_pos[:, 1] + weight * grad_P_pos[:, 1]
    grad[2::3] = grad_S_ang + weight * grad_P_ang
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial configuration: Hexagonal lattice
    def get_initial_config():
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        pos = []
        ang = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pos.append([x, y])
            ang.append(0.0)
        return np.array(pos), np.array(ang)
    
    best_side = np.inf
    best_x = None
    
    # Multi-restart
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        pos, ang = get_initial_config()
        
        # Add noise
        if r > 0:
            pos += np.random.normal(0, 0.5, pos.shape)
            ang += np.random.uniform(-10, 10, N)
        
        x0 = np.zeros(N * 3)
        x0[0::3] = pos[:, 0]
        x0[1::3] = pos[:, 1]
        x0[2::3] = ang
        
        # Optimize
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-5})
            
            x_opt = res.x
            pos_opt = x_opt[0::3].reshape(N, 2)
            ang_opt = x_opt[1::3].reshape(N)
            
            # Evaluate true side length (hard max, no penalty)
            S_true, _, _ = compute_outer_side_and_grad(pos_opt, ang_opt)
            
            # Check validity (optional, but good to know)
            # If penalty is 0, it should be valid.
            P_true, _, _ = compute_overlap_penalty_and_grad(pos_opt, ang_opt)
            
            if P_true < 1e-2: # Almost valid
                if S_true < best_side:
                    best_side = S_true
                    best_x = x_opt
        except Exception as e:
            pass
            
    # Final fallback if optimization failed
    if best_x is None:
        pos, ang = get_initial_config()
        x0 = np.zeros(N * 3)
        x0[0::3] = pos[:, 0]
        x0[1::3] = pos[:, 1]
        x0[2::3] = ang
        best_x = x0
        best_side, _, _ = compute_outer_side_and_grad(pos, ang)
        
    pos_final = best_x[0::3].reshape(N, 2)
    ang_final = best_x[1::3].reshape(N)
    
    # Return format
    inner = []
    for i in range(N):
        inner.append([float(pos_final[i, 0]), float(pos_final[i, 1]), float(ang_final[i])])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
