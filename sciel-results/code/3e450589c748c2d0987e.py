import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions):
        """
        positions: (12, 2) array of (x, y) for unit hexagons (side=1, angle=0).
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def objective_smooth(positions):
        """
        Smooth objective function using Voronoi diagrams to estimate the required outer hexagon size.
        The idea: For each hexagon i, its 'region of influence' is its Voronoi cell.
        The outer hexagon must contain all hexagons. 
        A tight bound for the outer hexagon side length S is determined by the furthest vertex of any inner hexagon.
        However, to make it smooth and differentiable, we can approximate the constraint.
        
        Actually, a better smooth proxy:
        The distance between centers must be at least sqrt(3) for non-overlap (for aligned hexagons).
        We want to minimize the maximum extent.
        
        Let's use a penalty-based approach with smooth functions for L-BFGS-B.
        Penalty for overlap: For each pair (i, j), d = ||p_i - p_j||.
        If d < sqrt(3) + eps, penalty = (sqrt(3) + eps - d)^2.
        Objective: max_extent + lambda * sum(penalties).
        
        max_extent calculation:
        For a set of points, the minimal enclosing regular hexagon (side S, angle 0) centered at origin
        requires S >= max over all vertices v of:
        max(|vx|, |vy|*2/sqrt(3), |vx| + |vy|/sqrt(3))
        
        This max function is not differentiable everywhere, but we can smooth it using log-sum-exp or similar,
        or just rely on the fact that L-BFGS-B can handle non-smoothness to some degree if we use finite differences or careful subgradients.
        However, scipy minimize with L-BFGS-B assumes smoothness.
        
        Alternative: Minimize S directly by parameterizing S and checking if all points are inside.
        But we want to minimize S.
        
        Let's try a different smooth formulation:
        Minimize sum of squared distances from the "boundary" of a candidate outer hexagon? No.
        
        Let's stick to the penalty method for overlap, and compute the exact outer hexagon size for the current configuration.
        To make it differentiable, we can use the fact that the outer hexagon size is determined by the max of linear functions (support functions).
        f(p) = max_k (w_k . p) where w_k are normals.
        This is convex. We can minimize a smooth approximation of the max, e.g., logsumexp.
        
        Let's define the support function for a regular hexagon (side 1, angle 0) centered at origin.
        The vertices are at angles 0, 60, ..., 300.
        The normals to the sides are at 30, 90, 150, ...
        
        Actually, simpler:
        The condition for a point (x,y) to be inside a regular hexagon of side S (angle 0) is:
        |x| <= S
        |y| <= S * sqrt(3)/2
        |x + y/sqrt(3)| <= S
        
        So S_point = max( |x|, |y|*2/sqrt(3), |x + y/sqrt(3)|, |-x + y/sqrt(3)| ) 
        Wait, the third condition is |x| + |y|/sqrt(3) <= S? No.
        Let's re-verify the inequalities for a regular hexagon with side S, flat-top (vertex at (S,0)).
        Vertices: (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S,0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
        Edges:
        1. (S,0) to (S/2, S*sqrt(3)/2): Line x + y/sqrt(3) = S. Normal (1, 1/sqrt(3)).
        2. (S/2, S*sqrt(3)/2) to (-S/2, S*sqrt(3)/2): Line y = S*sqrt(3)/2. Normal (0, 1).
        3. (-S/2, S*sqrt(3)/2) to (-S,0): Line -x + y/sqrt(3) = S. Normal (-1, 1/sqrt(3)).
        4. (-S,0) to (-S/2, -S*sqrt(3)/2): Line x + y/sqrt(3) = -S. Normal (1, 1/sqrt(3)) -> -x - y/sqrt(3) <= S.
        5. (-S/2, -S*sqrt(3)/2) to (S/2, -S*sqrt(3)/2): Line y = -S*sqrt(3)/2.
        6. (S/2, -S*sqrt(3)/2) to (S,0): Line x - y/sqrt(3) = S.
        
        So the constraints are:
        1. x + y/sqrt(3) <= S
        2. y <= S*sqrt(3)/2  => y*2/sqrt(3) <= S
        3. -x + y/sqrt(3) <= S
        4. -x - y/sqrt(3) <= S
        5. -y <= S*sqrt(3)/2  => -y*2/sqrt(3) <= S
        6. x - y/sqrt(3) <= S
        
        S_point = max( x + y/sqrt(3), y*2/sqrt(3), -x + y/sqrt(3), -x - y/sqrt(3), -y*2/sqrt(3), x - y/sqrt(3) )
        
        This is a max of linear functions. It is convex.
        We can smooth the max using the log-sum-exp trick:
        S_smooth = (1/beta) * log( sum( exp(beta * L_i) ) )
        where L_i are the linear terms.
        
        We want to minimize S_smooth for the "worst" vertex of the "worst" hexagon.
        Actually, we need to contain ALL vertices of ALL hexagons.
        So for each hexagon i, we have 6 vertices.
        For each vertex v, we compute S_req(v).
        The total S is max over all vertices.
        
        To make it differentiable, we can sum the penalties for being outside a hypothetical S, but we don't know S.
        Instead, we can minimize the smooth max of S_req(v) over all v.
        
        Let V be the set of all 72 vertices (12 hexagons * 6 vertices).
        For each v in V, compute the 6 linear constraints L_k(v).
        S_v = max_k L_k(v).
        S_total = max_v S_v.
        
        We can approximate S_total by:
        S_total_smooth = (1/beta) log( sum_{v in V} exp( beta * S_v ) )
        And S_v = (1/beta) log( sum_{k=1..6} exp( beta * L_k(v) ) )
        
        So S_total_smooth = (1/beta) log( sum_{v} exp( beta * (1/beta) log(sum_k exp(beta L_k(v))) ) )
        = (1/beta) log( sum_{v} sum_k exp(beta L_k(v)) )
        = (1/beta) log( sum_{all (v,k)} exp(beta L_{v,k}) )
        
        This is a single log-sum-exp over all 432 linear terms (72 vertices * 6 constraints).
        This is smooth and differentiable.
        
        Additionally, we need to enforce non-overlap.
        Penalty for overlap:
        For each pair (i,j), d = ||p_i - p_j||.
        Required d_min = sqrt(3) (for aligned hexagons).
        Penalty = sum_{i<j} max(0, d_min - d)^2.
        This is not differentiable at d = d_min.
        We can use a smooth approximation:
        Penalty = sum_{i<j} (d_min - d + sqrt((d_min - d)^2 + eps^2))/2)^2 ?
        Or simpler: use the softplus function.
        penalty = sum_{i<j} softplus(d_min - d)^2
        softplus(x) = log(1 + exp(x))
        
        Total Objective = S_total_smooth + lambda * Penalty
        
        We will optimize this with L-BFGS-B.
        """
        if time.time() > deadline:
            return 1e10
            
        positions = positions.reshape((N, 2))
        
        # Calculate vertices for all hexagons
        # vertices: (N, 6, 2)
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
            
        # Flatten vertices to (72, 2)
        verts_flat = verts.reshape(-1, 2)
        
        # Calculate linear terms L_{v,k} for all v and k
        # L1: x + y/sqrt(3)
        # L2: y * 2/sqrt(3)
        # L3: -x + y/sqrt(3)
        # L4: -x - y/sqrt(3)
        # L5: -y * 2/sqrt(3)
        # L6: x - y/sqrt(3)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1) # (72, 6)
        
        # Log-sum-exp for S_total
        beta = 50.0 # High beta makes it closer to max
        # To avoid overflow, subtract max
        max_L = np.max(all_L)
        log_sum_exp = max_L + np.log(np.sum(np.exp(all_L - max_L)))
        S_total_smooth = log_sum_exp / beta
        
        # Overlap penalty
        # d_min = sqrt(3)
        d_min = np.sqrt(3.0)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                d = np.linalg.norm(diff)
                # Smooth penalty for d < d_min
                # Use softplus
                val = d_min - d
                # softplus(val) = log(1 + exp(val))
                # To avoid overflow in exp, use np.logaddexp(0, val)
                sp = np.logaddexp(0, val)
                penalty += sp * sp
                
        lambda_penalty = 100.0
        
        return S_total_smooth + lambda_penalty * penalty

    def objective_grad(positions):
        """
        Analytical gradient of the smooth objective.
        """
        positions = positions.reshape((N, 2))
        
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
            
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        
        beta = 50.0
        max_L = np.max(all_L)
        exp_terms = np.exp(all_L - max_L)
        sum_exp = np.sum(exp_terms)
        
        # Gradient of log(sum(exp(L))) w.r.t L is exp(L)/sum(exp(L))
        probs = exp_terms / sum_exp
        
        # Gradient w.r.t positions
        # d L_{v,k} / d p_i
        # L depends on x, y of vertex v.
        # Vertex v of hexagon i is at p_i + R(0) * v_k_local
        # v_k_local are fixed vectors.
        
        # Local vertices of unit hexagon (angle 0):
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        local_verts = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3.0)/2.0],
            [-0.5, np.sqrt(3.0)/2.0],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3.0)/2.0],
            [0.5, -np.sqrt(3.0)/2.0]
        ])
        
        # d L1 / dx = 1, d L1 / dy = 1/sqrt3
        # d L2 / dx = 0, d L2 / dy = 2/sqrt3
        # d L3 / dx = -1, d L3 / dy = 1/sqrt3
        # d L4 / dx = -1, d L4 / dy = -1/sqrt3
        # d L5 / dx = 0, d L5 / dy = -2/sqrt3
        # d L6 / dx = 1, d L6 / dy = -1/sqrt3
        
        dL_dx = np.array([1.0, 0.0, -1.0, -1.0, 0.0, 1.0])
        dL_dy = np.array([1.0/sqrt3, 2.0/sqrt3, 1.0/sqrt3, -1.0/sqrt3, -2.0/sqrt3, -1.0/sqrt3])
        
        # probs shape: (72, 6)
        # We need to sum over all v,k for each hexagon i.
        # For hexagon i, the gradient contribution to p_i is:
        # sum_{k=0..5} probs[6*i+k, k] * (dL_dx[k], dL_dy[k])
        
        grad = np.zeros((N, 2))
        for i in range(N):
            idx = slice(6*i, 6*(i+1))
            p_i_probs = probs[idx] # (6, )
            # Sum of probs * dL_dx and probs * dL_dy
            gx = np.sum(p_i_probs * dL_dx)
            gy = np.sum(p_i_probs * dL_dy)
            grad[i, 0] = gx / beta
            grad[i, 1] = gy / beta
            
        # Penalty gradient
        d_min = np.sqrt(3.0)
        lambda_penalty = 100.0
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                d = np.linalg.norm(diff)
                if d < 1e-8:
                    continue
                unit_diff = diff / d
                val = d_min - d
                # softplus derivative: sigmoid(val) = 1 / (1 + exp(-val))
                # To avoid overflow: if val > 0, sigmoid ~ 1; if val < -20, sigmoid ~ 0
                if val > 20:
                    sig = 1.0
                elif val < -20:
                    sig = 0.0
                else:
                    sig = 1.0 / (1.0 + np.exp(-val))
                
                # d (softplus(val)^2) / d d = 2 * softplus(val) * sigmoid(val)
                sp = np.logaddexp(0, val)
                d_penalty_dd = 2.0 * sp * sig
                
                # d d / d p_i = -unit_diff
                # d d / d p_j = unit_diff
                grad[i] += -lambda_penalty * d_penalty_dd * unit_diff
                grad[j] += lambda_penalty * d_penalty_dd * unit_diff
                
        return grad.flatten()

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1).flatten())
    
    # 2. Circular arrangement
    init2 = np.zeros(N * 2)
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.0
        init2[2*i] = r * np.cos(angle)
        init2[2*i+1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, N * 2)
    configs.append(init3)
    
    best_val = float('inf')
    best_params = None
    
    for init in configs:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective_smooth,
                init,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x
        except Exception as e:
            pass

    # Fallback if optimization failed or returned invalid
    if best_params is None:
        best_params = configs[0]
        
    # Final validation and exact calculation
    positions = best_params.reshape((N, 2))
    if not check_validity(positions):
        # If invalid, try to find a valid one from initial configs
        for init in configs:
            pos = init.reshape((N, 2))
            if check_validity(pos):
                best_params = init
                break
        else:
            # Last resort: spread out
            best_params = np.array([i * 3.0 for i in range(N)] + [0.0 for i in range(N)])
            
    positions = best_params.reshape((N, 2))
    
    # Calculate exact outer hexagon side length
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
    verts_flat = verts.reshape(-1, 2)
    
    x = verts_flat[:, 0]
    y = verts_flat[:, 1]
    sqrt3 = np.sqrt(3.0)
    
    L1 = x + y / sqrt3
    L2 = y * 2.0 / sqrt3
    L3 = -x + y / sqrt3
    L4 = -x - y / sqrt3
    L5 = -y * 2.0 / sqrt3
    L6 = x - y / sqrt3
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    exact_side = np.max(all_L)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(exact_side), 0.0

# EVOLVE-BLOCK-END
