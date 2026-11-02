import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute vertex angles for unit hexagon (flat-top orientation, angle 0)
    # Vertices at angles 0, 60, ..., 300
    vertex_angles = np.radians(np.arange(0, 360, 60))
    cos_va = np.cos(vertex_angles)
    sin_va = np.sin(vertex_angles)
    
    # Direction vectors for the 6 support functions of the outer hexagon
    # The outer hexagon (side S, angle 0) has support functions h(u) = S * ||u||_inf in a rotated frame?
    # Actually, for a regular hexagon of side S centered at origin with vertices at angles 0, 60...
    # The support function h(u) in direction u is S * max(|u_x|, |u_x/SQRT3 + u_y/3| ...)? No.
    # Let's use the standard 6 linear inequalities.
    # A point (x,y) is inside a hexagon of side S (angle 0) iff:
    # 1. x <= S
    # 2. x + y/SQRT3 <= S
    # 3. -x + y/SQRT3 <= S
    # 4. -x <= S
    # 5. -x - y/SQRT3 <= S
    # 6. x - y/SQRT3 <= S
    # Wait, let's verify.
    # Vertex (S, 0). Normal (1, 0). h = S.
    # Vertex (S/2, S*SQRT3/2). Normal (1/2, SQRT3/2) -> (1, SQRT3). 
    # Dot product: x*1 + y*SQRT3 = S/2 + 3S/2 = 2S? No.
    # The side length is S. The distance from center to side (apothem) is S * SQRT3 / 2.
    # The distance from center to vertex is S.
    # For a regular hexagon, the support function in the direction of a vertex normal is S.
    # The 6 outward normals are at angles 0, 60, 120, 180, 240, 300.
    # n_k = (cos(k*pi/3), sin(k*pi/3)).
    # h(n_k) = S.
    # So the constraint is n_k . p <= S.
    
    normals = np.array([
        [1, 0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1, 0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])
    
    # To contain all hexagons i, we need for all k: max_j (n_k . (c_i + v_j)) <= t
    # where c_i is center of hex i, v_j is vertex of unit hex, t is outer side length.
    # max_j (n_k . c_i + n_k . v_j) = n_k . c_i + max_j (n_k . v_j)
    # Let R_k = max_j (n_k . v_j) for a unit hexagon.
    # Since the unit hexagon is regular and centered at 0, R_k is the support of the unit hex in direction n_k.
    # For a unit hexagon (side 1), the support in the direction of its own normal (which is also a vertex direction for the dual?)
    # Wait. The normals n_k are the directions perpendicular to the sides of the OUTER hexagon.
    # The inner hexagons can be rotated. But in the current seed, they are fixed at angle 0.
    # The task allows rotation. However, for a regular hexagon, rotating by 60 degrees is equivalent.
    # Rotating by 30 degrees makes it "pointy-top".
    # If we allow rotation, the support R_k(theta) changes.
    # To minimize the outer side, we should align the inner hexagons such that their support is minimized in the critical directions.
    # Actually, the support of a regular hexagon of side 1 in ANY direction is between 1 (vertex direction) and SQRT3/2 (side direction)?
    # No. The circumradius is 1. The inradius is SQRT3/2.
    # The support function h(u) for a regular hexagon of circumradius R is R * (cos(pi/6) / cos(pi/6 - theta))?
    # Actually, for a regular hexagon, the support is constant in 6 sectors? No.
    # Let's stick to the seed's assumption: inner hexagons are NOT rotated (angle 0).
    # The seed program fixes angle to 0.0 in the return.
    # If we allow rotation, we add 12 more variables.
    # Let's first try to improve the translation optimization with the better smooth objective.
    # If we assume inner hexagons are fixed at angle 0:
    # R_k = max_{j} (n_k . v_j).
    # v_j are vertices of unit hex (angle 0).
    
    verts_unit = np.column_stack([cos_va, sin_va])
    # Compute R_k for fixed angle 0
    # R_k = max_j (n_k . v_j)
    # Since n_k and v_j are symmetric, let's compute.
    # n_0 = (1, 0). v_0 = (1, 0). Dot = 1. v_1 = (0.5, SQRT3/2). Dot = 0.5. Max is 1.
    # n_1 = (0.5, SQRT3/2). v_0 = (1, 0). Dot = 0.5. v_1 = (0.5, SQRT3/2). Dot = 0.25 + 3/4 = 1. Max is 1.
    # It turns out for a regular hexagon, the support in the direction of its own normals is always 1 (circumradius).
    # Wait, the normals of the OUTER hexagon are the same as the vertex directions of the INNER hexagon (if both angle 0).
    # So R_k = 1 for all k.
    
    R_k = np.ones(6)
    
    def objective(params):
        # params: [c_0x, c_0y, ..., c_11x, c_11y, t]
        # t is the outer side length (auxiliary variable)
        centers = params[:2*N].reshape(N, 2)
        t = params[2*N]
        
        # 1. Outer constraint: t >= max_{i, k} (n_k . c_i + R_k)
        # We use a smooth max (Log-Sum-Exp) over all (i, k) pairs.
        # There are N * 6 = 72 terms.
        # terms[i, k] = n_k . c_i + R_k
        
        # Vectorized computation of n_k . c_i
        # c_i is (N, 2). normals is (6, 2).
        # We want (N, 6) matrix of dots.
        dots = centers @ normals.T  # (N, 6)
        # Add R_k (broadcast)
        support_vals = dots + R_k  # (N, 6)
        
        # Smooth max
        # LSE(x) = beta * log(sum(exp(x/beta)))
        # As beta -> 0, LSE -> max(x)
        # We want to minimize t, so we penalize if t < smooth_max.
        # Or simply minimize t + penalty * (smooth_max - t)^2?
        # Better: Minimize t + lambda * sum(softplus(support_vals - t))
        # softplus(z) = log(1 + exp(z)) approximates max(0, z).
        # This enforces t >= support_vals softly.
        
        beta = 0.1  # Smoothness parameter. Smaller is sharper.
        
        # Calculate softplus(support_vals - t)
        # To avoid overflow, use stable log-sum-exp or just clip.
        # support_vals - t can be large.
        # softplus(x) = log(1 + exp(x))
        # If x > 30, softplus(x) ~ x.
        # If x < -30, softplus(x) ~ 0.
        
        diff = support_vals - t
        # Stable softplus
        softplus_vals = np.where(diff > 30, diff, np.log1p(np.exp(np.clip(diff, -500, 30))))
        
        outer_penalty = np.sum(softplus_vals)
        
        # 2. Disjointness constraint
        # We need a smooth approximation of the distance between two hexagons.
        # The Minkowski difference of two convex polygons is a convex polygon.
        # The distance between two sets A and B is 0 if they intersect, > 0 if disjoint.
        # We want to penalize intersection.
        # A smooth proxy for "intersection" is the maximum over all separating hyperplanes of the gap.
        # If gap > 0, they are disjoint. If gap < 0, they intersect.
        # We want to minimize max(0, -gap) or similar.
        # 
        # For two unit hexagons (angle 0) at centers c_i and c_j:
        # The separating axes are the 12 edge normals of the two hexagons.
        # Since they are identical and parallel, the 6 normals of one are sufficient?
        # Yes, for two congruent parallel hexagons, the 6 normals of the hexagon are the critical directions.
        # 
        # Let's compute the "gap" in each of the 6 directions.
        # In direction n_k, the projection of hex i is [c_i . n_k - R_k, c_i . n_k + R_k]
        # The projection of hex j is [c_j . n_k - R_k, c_j . n_k + R_k]
        # The gap in direction n_k is:
        # gap_k = min(proj_i, proj_j) - max(proj_i, proj_j) ? No.
        # They are disjoint in direction n_k if the intervals do not overlap.
        # Interval i: [L_i, U_i], Interval j: [L_j, U_j].
        # Disjoint if U_i < L_j or U_j < L_i.
        # Overlap if max(L_i, L_j) < min(U_i, U_j).
        # The "depth of overlap" in direction n_k is:
        # overlap_k = min(U_i, U_j) - max(L_i, L_j)
        # If overlap_k > 0, they overlap in this direction.
        # They intersect if overlap_k > 0 for ALL k.
        # They are disjoint if there EXISTS k such that overlap_k <= 0.
        # 
        # So, a smooth penalty for intersection is:
        # P = - smooth_min_k (overlap_k)
        # If they are disjoint, some overlap_k <= 0, so min <= 0, -min >= 0.
        # Wait, if they are disjoint, min(overlap) is negative (or zero). We want penalty 0.
        # If they intersect, all overlap_k > 0, so min(overlap) > 0. Penalty should be positive.
        # So P = smooth_max(0, smooth_min_k(overlap_k))?
        # Or simply P = smooth_min_k(overlap_k).
        # If disjoint, min < 0. We want 0.
        # So P = softplus(smooth_min_k(overlap_k)).
        
        # Compute overlap_k for all pairs (i, j) and directions k.
        # This is O(N^2 * 6).
        
        penalty_disjoint = 0.0
        # For speed, we can vectorize partially, but N=12 is small.
        for i in range(N):
            for j in range(i + 1, N):
                ci = centers[i]
                cj = centers[j]
                
                # Projections
                # proj_i_k = ci . n_k
                # proj_j_k = cj . n_k
                pi = ci @ normals.T  # (6,)
                pj = cj @ normals.T  # (6,)
                
                # Intervals
                # L_i = pi - R_k, U_i = pi + R_k
                # L_j = pj - R_k, U_j = pj + R_k
                
                # overlap_k = min(pi+R, pj+R) - max(pi-R, pj-R)
                # = min(pi, pj) + R - (max(pi, pj) - R)
                # = min(pi, pj) - max(pi, pj) + 2*R
                # = -|pi - pj| + 2*R_k
                
                diff_proj = np.abs(pi - pj)
                overlaps = 2 * R_k - diff_proj
                
                # We want to penalize if min(overlaps) > 0 (intersection)
                # Smooth min of overlaps
                # min(x) = -max(-x)
                # smooth_min(x) = -beta * log(sum(exp(-x/beta)))
                
                # stable smooth min
                neg_overlaps = -overlaps
                max_neg = np.max(neg_overlaps)
                # log(sum(exp(neg_overlaps - max_neg))) + max_neg
                # If max_neg is large, exp might overflow? No, we subtracted max.
                # But if overlaps are very negative, neg_overlaps are very positive.
                # Let's just use a simple approximation or clip.
                # For robustness, let's use:
                # smooth_min = max_neg + beta * log(sum(exp((neg_overlaps - max_neg)/beta)))
                # Then P_pair = softplus(smooth_min)
                
                # Actually, if they are far apart, overlaps are very negative.
                # smooth_min will be very negative. softplus will be ~0.
                # If they intersect, overlaps are positive. smooth_min positive. softplus ~ smooth_min.
                
                # Let's use beta = 0.1 for smooth min as well.
                sm = max_neg + beta * np.log(np.sum(np.exp((neg_overlaps - max_neg) / beta)))
                penalty_disjoint += np.log1p(np.exp(np.clip(sm, -500, 30)))
        
        total_penalty = 10.0 * outer_penalty + 100.0 * penalty_disjoint
        
        # Objective: Minimize t + penalties
        # Note: t is in params.
        return t + total_penalty
    
    def gradient(params):
        # This is complex to derive analytically with all the softplus and smooth min.
        # We will use finite differences or rely on L-BFGS-B with finite diff?
        # L-BFGS-B with finite diff for 25 variables is slow but acceptable for 120s?
        # 25 variables, ~1000 iterations * 25 evals = 25000 evals.
        # Each eval is O(N^2) ~ 144 ops. Very fast.
        # So finite differences are fine.
        return None # Force finite diff

    # Initial guesses
    configs = []
    
    # 1. Hexagonal lattice
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    init1_arr = np.array(init1)
    # Estimate t
    t_est = 5.0
    configs.append(np.concatenate([init1_arr.flatten(), [t_est]]))
    
    # 2. Circular
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(np.concatenate([init2.flatten(), [t_est]]))
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-4, 4, (N, 2))
    configs.append(np.concatenate([init3.flatten(), [t_est]]))
    
    best_params = None
    best_score = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False}
            )
            
            # Check validity and compute exact score
            centers = res.x[:2*N].reshape(N, 2)
            t_val = res.x[2*N]
            
            # Exact check:
            # 1. Disjointness
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    ci = centers[i]
                    cj = centers[j]
                    pi = ci @ normals.T
                    pj = cj @ normals.T
                    overlaps = 2 * R_k - np.abs(pi - pj)
                    if np.min(overlaps) > 1e-5: # Intersection
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # 2. Exact outer side
                # t_exact = max_{i,k} (n_k . c_i + R_k)
                dots = centers @ normals.T
                support_vals = dots + R_k
                t_exact = np.max(support_vals)
                
                if t_exact < best_score:
                    best_score = t_exact
                    best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = configs[0]
        centers = best_params[:2*N].reshape(N, 2)
        dots = centers @ normals.T
        support_vals = dots + R_k
        best_score = np.max(support_vals)
    
    final_centers = best_params[:2*N].reshape(N, 2)
    final_t = best_score
    
    inner = []
    for i in range(N):
        inner.append([float(final_centers[i, 0]), float(final_centers[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_t), 0.0

# EVOLVE-BLOCK-END
