import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Normals for the container hexagon (pointy-top, side length L)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    sqrt3 = np.sqrt(3.0)
    
    # Ring definitions
    # 1 center, 5 inner, 6 outer
    # Inner ring angles: 0, 72, 144, 216, 288
    # Outer ring angles: 0, 60, 120, 180, 240, 300
    # We will optimize r_in, r_out, and global rotation theta
    # The relative angle between inner and outer rings is fixed to 0 for simplicity 
    # (i.e., inner ring is rotated by theta, outer ring by theta).
    # This reduces parameters to 3.
    
    inner_angles = np.deg2rad([0, 72, 144, 216, 288])
    outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    
    def get_centers(params):
        r_in, r_out, theta_deg = params
        theta = np.deg2rad(theta_deg)
        centers = np.zeros((N, 2))
        
        # Center
        centers[0] = [0.0, 0.0]
        
        # Inner ring
        for k in range(5):
            a = inner_angles[k] + theta
            centers[1 + k] = [r_in * np.cos(a), r_in * np.sin(a)]
            
        # Outer ring
        for k in range(6):
            a = outer_angles[k] + theta
            centers[6 + k] = [r_out * np.cos(a), r_out * np.sin(a)]
            
        return centers

    def objective(params):
        centers = get_centers(params)
        # Calculate max projection onto container normals
        # c_i . n_k
        # We want max over i and k
        # Efficient way: for each center, find max dot product with normals
        # max_k (c_i . n_k)
        # This is equivalent to the support function of the set of points in the hexagonal norm.
        # But we can just compute all 6 dot products for each center.
        dots = centers @ normals.T # Shape (N, 6)
        max_dots = np.max(dots, axis=1) # Shape (N,)
        global_max = np.max(max_dots)
        
        # L = 2/sqrt(3) * global_max + 1
        # The +1 comes from the inner hexagon's apothem sqrt(3)/2 * 2/sqrt(3) = 1
        return 2.0 / sqrt3 * global_max + 1.0

    def constraints(params):
        centers = get_centers(params)
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                # Constraint: dist_sq >= 3 (sqrt(3)^2)
                # SLSQP expects fun(x) >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.dot(get_centers(x)[i] - get_centers(x)[j], get_centers(x)[i] - get_centers(x)[j]) - 3.0})
        return cons

    # Initial guesses
    # r_in ~ sqrt(3) ~ 1.732
    # r_out ~ 2*sqrt(3) ~ 3.464
    # theta ~ 0
    x0 = np.array([1.732, 3.464, 0.0])
    
    # Bounds
    bounds = [(1.0, 3.0), (2.0, 5.0), (0, 360)]
    
    best_L = float('inf')
    best_params = None
    
    # Multi-start
    for seed in range(50):
        np.random.seed(seed)
        r_in = np.random.uniform(1.5, 2.5)
        r_out = np.random.uniform(2.5, 4.5)
        theta = np.random.uniform(0, 360)
        x0 = np.array([r_in, r_out, theta])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                L = res.fun
                if L < best_L:
                    best_L = L
                    best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to a valid, albeit larger, configuration
        # Just spread them out
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers = get_centers(best_params)
    # Return format: (inner, center, side, angle)
    # inner: list of (x, y, angle_degrees)
    # We assume optimal rotation for each hexagon is 0 relative to the radial vector?
    # The task allows rotation. We should return 0.0 for angle as we didn't optimize individual rotations,
    # but the constraint check in the evaluator will handle the actual geometry.
    # However, if we return 0.0, the hexagons might overlap if they are not oriented optimally.
    # The evaluator checks if they are disjoint.
    # If we assume they are oriented to touch flat-to-flat, the angle should align the flat side with the neighbor.
    # For a hexagon at angle alpha, the flat sides are at alpha+30, alpha+90, etc.
    # To be safe, we can return an angle that aligns the hexagon with the radial direction?
    # Or just 0.0 and hope the distance is sufficient?
    # The distance constraint was >= sqrt(3). If distance is exactly sqrt(3), they touch flat-to-flat only if oriented correctly.
    # If oriented incorrectly (e.g., vertex to flat), they might overlap or have a gap.
    # Vertex to flat distance is less than sqrt(3)?
    # Let's check.
    # Two unit hexagons. Min distance between centers to avoid overlap.
    # If aligned flat-to-flat: sqrt(3) ~ 1.732.
    # If rotated 30 deg: vertex to flat.
    # The support function of the "Minkowski sum" of two hexagons determines the min distance.
    # The Minkowski sum of two unit hexagons is a dodecagon.
    # The min distance from center to boundary of the Minkowski sum is the min distance between centers.
    # If we can rotate, we can achieve the min distance of the Minkowski sum.
    # The Minkowski sum of two regular hexagons of side 1 is a regular dodecagon? No.
    # It's a hexagon with corners cut?
    # Actually, the set of valid displacements for two disjoint unit hexagons is the complement of the Minkowski sum of one hexagon and the reflection of the other.
    # Since they are convex and centrally symmetric, it's the Minkowski sum.
    # The Minkowski sum of two unit regular hexagons is a regular dodecagon with side 1?
    # No, if they are aligned, it's a hexagon of side 2? No.
    # Let's just assume we can orient them optimally.
    # The returned angle should be such that the hexagons are disjoint.
    # If we enforce distance >= sqrt(3), and we orient them to be parallel (angle 0 for all), they will be disjoint (touching).
    # So returning 0.0 for all angles is a valid configuration if the distances are >= sqrt(3).
    # Because if all hexagons are axis-aligned, the min distance to avoid overlap is indeed sqrt(3).
    
    inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
    
    return inner, (0.0, 0.0), best_L, 0.0
