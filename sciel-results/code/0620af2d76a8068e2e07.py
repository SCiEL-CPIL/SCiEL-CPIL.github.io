import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _separation_violations(centers, normals):
    """Return a list of violation values for each pair.
    Violation > 0 means overlap.
    """
    sqrt3 = np.sqrt(3.0)
    n = len(centers)
    violations = []
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            d_dots = np.abs(diff @ normals.T)
            max_d_dot = np.max(d_dots)
            viol = sqrt3 - max_d_dot
            violations.append(viol)
    return violations

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    best_L = float('inf')
    best_inner = None
    best_params = None
    
    # Radial Slicing: 1 center, 5 inner ring, 6 outer ring
    # Parameters: [r_inner, r_outer, rot_inner, rot_outer]
    # Inner ring: 5 hexagons at angles 0, 72, 144, 216, 288 + rot_inner
    # Outer ring: 6 hexagons at angles 0, 60, 120, 180, 240, 300 + rot_outer
    
    def build_centers(params):
        r_in, r_out, rot_in, rot_out = params
        rot_in_rad = np.deg2rad(rot_in)
        rot_out_rad = np.deg2rad(rot_out)
        centers = np.zeros((N, 2))
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # Inner ring: 5 hexagons
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_in_rad
        for k in range(5):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
            
        # Outer ring: 6 hexagons
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_out_rad
        for k in range(6):
            centers[6 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
            
        return centers
    
    def objective(params):
        centers = build_centers(params)
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 2.0 * max_abs_dot / sqrt3 + 1.0
        return L_req
    
    # Constraints: separation
    # We need sqrt(3) - max_k |u_k . (c_i - c_j)| <= 0
    # This is equivalent to max_k |u_k . (c_i - c_j)| >= sqrt(3)
    # We can formulate this as multiple linear constraints? No, it's a max.
    # max_k A_k >= C  <=>  A_k >= C for all k? No.
    # max_k A_k >= C is satisfied if at least one A_k >= C.
    # This is a disjunctive constraint, which SLSQP cannot handle directly.
    # However, for "close" hexagons in a ring, the dominant direction is usually the radial one or the tangential one.
    # A common trick is to approximate the non-convex constraint by the convex hull of the feasible region?
    # The feasible region for $d$ is the complement of the hexagon. It is non-convex.
    # But if we know which face is the "active" one, it becomes linear.
    # Alternatively, we can use the penalty method but with a much higher weight and SLSQP to find the boundary.
    # Or, we can use the fact that for the specific radial arrangement, the critical pairs are neighbors.
    # Let's stick to the penalty method but improve the optimizer to SLSQP with a large penalty, or use Nelder-Mead with more iterations and better initial guesses.
    # Actually, the current Nelder-Mead is quite good for 4 parameters.
    # The issue might be getting stuck in local minima.
    # Let's try to optimize the objective L directly, and check validity.
    # If we use SLSQP, we need gradients.
    
    # Let's try a different strategy: 
    # Minimize L subject to "soft" separation constraints using SLSQP.
    # Constraint: sum of violations is 0? No.
    # Let's use the penalty method but optimize with L-BFGS-B? No, L-BFGS-B is for bounds.
    # Let's use Nelder-Mead but with a smarter objective that penalizes violations more aggressively only when they are present, or uses a barrier.
    
    # Actually, the current code's score is 8.0. This is likely the "seed" score or a baseline.
    # Wait, "score 8.0000" is the current program's score.
    # The task says "score is the outer side length".
    # If the score is 8.0, that's a very large hexagon.
    # A packing of 12 unit hexagons should fit in a hexagon of side ~ 4 or 5?
    # Area of 12 unit hexagons = 12 * (3*sqrt(3)/2) = 18*sqrt(3) ≈ 31.17.
    # Area of container hexagon side L = (3*sqrt(3)/2) L^2.
    # L^2 >= 31.17 / (3*sqrt(3)/2) = 31.17 / 2.598 ≈ 12.
    # L >= sqrt(12) ≈ 3.46.
    # So 8.0 is very bad. The current program is failing to find a good packing.
    # Why?
    # The constraint `max_d_dot < sqrt(3)` is the violation.
    # If the hexagons are too far apart, `max_d_dot` is large, viol is negative.
    # The penalty is `1000 * viol * viol`.
    # If the hexagons overlap, viol is positive.
    # The objective is `L_req + penalty`.
    # If the optimizer finds a configuration with L=4 but significant overlap, the penalty might be huge.
    # If the optimizer finds a configuration with L=8 and no overlap, the score is 8.
    # It seems the optimizer is stuck in a "spread out" local minimum to avoid the penalty.
    
    # To improve, we need to guide the optimizer to pack tightly.
    # We can use a "soft" constraint that allows slight overlap but penalizes it, and then polish.
    # Or, we can use a multi-start approach with tighter bounds.
    
    # Let's try to constrain the radii to be small.
    # r_in should be around 1.732 (touching center).
    # r_out should be around 3.464 (touching inner ring).
    
    # Let's rewrite the optimization loop to use SLSQP with explicit constraints if possible, 
    # or just a better Nelder-Mead setup.
    
    # Since SLSQP can't handle the disjunctive constraint `max(...) >= C`, we will stick to penalty but tune it.
    # We will use a "ramp" penalty: if viol > 0, penalty is huge.
    # Also, we will add a small regularization to keep parameters in a reasonable range.
    
    # Let's try to optimize the 4 parameters with Nelder-Mead, but with a better objective:
    # Objective = L_req + M * max(0, max_violation)
    # Using max(0, viol) makes it non-differentiable but Nelder-Mead doesn't need gradients.
    # This is better than squared penalty because it doesn't grow quadratically, which can distort the landscape.
    
    def objective_new(params):
        centers = build_centers(params)
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 2.0 * max_abs_dot / sqrt3 + 1.0
        
        # Calculate max violation
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                d_dots = np.abs(diff @ normals.T)
                max_d_dot = np.max(d_dots)
                viol = sqrt3 - max_d_dot
                if viol > max_viol:
                    max_viol = viol
                    
        # Penalize overlap linearly with a large factor
        penalty = 100.0 * max_viol
        return L_req + penalty
    
    # Try multiple random starts with tighter bounds
    for seed in range(200):
        np.random.seed(seed)
        # Tighter initial guesses
        r_in = np.random.uniform(1.7, 2.2)
        r_out = np.random.uniform(3.0, 4.0)
        rot_in = np.random.uniform(0, 72)
        rot_out = np.random.uniform(0, 60)
        
        x0 = np.array([r_in, r_out, rot_in, rot_out])
        
        try:
            result = minimize(objective_new, x0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            params = result.x
            
            # Verify
            centers = build_centers(params)
            max_viol = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    diff = centers[i] - centers[j]
                    d_dots = np.abs(diff @ normals.T)
                    max_d_dot = np.max(d_dots)
                    viol = sqrt3 - max_d_dot
                    if viol > max_viol:
                        max_viol = viol
                        
            if max_viol > 1e-5:
                continue
            
            max_abs_dot = np.max(np.abs(centers @ normals.T))
            L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
                
        except Exception:
            continue
            
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_inner, (0.0, 0.0), best_L, 0.0
