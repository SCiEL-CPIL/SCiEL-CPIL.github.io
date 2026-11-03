import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configuration: A known good packing (lattice-like)
    # We use axial coordinates for initialization but convert to Cartesian for optimization
    # A standard "flower" packing or a linear strip. 
    # Let's start with a compact cluster.
    # Axial coordinates for a compact shape of 11 hexagons:
    # (0,0) center
    # 6 neighbors
    # 4 more around
    # Let's use a specific known good configuration for N=11 to ensure we start valid.
    # A 3-3-3-2 strip? Or a 2-3-3-3?
    # Let's use the "hexagonal number" logic: 1 + 6 + 4 = 11.
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (partial): (2,0), (0,2), (-2,2), (2,-2) -- wait, this is 11 points.
    # Let's verify disjointness for unit hexagons on this lattice.
    # Distance between centers in axial (q,r) is sqrt(3) * distance in cartesian.
    # Nearest neighbors in lattice have distance sqrt(3).
    # Hexagon width (corner to corner) is 2. Apothem is sqrt(3)/2.
    # If centers are distance d apart, hexagons are disjoint if d >= 2 * apothem * 2 / sqrt(3) ?
    # Actually, for regular hexagons of side 1, the minimum distance between centers to be disjoint is 2 * apothem = sqrt(3).
    # So a lattice packing with nearest neighbor distance sqrt(3) is exactly touching.
    
    axial_init = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    pts = axial_to_cartesian(axial_init)
    
    # Center of mass
    center = np.mean(pts, axis=0)
    # Initial beta (rotation of the outer hexagon)
    beta = 0.0
    
    # Objective function: Minimize the side length of the enclosing regular hexagon
    # Side length S = (D + APOTHEM) / APOTHEM
    # D is the maximum projection of the points onto the 6 normals of the outer hexagon,
    # centered at 'center'.
    # To make this differentiable, we use a soft-max approximation.
    
    def get_normals(beta):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        return np.array(normals)

    def smooth_objective(params, temperature=0.1):
        # params: [cx, cy, beta, x1, y1, ..., x11, y11]
        # But we can optimize relative to center of mass to reduce dimensions?
        # Or just optimize all positions and center.
        # Let's optimize: cx, cy, beta, and 11*2 positions.
        # Total dims: 3 + 22 = 25.
        
        cx = params[0]
        cy = params[1]
        b = params[2]
        
        # Extract points
        pts_curr = params[3:].reshape((11, 2))
        
        # Ensure points are disjoint? 
        # The task says "disjoint". 
        # If we just minimize the container, they might overlap.
        # We need a penalty for overlap or ensure the initial state is valid and moves are small.
        # Overlap penalty: For each pair, if distance < 2*APOTHEM, add penalty.
        # 2*APOTHEM = sqrt(3).
        # Distance between hexagons of side 1:
        # The support function of a hexagon is a piecewise linear function.
        # Approximation: If centers are closer than 2*APOTHEM, they overlap.
        # Actually, for regular hexagons, the Minkowski sum of two unit hexagons is a larger hexagon.
        # Two unit hexagons are disjoint iff the distance between centers is >= 2 * APOTHEM = sqrt(3) ALONG THE APOTHEM DIRECTION?
        # No, the minimum distance between two disjoint unit hexagons is 2 * APOTHEM = sqrt(3) if they are aligned.
        # If rotated, the minimum distance can be smaller? No, the inradius is the minimum distance to the boundary.
        # So if distance between centers > 2 * inradius, they are definitely disjoint.
        # 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3).
        
        overlap_penalty = 0.0
        # Check all pairs
        for i in range(11):
            for j in range(i+1, 11):
                dx = pts_curr[i, 0] - pts_curr[j, 0]
                dy = pts_curr[i, 1] - pts_curr[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = math.sqrt(3) # 2 * apothem
                if dist < min_dist:
                    # Quadratic penalty
                    overlap_penalty += (min_dist - dist) ** 2
        
        normals = get_normals(b)
        
        # Projections
        # pts_curr relative to center
        pts_rel = pts_curr - np.array([cx, cy])
        
        # Project onto each normal
        # projs[i, k] = dot(pts_rel[i], normals[k])
        projs = pts_rel @ normals.T
        
        # We need max over all points and all normals of |proj|
        # This is equivalent to max over all points and all normals of proj, and min over all points and all normals of proj.
        # D = max( max(projs), -min(projs) )
        
        # Soft max
        max_p = np.max(projs)
        min_p = np.min(projs)
        
        # Soft max of max_p
        shifted_max = (projs - max_p) / temperature
        shifted_max = np.clip(shifted_max, -50, 50)
        log_sum_max = np.log(np.sum(np.exp(shifted_max))) + max_p / temperature
        D_max = temperature * log_sum_max
        
        # Soft min of min_p is - Soft max of -projs
        shifted_min = (-projs - (-min_p)) / temperature
        shifted_min = np.clip(shifted_min, -50, 50)
        log_sum_min = np.log(np.sum(np.exp(shifted_min))) + (-min_p) / temperature
        D_min = temperature * log_sum_min
        
        D = max(D_max, D_min)
        
        S = (D + APOTHEM) / APOTHEM
        
        return S + 10.0 * overlap_penalty

    # Optimization using Nelder-Mead or BFGS
    # Nelder-Mead is robust but slow. BFGS is faster but needs gradients.
    # Let's use Nelder-Mead with a reasonable number of iterations.
    
    init_params = np.concatenate(([center[0], center[1], beta], pts.ravel()))
    
    best_S = float('inf')
    best_params = init_params.copy()
    
    # Run multiple restarts with different initial temperatures and random perturbations
    # Time budget: ~140 seconds
    
    # Define a helper to run one optimization
    def run_optimization(init_p, temp, max_iter=2000):
        def obj(p):
            return smooth_objective(p, temperature=temp)
        
        # Use scipy minimize
        from scipy.optimize import minimize
        
        res = minimize(obj, init_p, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        return res.fun, res.x

    # Strategy:
    # 1. Coarse search with high temperature to find a good basin.
    # 2. Fine search with low temperature to polish.
    
    # Since we don't have gradients, Nelder-Mead is the way.
    # To speed it up, we can use a smaller simplex or fewer iterations initially.
    
    # Let's do a sequence of optimizations, lowering temperature.
    temps = [0.5, 0.2, 0.1, 0.05, 0.01]
    
    current_params = init_params.copy()
    
    for t in temps:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Add small random noise to escape local minima
        noise = np.random.normal(0, 0.1, size=current_params.shape)
        # Don't move center/beta too much
        noise[0] *= 0.1
        noise[1] *= 0.1
        noise[2] *= 0.05
        
        start_p = current_params + noise
        
        try:
            S, new_params = run_optimization(start_p, t, max_iter=1000)
            if S < best_S:
                best_S = S
                best_params = new_params
                current_params = new_params
        except Exception:
            pass
    
    # Final exact evaluation and extraction
    if best_S == float('inf'):
        # Fallback to initial
        cx, cy = center
        beta = 0.0
        pts_final = pts
    else:
        cx = best_params[0]
        cy = best_params[1]
        beta = best_params[2]
        pts_final = best_params[3:].reshape((11, 2))
        
        # Calculate exact S for reporting
        normals = get_normals(beta)
        pts_rel = pts_final - np.array([cx, cy])
        projs = pts_rel @ normals.T
        D = max(np.max(projs), -np.min(projs))
        exact_S = (D + APOTHEM) / APOTHEM
        
        # If the optimized S was due to overlap penalty, we might have overlaps.
        # We must ensure the final configuration is valid (disjoint).
        # Check for overlaps in the final configuration.
        is_valid = True
        for i in range(11):
            for j in range(i+1, 11):
                dx = pts_final[i, 0] - pts_final[j, 0]
                dy = pts_final[i, 1] - pts_final[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < math.sqrt(3) - 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if not is_valid:
            # If invalid, we should probably revert to a known valid configuration or fix it.
            # For safety, if the optimizer broke validity, we return the best VALID configuration found during the process?
            # We didn't track that. 
            # Alternative: Use the initial configuration if the optimized one is invalid.
            # Or, we can try to "push apart" overlapping hexagons.
            # Given time constraints, reverting to initial is safer than returning an invalid score (0).
            # But the initial score is ~10. The optimized one might be <10 but invalid.
            # Let's check if the penalty was high.
            # If S > 10.5, it's likely bad.
            if exact_S > 10.0:
                # Revert to initial
                cx, cy = center
                beta = 0.0
                pts_final = pts
                exact_S = 10.0 # Approx
                # Recalc exact S for initial
                normals = get_normals(beta)
                pts_rel = pts_final - np.array([cx, cy])
                projs = pts_rel @ normals.T
                D = max(np.max(projs), -np.min(projs))
                exact_S = (D + APOTHEM) / APOTHEM
            else:
                # It's close to optimal but has overlaps?
                # This is a risk. The evaluator will score 0 if invalid.
                # To be safe, if there is ANY overlap, we should fix it or revert.
                # Let's implement a simple repulsion step for the final result.
                for _ in range(50):
                    moved = False
                    for i in range(11):
                        for j in range(i+1, 11):
                            dx = pts_final[i, 0] - pts_final[j, 0]
                            dy = pts_final[i, 1] - pts_final[j, 1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            min_dist = math.sqrt(3)
                            if dist < min_dist:
                                # Push apart
                                if dist < 1e-6:
                                    dx = 1.0
                                    dy = 0.0
                                    dist = 1.0
                                push = (min_dist - dist) / 2.0
                                nx = dx / dist
                                ny = dy / dist
                                pts_final[i, 0] += nx * push
                                pts_final[i, 1] += ny * push
                                pts_final[j, 0] -= nx * push
                                pts_final[j, 1] -= ny * push
                                moved = True
                    if not moved:
                        break
                
                # Recalc center and S
                center_new = np.mean(pts_final, axis=0)
                cx, cy = center_new[0], center_new[1]
                normals = get_normals(beta)
                pts_rel = pts_final - np.array([cx, cy])
                projs = pts_rel @ normals.T
                D = max(np.max(projs), -np.min(projs))
                exact_S = (D + APOTHEM) / APOTHEM

    # Final output
    angle_deg = math.degrees(beta)
    
    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center_out = [float(cx), float(cy)]
    side_out = float(exact_S)
    rotation_out = float(angle_deg)
    
    return inner, center_out, side_out, rotation_out

