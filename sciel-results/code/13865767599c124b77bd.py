"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will try a range of aspect ratios.
    # Hexagonal packing is densest in infinite plane, but boundary effects matter.
    # Square (1x1) is a strong candidate. Slightly rectangular might help fit rows better.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.95, 1.05]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Generate Hexagonal Lattice Points
        # Standard hex packing:
        # Row i: y = i * sqrt(3)/2 * d
        # Col j: x = j * d + (i % 2) * d/2
        # We want to fit N=21 points.
        # Let's estimate the number of rows and cols.
        # Area approx: N * (pi/2) * r^2 ? No, packing density ~0.9069
        # Let's just generate a grid of candidates and pick the best 21?
        # Better: Generate a hex lattice with spacing d=1, scale it to fit, then optimize.
        
        # Let's create a hex lattice with enough points (e.g., 25) and trim to 21?
        # Or generate exactly 21 in a hex shape?
        # 1, 3, 5, 7, 5, 3, 1 = 25 points.
        # 21 points could be 1, 3, 5, 7, 5 = 21? 
        # Rows: 1, 3, 5, 7, 5.
        # Let's try to generate a hexagonal cluster of 21 points.
        
        points = []
        # Center at (0,0)
        # Hexagonal coordinates (q, r)
        # Max distance from center?
        # Let's just generate a square grid of hex points and filter.
        
        # Generate a range of hex points
        # Spacing s = 1.0 initially
        s = 1.0
        sqrt3 = math.sqrt(3)
        
        # Generate points in a bounding box large enough
        max_idx = 5
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                x = s * (j + 0.5 * (i & 1))
                y = s * (sqrt3 / 2.0 * i)
                points.append((x, y))
        
        # Sort by distance from center to get a compact cluster?
        # Or just take the first N?
        # Let's sort by distance to ensure we get a roughly circular/hexagonal blob.
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take the first 21
        hex_pts = np.array(points[:N])
        
        # Now we have 21 points in a hex arrangement with spacing s=1.
        # We need to map this to the rectangle [0, w] x [0, h].
        # Current bounds of hex_pts:
        min_x = np.min(hex_pts[:, 0])
        max_x = np.max(hex_pts[:, 0])
        min_y = np.min(hex_pts[:, 1])
        max_y = np.max(hex_pts[:, 1])
        
        width_pts = max_x - min_x
        height_pts = max_y - min_y
        
        if width_pts == 0 or height_pts == 0:
            continue
            
        # Scale factors to fit into w and h
        # We want to maximize radii.
        # If we scale x by sx and y by sy, the "effective radius" in x is r_x = r * sx?
        # No, circles are isotropic. If we scale the coordinates, the distances change.
        # Let's define the optimization variables as the centers (cx, cy) and radii (r).
        # Initial guess:
        # Map hex_pts to [0, w] x [0, h]
        # cx = (hex_pts[:, 0] - min_x) / width_pts * w
        # cy = (hex_pts[:, 1] - min_y) / height_pts * h
        
        cx_init = (hex_pts[:, 0] - min_x) / width_pts * w
        cy_init = (hex_pts[:, 1] - min_y) / height_pts * h
        
        # Initial radius:
        # The minimum distance between points in the scaled hex lattice.
        # In original lattice, min dist is s=1.
        # After scaling:
        # dx_scaled = dx_orig / width_pts * w
        # dy_scaled = dy_orig / height_pts * h
        # This is an anisotropic scaling, which distorts circles into ellipses effectively.
        # To keep circles, we should use a uniform scale factor first, then allow the optimizer to adjust.
        # Or, initialize with a uniform scale that fits in the smaller dimension?
        
        # Let's use a uniform scale based on the larger dimension to avoid immediate overlap,
        # then let the optimizer push points to boundaries.
        
        # Actually, a better init:
        # Assume uniform radius r.
        # In hex packing, density is pi / (2*sqrt(3)) ~ 0.9069.
        # Area of rect = w*h.
        # Total area of circles = N * pi * r^2.
        # Max r approx sqrt(w*h * 0.9 / N)
        
        r_init = math.sqrt(w * h * 0.8 / N) # 0.8 to be safe
        
        # But we must respect the geometry.
        # Let's just use the scaled hex positions and set r to a small fraction of min distance.
        
        # Compute min distance in scaled coordinates
        min_dist = 1e9
        for i in range(N):
            for j in range(i+1, N):
                dx = cx_init[i] - cx_init[j]
                dy = cy_init[i] - cy_init[j]
                d = math.hypot(dx, dy)
                if d < min_dist:
                    min_dist = d
        
        r_init = min_dist * 0.49 # Safe initial radius
        
        cx = cx_init.copy()
        cy = cy_init.copy()
        r = np.full(N, r_init)
        
        x0 = np.concatenate([cx, cy, r])
        
        # Bounds
        # cx in [r, w-r] -> [0, w] with constraint cx >= r
        # It's easier to bound cx in [0, w] and cy in [0, h] and r in [0, min(w,h)/2]
        # and handle boundary constraints separately.
        
        r_max = min(w, h) / 2.0
        
        bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N
        
        # Objective: Maximize sum of radii
        def objective(x):
            return -np.sum(x[2*N:])
        
        def objective_grad(x):
            g = np.zeros(3*N)
            g[2*N:] = -1.0
            return g
        
        # Constraints
        # We will use a penalty method inside the objective for SLSQP?
        # No, SLSQP handles constraints well.
        # But 210 non-overlap constraints + 84 boundary constraints is a lot.
        # Let's try to reduce constraints or use a penalty approach which is faster for this size.
        
        # Penalty Method Strategy:
        # Minimize -sum(r) + lambda * (sum of violations)
        # Violations:
        # 1. Boundary: max(0, r - cx) + max(0, cx + r - w) + ...
        # 2. Overlap: max(0, r_i + r_j - dist(i,j))
        
        # This is non-smooth. We can smooth it: softplus or just use SLSQP with hard constraints.
        # SLSQP with 294 constraints might be slow.
        # Let's try a simplified constraint set or a penalty approach with scipy.optimize.minimize 'L-BFGS-B'
        # which doesn't support constraints, so we must penalize.
        
        # Let's use a penalty function with L-BFGS-B. It's fast.
        # Penalty weight needs to be high enough to enforce constraints.
        
        def penalized_objective(x, penalty_weight=100.0):
            cx = x[:N]
            cy = x[N:2*N]
            r = x[2*N:]
            
            base_obj = -np.sum(r)
            
            penalty = 0.0
            
            # Boundary penalties
            # Use squared violations for smoothness? Or linear?
            # Linear is better for keeping them out.
            # v = max(0, violation)
            
            # Left
            v = np.maximum(0, r - cx)
            penalty += np.sum(v)
            # Right
            v = np.maximum(0, cx + r - w)
            penalty += np.sum(v)
            # Bottom
            v = np.maximum(0, r - cy)
            penalty += np.sum(v)
            # Top
            v = np.maximum(0, cy + r - h)
            penalty += np.sum(v)
            
            # Overlap penalties
            # For each pair, penalty = max(0, r_i + r_j - dist)^2 ?
            # Squared penalty helps gradient vanish at boundary? No, we want strong penalty.
            # Let's use squared to make it differentiable everywhere (except kinks, but max is kinked).
            # Actually, max(0, x)^2 is differentiable.
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = math.hypot(dx, dy)
                    violation = r[i] + r[j] - dist
                    if violation > 0:
                        penalty += violation * violation
            
            return base_obj + penalty_weight * penalty
        
        # We need to tune penalty_weight. If too low, invalid. If too high, ill-conditioned.
        # Let's start with 10.0 and increase if invalid.
        
        # Try multiple penalty weights
        penalty_weights = [1.0, 10.0, 100.0]
        
        for pw in penalty_weights:
            if time.time() > deadline - 5:
                break
                
            def obj_pw(x):
                return penalized_objective(x, pw)
            
            # We don't provide gradient to L-BFGS-B to save coding time, 
            # finite differences might be okay for 63 vars?
            # 63 vars, finite diff takes 64 evals.
            # If each eval is fast (numpy ops), it's okay.
            
            try:
                res = minimize(
                    obj_pw,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                x_opt = res.x
                cx_opt = x_opt[:N]
                cy_opt = x_opt[N:2*N]
                r_opt = x_opt[2*N:]
                
                # Validate strictly
                valid = True
                if np.any(cx_opt - r_opt < -1e-7) or np.any(cx_opt + r_opt - w > 1e-7):
                    valid = False
                if np.any(cy_opt - r_opt < -1e-7) or np.any(cy_opt + r_opt - h > 1e-7):
                    valid = False
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = math.hypot(cx_opt[i] - cx_opt[j], cy_opt[i] - cy_opt[j])
                            if dist < r_opt[i] + r_opt[j] - 1e-7:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((cx_opt, cy_opt, r_opt))
                        
            except Exception:
                pass

    # Fallback: simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
