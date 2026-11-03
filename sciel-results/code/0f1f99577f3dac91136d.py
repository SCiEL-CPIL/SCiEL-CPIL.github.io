"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will explore a range of aspect ratios.
    # Hexagonal packing is densest in infinite plane, but boundary effects matter.
    
    # Pre-compute pairs for overlap constraints
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    def validate_and_score(x, w, h):
        cx_v = x[:N]
        cy_v = x[N:2*N]
        r_v = x[2*N:]
        
        # Check bounds
        if np.any(cx_v - r_v < -1e-7) or np.any(cx_v + r_v - w > 1e-7):
            return -1, None
        if np.any(cy_v - r_v < -1e-7) or np.any(cy_v + r_v - h > 1e-7):
            return -1, None
        
        # Check overlaps
        dx = cx_v[:, None] - cx_v[None, :]
        dy = cy_v[:, None] - cy_v[None, :]
        dist_sq = dx*dx + dy*dy
        r_sum = r_v[:, None] + r_v[None, :]
        
        # Only check upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol_sq = dist_sq[mask] - r_sum[mask]**2
        if np.any(viol_sq < -1e-7):
            return -1, None
        
        score = np.sum(r_v)
        return score, np.column_stack((cx_v, cy_v, r_v))

    def refine_coordinates(x0, w, h):
        """
        Refine a given configuration using L-BFGS-B with penalty method.
        Returns (score, result_array).
        """
        r_max = min(w, h) / 2.0
        bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N
        
        def objective_and_grad(x):
            cx_v = x[:N]
            cy_v = x[N:2*N]
            r_v = x[2*N:]
            
            obj = -np.sum(r_v)
            grad = np.zeros(3*N)
            grad[2*N:] = -1.0
            
            # Overlap penalty
            dx = cx_v[:, None] - cx_v[None, :]
            dy = cy_v[:, None] - cy_v[None, :]
            dist_sq = dx*dx + dy*dy
            dist_sq[dist_sq < 1e-12] = 1e-12
            dist = np.sqrt(dist_sq)
            
            r_sum = r_v[:, None] + r_v[None, :]
            
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            d_pairs = dist[mask]
            rs_pairs = r_sum[mask]
            
            viol = rs_pairs - d_pairs
            viol_pos = np.maximum(0, viol)
            
            penalty = 50.0 # High penalty to enforce constraints strictly
            obj += penalty * np.sum(viol_pos**2)
            
            # Gradient of penalty
            ii, jj = np.where(mask)
            active = viol_pos > 0
            ii_act = ii[active]
            jj_act = jj[active]
            viol_act = viol_pos[active]
            d_act = d_pairs[active]
            
            if len(ii_act) > 0:
                factor = -2.0 * penalty * viol_act / d_act
                dx_act = cx_v[ii_act] - cx_v[jj_act]
                dy_act = cy_v[ii_act] - cy_v[jj_act]
                
                np.add.at(grad, ii_act, factor * dx_act)
                np.add.at(grad, jj_act, -factor * dx_act)
                np.add.at(grad, N + ii_act, factor * dy_act)
                np.add.at(grad, N + jj_act, -factor * dy_act)
                
                np.add.at(grad, 2*N + ii_act, 2.0 * penalty * viol_act)
                np.add.at(grad, 2*N + jj_act, 2.0 * penalty * viol_act)
            
            return obj, grad
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return validate_and_score(res.x, w, h)
        except Exception:
            return -1, None

    # Strategy:
    # 1. Generate hexagonal lattice points.
    # 2. Apply affine transformations (shear, scale) to fit into rectangle.
    # 3. Optimize radii for the fixed transformed positions? No, better to optimize positions and radii together from this good start.
    
    # Hexagonal lattice parameters
    # Standard hex packing: 
    # x = i * sqrt(3) * r + (j % 2) * sqrt(3)/2 * r
    # y = j * 1.5 * r
    # Here we fix the relative geometry and scale it.
    
    # We define the lattice in a "unit" space where the radius is 1.
    # Then we scale it to fit the rectangle.
    
    # Let's try a few aspect ratios and lattice orientations.
    # Orientation 0: Flat top
    # Orientation 1: Pointy top (rotated 90 deg)
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        
        # Try both orientations
        for orientation in [0, 1]:
            if time.time() > deadline - 5:
                break
            
            # Generate lattice points in unit space (r=1)
            # We need enough points to cover N=21
            # 5x5 grid is 25 points.
            # Hex grid 5x5 is roughly 25 points.
            
            pts = []
            # Generate a 6x6 grid to have plenty of candidates
            for i in range(6):
                for j in range(6):
                    if orientation == 0:
                        # Flat top
                        x = i * math.sqrt(3) + (j % 2) * math.sqrt(3) / 2
                        y = j * 1.5
                    else:
                        # Pointy top
                        x = i * 1.5
                        y = j * math.sqrt(3) + (i % 2) * math.sqrt(3) / 2
                    pts.append((x, y))
            
            pts = np.array(pts)
            
            # We want to select a subset of 21 points that, when scaled, fits in w x h.
            # Actually, it's better to generate the lattice, then apply an affine transform 
            # (scale_x, scale_y, shear) to fit it into the box, then optimize.
            
            # Let's try a simpler approach:
            # 1. Take the 21 points from the lattice that are closest to the center of the lattice.
            # 2. Calculate their bounding box.
            # 3. Scale them to fit w x h.
            # 4. Use this as the initial guess for L-BFGS-B.
            
            # Center the lattice points
            pts_centered = pts - np.mean(pts, axis=0)
            
            # Select 21 points closest to origin
            dists = np.linalg.norm(pts_centered, axis=1)
            idx_sorted = np.argsort(dists)
            selected_idx = idx_sorted[:N]
            pts_sel = pts_centered[selected_idx]
            
            # Calculate bounding box of selected points
            min_x, min_y = np.min(pts_sel, axis=0)
            max_x, max_y = np.max(pts_sel, axis=0)
            width_pts = max_x - min_x
            height_pts = max_y - min_y
            
            # Scale to fit rectangle
            # We want to maximize the radius. 
            # If we scale the points by factor s, the distances scale by s.
            # The radius in the final configuration will be r_final = r_unit * s.
            # We want to choose s such that the circles touch the boundary.
            
            # Let's just set the initial positions to be the scaled points.
            # Scale factor sx, sy to fit in [0, w] x [0, h]
            sx = w / width_pts if width_pts > 0 else 1
            sy = h / height_pts if height_pts > 0 else 1
            
            # Uniform scaling might be better to preserve hexagonal properties initially
            s = min(sx, sy) * 0.95 # 5% margin
            
            # Apply scaling and translation to center in box
            cx_init = pts_sel[:, 0] * s + w / 2
            cy_init = pts_sel[:, 1] * s + h / 2
            
            # Initial radius guess
            # The minimum distance between centers is sqrt(3) in unit space.
            # So r_unit = sqrt(3)/2.
            # r_init = r_unit * s
            r_init = (math.sqrt(3)/2) * s * 0.95
            
            x0 = np.concatenate([cx_init, cy_init, np.full(N, r_init)])
            
            # Run refinement
            score, res = refine_coordinates(x0, w, h)
            
            if score > best_score:
                best_score = score
                best_result = res

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
