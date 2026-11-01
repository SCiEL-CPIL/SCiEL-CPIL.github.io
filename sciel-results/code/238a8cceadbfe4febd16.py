"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)

def _hex_lattice_init(n_points, r_target, seed=0):
    """
    Generate initial positions using a hexagonal lattice scaled to fit radius r_target.
    We generate a large grid and pick the best n_points that fit within (1-r_target).
    """
    rng = np.random.RandomState(seed)
    
    # Optimal hex spacing for radius r is 2r
    # But we need to fit inside 1-r_target.
    # Let's generate points on a hex lattice with spacing s.
    # We want to find s such that roughly n_points fit in radius R_max = 1 - r_target.
    # Area of hex lattice point ~ s^2 * sqrt(3)/2.
    # Area of circle ~ pi * R_max^2.
    # n ~ pi * R_max^2 / (s^2 * sqrt(3)/2) => s ~ R_max * sqrt(2*pi / (n*sqrt(3)))
    R_max = 1.0 - r_target
    if R_max < 1e-6:
        return np.zeros((n_points, 2))
    
    # Estimate spacing
    s = R_max * np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # Generate grid
    # Range needs to cover the circle
    extent = R_max / s + 2
    points = []
    for i in range(-int(extent), int(extent) + 1):
        for j in range(-int(extent), int(extent) + 1):
            x = i * s + (j % 2) * 0.5 * s
            y = j * s * np.sqrt(3) / 2
            points.append([x, y])
    
    points = np.array(points)
    # Filter points within R_max
    dists = np.linalg.norm(points, axis=1)
    valid_idx = np.where(dists <= R_max)[0]
    
    if len(valid_idx) < n_points:
        # Fallback to random if lattice is too sparse or dense
        theta = rng.uniform(0, 2 * np.pi, n_points)
        r = np.sqrt(rng.uniform(0, R_max**2, n_points))
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Select n_points closest to center? Or random?
    # Random selection from valid points helps break symmetry
    chosen_idx = rng.choice(valid_idx, size=n_points, replace=False)
    pts = points[chosen_idx]
    
    # Add small jitter to break perfect symmetry
    pts += rng.normal(0, 0.01 * s, pts.shape)
    return pts

def _feasibility_optimize(r_target, init_pts, deadline=None):
    """
    Try to find a configuration of N circles with radius r_target inside unit circle.
    Objective: Sum of squared violations.
    Violations:
    1. Pairwise: max(0, 2*r_target - ||p_i - p_j||)^2
    2. Boundary: max(0, ||p_i|| + r_target - 1)^2
    """
    n = len(init_pts)
    x0 = init_pts.flatten()
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 2)
        
        # Boundary violations
        radii = np.linalg.norm(pts, axis=1)
        b_viol = np.maximum(0.0, radii + r_target - 1.0)
        obj_b = np.sum(b_viol ** 2)
        
        # Pairwise violations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We only care about pairs where dist < 2*r_target
        # v_ij = max(0, 2*r_target - dist_ij)
        # To compute gradient efficiently, we can just compute for all pairs and mask,
        # or use the fact that for N=50, full matrix is fine.
        
        # v = 2*r_target - dists
        # Only positive parts contribute
        v = 2.0 * r_target - dists
        v_pos = np.maximum(0.0, v)
        
        # Mask for active pairs (i != j and v > 0)
        mask = (v_pos > 0) & (dists < np.inf)
        
        # Objective is sum of v_pos^2 over i<j
        # Since v_pos is symmetric, sum over i<j is half of sum over i!=j
        # But we can just sum the upper triangle
        iu = np.triu_indices(n, k=1)
        v_pos_pairs = v_pos[iu]
        obj_p = np.sum(v_pos_pairs ** 2)
        
        total_obj = obj_b + obj_p
        
        # Gradients
        grad = np.zeros_like(x)
        
        # Boundary gradient
        # d/dp_i (max(0, ||p_i|| + r - 1)^2)
        # If viol > 0: 2 * viol * (p_i / ||p_i||)
        active_b = b_viol > 0
        if np.any(active_b):
            pts_active = pts[active_b]
            radii_active = radii[active_b]
            safe_radii = np.where(radii_active < 1e-10, 1e-10, radii_active)
            grad_b = 2.0 * b_viol[active_b, np.newaxis] * (pts_active / safe_radii[:, np.newaxis])
            # Accumulate
            grad[active_b] += grad_b
            
        # Pairwise gradient
        # d/dp_i sum_{j} max(0, 2r - ||p_i - p_j||)^2
        # For each active pair (i,j) with v > 0:
        # grad_i += 2 * v * (-(p_i - p_j) / ||p_i - p_j||) = 2 * v * (p_j - p_i) / dist
        # grad_j += 2 * v * (p_i - p_j) / dist
        
        # We have v_pos and dists matrices.
        # We need to accumulate contributions from all j for each i.
        # Let's use the full matrix logic for simplicity and speed (N=50 is small)
        
        # v_pos is N x N.
        # We want to sum over j != i.
        # term_ij = 2 * v_pos[i,j] * (p_j - p_i) / dists[i,j]
        
        # To avoid division by zero, use safe dists
        safe_dists = np.where(dists < 1e-10, 1e-10, dists)
        
        # Unit vector from i to j: u_ij = (p_j - p_i) / dists[i,j]
        # diff_ij = p_i - p_j, so u_ij = -diff_ij / dists[i,j]
        # Let's compute (p_j - p_i) directly
        # p_j - p_i = - (p_i - p_j) = - diff[i, j]
        
        # We need sum_j 2 * v_pos[i,j] * (p_j - p_i) / dists[i,j]
        
        # Let's construct the force matrix
        # F[i, j] = 2 * v_pos[i, j] / dists[i, j]
        # Then grad_i = sum_j F[i, j] * (p_j - p_i)
        
        # Note: v_pos[i, i] = 0 because dists[i,i]=inf -> v=2r-inf = -inf -> max(0, -inf)=0.
        # So diagonal is 0.
        
        F = 2.0 * v_pos / safe_dists
        
        # Zero out diagonal just in case
        np.fill_diagonal(F, 0.0)
        
        # grad_i = sum_j F[i, j] * (p_j - p_i)
        # = sum_j F[i, j] * p_j - sum_j F[i, j] * p_i
        # = (F @ pts).T ? No.
        # Let's do it via broadcasting
        # term_j = F[:, np.newaxis, :] * pts[np.newaxis, :, :] # shape (N, N, 2) where [i,j] is F[i,j]*p_j
        # sum_j term_j[i, j] = sum_j F[i,j] * p_j
        
        # This might be memory heavy for large N, but N=50 is fine.
        # F is 50x50. pts is 50x2.
        # term_sum = np.sum(F[:, :, np.newaxis] * pts[np.newaxis, :, :], axis=1) # Shape (N, 2)
        # This computes sum_j F[i,j] * p_j for each i.
        
        sum_F_pj = np.sum(F[:, :, np.newaxis] * pts[np.newaxis, :, :], axis=1)
        
        # sum_j F[i,j] * p_i = p_i * sum_j F[i,j]
        sum_F = np.sum(F, axis=1)
        sum_F_pi = sum_F[:, np.newaxis] * pts
        
        grad_p = sum_F_pj - sum_F_pi
        
        grad += grad_p.flatten()
        
        return total_obj, grad

    # Optimize
    # Use L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 2), res.fun
    except Exception:
        return init_pts, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Binary search for max radius r
    # Known best is ~0.1258. Let's search [0.1, 0.13]
    lo = 0.10
    hi = 0.13
    
    best_pts = None
    best_score = -1.0
    
    # We will perform a few binary search steps
    # At each step, we try to find a feasible configuration for r_mid.
    # If found, update best and move lo up.
    # If not, move hi down.
    
    # To be robust, we'll do a coarse search first, then refine.
    # But since optimization is expensive, we limit the number of feasibility checks.
    
    # Strategy:
    # 1. Try r = 0.125 (close to known best)
    # 2. If success, try slightly higher.
    # 3. If fail, try slightly lower.
    
    # Let's do a fixed number of attempts around the expected optimum.
    # Or standard binary search with limited iterations.
    
    # Since we have ~140s, and each optimization takes maybe 1-5s, we can do ~20-50 checks.
    # But L-BFGS-B with 100 vars and 500 iters is fast.
    
    # Let's do 15 iterations of binary search.
    
    for i in range(15):
        if time.time() > deadline - 10.0:
            break
            
        r_mid = (lo + hi) / 2.0
        
        # Try to find feasible config for r_mid
        # Use multiple restarts with different seeds to avoid local minima
        success = False
        best_obj_for_r = np.inf
        best_x_for_r = None
        
        # Try 3 restarts
        for seed in [0, 1, 2]:
            if time.time() > deadline - 5.0:
                break
                
            init_pts = _hex_lattice_init(N, r_mid, seed=seed)
            pts, obj = _feasibility_optimize(r_mid, init_pts, deadline=deadline)
            
            if obj < 1e-6: # Feasible
                success = True
                best_x_for_r = pts
                break # Found feasible, no need to try other seeds for this r
            else:
                if obj < best_obj_for_r:
                    best_obj_for_r = obj
                    best_x_for_r = pts
        
        if success:
            # Update best
            score = _compute_score(best_x_for_r)
            if score > best_score:
                best_score = score
                best_pts = best_x_for_r.copy()
            lo = r_mid
        else:
            hi = r_mid
            
    # Final polish: take the best found and run a direct maximization (softmin) for a bit
    # to squeeze out any remaining slack.
    if best_pts is not None and time.time() < deadline - 20.0:
        # Re-use the softmin optimizer from the original idea?
        # Or just do a few more L-BFGS steps on the feasibility objective with r = best_score?
        # Actually, if we found a feasible config for r, it might be strictly feasible.
        # We can try to increase r slightly.
        
        # Let's do a few more binary search steps with tighter tolerance or more restarts?
        # Or just return best_pts.
        
        # To improve score, we can run a "polish" phase where we maximize the min distance directly.
        # Let's implement a quick softmin polish.
        
        def _softmin_polish(points, deadline):
            # Convert to softmin objective
            x0 = points.flatten()
            
            def obj_grad(x, beta):
                pts = x.reshape(-1, 2)
                n = len(pts)
                radii = np.linalg.norm(pts, axis=1)
                boundary_dists = 1.0 - radii
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                iu = np.triu_indices(n, k=1)
                pair_dists = dists[iu] / 2.0
                all_dists = np.concatenate([pair_dists, boundary_dists])
                
                m = np.max(all_dists)
                val = m - np.log(np.sum(np.exp(-beta * (all_dists - m)))) / beta
                
                exp_vals = np.exp(-beta * (all_dists - m))
                sum_exp = np.sum(exp_vals)
                p = exp_vals / sum_exp
                
                grad = np.zeros_like(x)
                
                p_boundary = p[len(pair_dists):]
                safe_radii = np.where(radii < 1e-10, 1e-10, radii)
                boundary_grad = -p_boundary[:, np.newaxis] * (pts / safe_radii[:, np.newaxis])
                grad += boundary_grad.flatten()
                
                p_pairs = p[:len(pair_dists)]
                i_idx = iu[0]
                j_idx = iu[1]
                d_k = dists[i_idx, j_idx]
                safe_d_k = np.where(d_k < 1e-10, 1e-10, d_k)
                u = (pts[i_idx] - pts[j_idx]) / safe_d_k[:, np.newaxis]
                w_u = p_pairs[:, np.newaxis] * u
                np.add.at(grad, i_idx, w_u)
                np.add.at(grad, j_idx, -w_u)
                
                return -val, -grad

            current_x = x0
            best_x = x0.copy()
            best_val = _compute_score(x0.reshape(-1,2))
            
            # Anneal beta
            for beta in [10, 50, 200]:
                if time.time() > deadline:
                    break
                try:
                    res = minimize(
                        lambda x: obj_grad(x, beta),
                        current_x,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    current_x = res.x
                    score = _compute_score(current_x.reshape(-1,2))
                    if score > best_val:
                        best_val = score
                        best_x = current_x.copy()
                except Exception:
                    pass
                    
            return best_x.reshape(-1,2)

        polished_pts = _softmin_polish(best_pts, deadline)
        score = _compute_score(polished_pts)
        if score > best_score:
            best_score = score
            best_pts = polished_pts

    if best_pts is None:
        # Fallback
        best_pts = _hex_lattice_init(N, 0.1, seed=0)

    return best_pts

# EVOLVE-BLOCK-END
