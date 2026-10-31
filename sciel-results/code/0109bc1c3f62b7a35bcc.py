"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 26

# EVOLVE-BLOCK-START

def _objective_and_grads(params):
    """
    Objective function and gradient for L-BFGS-B.
    params: array of length 3*N -> [x0, y0, r0, x1, y1, r1, ...]
    Minimize: -sum(r) + penalty(overlaps) + penalty(boundary)
    """
    pos = params[0::3].reshape(N, 2)
    radii = params[1::3].reshape(N)
    
    # Ensure radii are non-negative for calculation stability in penalty, 
    # though L-BFGS-B handles bounds, we clip for safety in distance calc.
    r_eff = np.maximum(radii, 0.0)
    
    # 1. Boundary Penalty (Quadratic)
    # x - r < 0 -> max(0, r - x)^2
    # x + r > 1 -> max(0, x + r - 1)^2
    bv_left = np.maximum(0, r_eff - pos[:, 0])
    bv_right = np.maximum(0, pos[:, 0] + r_eff - 1.0)
    bv_bottom = np.maximum(0, r_eff - pos[:, 1])
    bv_top = np.maximum(0, pos[:, 1] + r_eff - 1.0)
    
    boundary_pen = np.sum(bv_left**2) + np.sum(bv_right**2) + np.sum(bv_bottom**2) + np.sum(bv_top**2)
    
    # 2. Overlap Penalty (Quadratic)
    # d_ij < r_i + r_j -> max(0, r_i + r_j - d_ij)^2
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    # Avoid division by zero in gradient calculation later if we do analytic, 
    # but here we just need value.
    dist = np.sqrt(dist_sq + 1e-12) # epsilon for stability
    
    min_dist = r_eff[:, np.newaxis] + r_eff[np.newaxis, :]
    viol = np.maximum(0, min_dist - dist)
    
    # Mask for i < j to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(viol[mask] ** 2)
    
    penalty_weight = 100.0
    value = -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)
    
    # --- Analytical Gradients ---
    # We need d(value)/d(params)
    # params order: x0, y0, r0, x1, y1, r1 ...
    grads = np.zeros_like(params)
    
    # Gradient w.r.t radii:
    # d(-sum(r))/dr_i = -1
    # d(boundary_pen)/dr_i:
    # if r_i > x_i: d/dr_i (r_i - x_i)^2 = 2(r_i - x_i)
    # if r_i > y_i: 2(r_i - y_i)
    # if x_i + r_i > 1: 2(x_i + r_i - 1)
    # if y_i + r_i > 1: 2(y_i + r_i - 1)
    
    # Gradient w.r.t positions:
    # d(boundary_pen)/dx_i:
    # if r_i > x_i: -2(r_i - x_i)
    # if x_i + r_i > 1: 2(x_i + r_i - 1)
    # similarly for y
    
    # d(overlap_pen)/dx_i:
    # Sum over j != i of d/dx_i max(0, r_i+r_j - d_ij)^2
    # = Sum 2 * max(0, ...) * d/dx_i (r_i+r_j - d_ij)
    # d/dx_i (r_i+r_j - d_ij) = - d/dx_i d_ij = - (x_i - x_j) / d_ij
    
    # Let's compute gradients efficiently.
    
    # 1. Radii gradients
    # Base -1
    grads[1::3] = -1.0
    
    # Boundary contributions to radius grads
    grads[1::3] += penalty_weight * (
        2 * bv_left + 2 * bv_right + 2 * bv_bottom + 2 * bv_top
    )
    
    # Overlap contributions to radius grads
    # For each pair (i,j) where viol > 0:
    # d/dr_i (r_i+r_j-d)^2 = 2(r_i+r_j-d)
    # d/dr_j (r_i+r_j-d)^2 = 2(r_i+r_j-d)
    # We sum these.
    # viol is (N,N). mask is i<j.
    # We need to scatter the contributions back to the radius indices.
    # Create a zero array for radius overlap grads
    rad_grads_overlap = np.zeros(N)
    
    # Only where viol > 0
    active_viol = viol * mask
    # Sum contributions for each i
    # rad_grads_overlap[i] += sum_j 2 * active_viol[i,j]
    # rad_grads_overlap[j] += sum_i 2 * active_viol[i,j]
    
    # Using matrix multiplication or sum:
    # sum over j for each i: np.sum(active_viol, axis=1)
    # sum over i for each j: np.sum(active_viol, axis=0)
    rad_grads_overlap += 2 * np.sum(active_viol, axis=1)
    rad_grads_overlap += 2 * np.sum(active_viol, axis=0)
    
    grads[1::3] += penalty_weight * rad_grads_overlap
    
    # 2. Position gradients
    # Boundary contributions to x grads
    grads[0::3] += penalty_weight * (
        -2 * bv_left + 2 * bv_right
    )
    # Boundary contributions to y grads
    grads[2::3] += penalty_weight * (
        -2 * bv_bottom + 2 * bv_top
    )
    
    # Overlap contributions to position grads
    # d/dx_i overlap_pen = Sum_{j!=i} 2 * max(0, r_i+r_j-d_ij) * (-(x_i-x_j)/d_ij)
    # Let V_ij = max(0, r_i+r_j-d_ij)
    # Term for x_i: -2 * V_ij * (x_i-x_j) / d_ij
    # Note: V_ij is symmetric.
    
    # We can compute the force vector F_i = Sum_j F_ij
    # F_ij_x = -2 * V_ij * (x_i-x_j) / d_ij
    # F_ij_y = -2 * V_ij * (y_i-y_j) / d_ij
    
    # Efficient calculation:
    # V is (N,N). We only care about V > 0.
    # Let W = V * mask (upper triangle only, to avoid double counting in sum? 
    # Actually, the gradient for x_i involves sum over ALL j.
    # If we use full V, we sum over all j.
    # d/dx_i Sum_{j} V_ij^2 = Sum_j 2 V_ij dV_ij/dx_i
    # V_ij = max(0, r_i+r_j-d_ij)
    # dV_ij/dx_i = - (x_i-x_j)/d_ij if V_ij > 0 else 0
    # So Grad_x_i = Sum_j 2 V_ij * (-(x_i-x_j)/d_ij)
    
    # We can vectorize this.
    # Create matrix D = V * mask? No, V is symmetric if we define it properly.
    # In our calculation, viol is symmetric (min_dist - dist is symmetric).
    # mask was used to prevent double counting in the *value* (summing only i<j).
    # For the *gradient*, we must sum over all j != i.
    # So we should use the full V matrix, but set diagonal to 0.
    # And we must be careful: V_ij = V_ji.
    # Grad_x_i = -2 * Sum_{j != i} V_ij * (x_i - x_j) / d_ij
    
    # Let's compute the full interaction matrix.
    # d is (N,N) symmetric.
    # V_full = np.maximum(0, min_dist - dist)
    # np.fill_diagonal(V_full, 0)
    
    # We need to compute:
    # Gx_i = -2 * Sum_j V_ij * (x_i - x_j) / d_ij
    # Gy_i = -2 * Sum_j V_ij * (y_i - y_j) / d_ij
    
    # To avoid division by zero, d_ij is sqrt(dx^2+dy^2+eps).
    # If d_ij is very small, the gradient is unstable. 
    # However, if d_ij is small, V_ij is likely large (if radii > 0).
    # The term (x_i-x_j)/d_ij is the unit vector.
    
    # Let's compute unit vectors.
    # ux = dx / dist
    # uy = dy / dist
    # dx = x_i - x_j (broadcasted)
    # dist = sqrt(dx^2+dy^2+eps)
    
    # ux = dx / dist
    # uy = dy / dist
    
    # Gx_i = -2 * Sum_j V_ij * ux_ij
    # Gy_i = -2 * Sum_j V_ij * uy_ij
    
    # Note: V_ij is zero if no overlap.
    
    V_full = np.maximum(0, min_dist - dist)
    np.fill_diagonal(V_full, 0)
    
    # ux, uy are (N,N)
    ux = dx / dist
    uy = dy / dist
    
    # Sum over j (axis=1)
    Gx = -2.0 * np.sum(V_full * ux, axis=1)
    Gy = -2.0 * np.sum(V_full * uy, axis=1)
    
    grads[0::3] += penalty_weight * Gx
    grads[2::3] += penalty_weight * Gy
    
    return value, grads

def _enforce_validity(pos, radii):
    """
    Greedily shrink radii to ensure validity.
    Returns valid pos, radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # 1. Boundary constraints
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    # 2. Overlap constraints
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-10:
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = 0.0
                        radii[j] = 0.0
                    changed = True
        if not changed:
            break
            
    # Final safety clip
    for i in range(N):
        max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        if radii[i] > max_r:
            radii[i] = max_r
            
    return pos, radii

def _is_valid(pos, radii, tol=1e-8):
    """Check if configuration is strictly valid."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist = np.sqrt(dist_sq)
    if np.any(dist[mask] < min_dist[mask] - tol):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Strategy: Multi-start L-BFGS-B
    # 1. Generate diverse initial configurations
    # 2. Run L-BFGS-B on each
    # 3. Enforce validity and check score
    
    n_starts = 20
    time_per_start = (deadline - start_time) / (n_starts * 2) # Leave time for final check
    
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if s == 0:
            # Grid init
            pos = np.zeros((N, 2))
            radii = np.zeros(N)
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N:
                        pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                        radii[k] = 0.09
                        k += 1
            if k < N:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
        else:
            # Random init
            pos = rng.uniform(0.1, 0.9, (N, 2))
            radii = rng.uniform(0.05, 0.12, N)
            # Enforce validity immediately to get a valid start
            pos, radii = _enforce_validity(pos, radii)
            
        # Flatten for L-BFGS-B
        params = np.zeros(3 * N)
        params[0::3] = pos[:, 0]
        params[1::3] = pos[:, 1]
        params[2::3] = radii
        
        # Bounds:
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible radius in unit square is 0.5)
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # Run L-BFGS-B
        try:
            # maxiter to prevent hanging on a single start
            max_iter = 200
            res = minimize(
                _objective_and_grads, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception as e:
            continue
            
        if res.success or res.fun < 1e6: # Check if it produced something reasonable
            new_params = res.x
            new_pos = np.zeros((N, 2))
            new_radii = np.zeros(N)
            new_pos[:, 0] = new_params[0::3]
            new_pos[:, 1] = new_params[1::3]
            new_radii = new_params[2::3]
            
            # Enforce validity (L-BFGS-B minimizes penalty, might leave slight overlaps)
            new_pos, new_radii = _enforce_validity(new_pos, new_radii)
            
            if _is_valid(new_pos, new_radii):
                score = np.sum(new_radii)
                if score > best_score:
                    best_score = score
                    best_pos = new_pos
                    best_radii = new_radii
                    
        # Early stop if very good
        if best_score > 2.4:
            break

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        best_pos, best_radii = _enforce_validity(pos, radii)
        best_pos, best_radii = _enforce_validity(best_pos, best_radii) # double check

    # Final strict validation
    best_pos, best_radii = _enforce_validity(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
