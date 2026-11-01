"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    n = len(centers)
    # Wall distances
    wall_dists = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _violation_objective(x, r):
    """Minimize violation of radius r."""
    centers = x.reshape(N, 2)
    
    # Wall distances
    wall_dists = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    wall_viol = np.maximum(0, r - wall_dists)
    wall_cost = np.sum(wall_viol**2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    pair_viol = np.maximum(0, 2*r - pair_dists)
    pair_cost = np.sum(pair_viol**2)
    
    return pair_cost + wall_cost

def _violation_grad(x, r):
    """Gradient of the violation objective."""
    centers = x.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    # Wall gradient
    wall_dists = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    wall_viol = np.maximum(0, r - wall_dists)
    
    # For each coordinate, compute sign based on which wall is closer
    # x-wall dist: 0.5 - |x|
    # If x >= 0: dist = 0.5 - x, d(dist)/dx = -1, so -d(dist)/dx = 1
    # If x < 0: dist = 0.5 + x, d(dist)/dx = 1, so -d(dist)/dx = -1
    sign_x = np.where(centers[:, 0] >= 0, 1.0, -1.0)
    sign_y = np.where(centers[:, 1] >= 0, 1.0, -1.0)
    
    # But we need to use the minimum of the two wall distances
    # This is tricky. Let's compute both and take the min.
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    
    # The wall distance is min(wx, wy)
    # If wx < wy, then the x-wall is binding, gradient wrt x is sign_x
    # If wy < wx, then the y-wall is binding, gradient wrt y is sign_y
    # If equal, both contribute
    
    # Actually, the objective uses wall_dists = min(wx, wy) per circle
    # But in the original code, it was using column_stack and then min, which gives
    # two separate wall violations (one for x, one for y), not the min.
    # Let me re-read the original...
    
    # Original code:
    # wall_dists = np.column_stack((0.5 - |x|, 0.5 - |y|))
    # wall_viol = np.maximum(0, r - wall_dists)
    # wall_cost = np.sum(wall_viol**2)
    
    # So it's treating x-wall and y-wall as separate constraints!
    # This means a circle can be close to the x-wall AND the y-wall,
    # and both violations are counted.
    
    # So the gradient is simpler:
    # For x-coord: if 0.5 - |x| < r, violation = r - (0.5 - |x|)
    # d(violation^2)/dx = 2*violation * d(violation)/dx
    # d(violation)/dx = -d(0.5 - |x|)/dx = -(-sign(x)) = sign(x)
    
    wx_viol = np.maximum(0, r - (0.5 - np.abs(centers[:, 0])))
    wy_viol = np.maximum(0, r - (0.5 - np.abs(centers[:, 1])))
    
    grad[:, 0] += 2 * wx_viol * sign_x
    grad[:, 1] += 2 * wy_viol * sign_y
    
    # Pair gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2*r - pair_dists)
    
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        viol = pair_viol[k]
        if viol == 0:
            continue
        d_ij = pair_dists[k]
        vec = centers[i] - centers[j]
        grad_vec = vec / d_ij
        
        grad[i] -= 2 * viol * grad_vec
        grad[j] += 2 * viol * grad_vec
    
    return grad.flatten()

def _construct_initial_grid():
    """Construct a good initial grid packing."""
    # Use a hexagonal-like pattern or just a grid
    # For 50 circles, a 7x7 grid with 2 extra
    m = 7
    spacing = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            if len(centers) < N:
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                centers.append([x, y])
    return np.array(centers[:N])

def _construct_hexagonal():
    """Construct a hexagonal packing initial guess."""
    centers = []
    # Hexagonal packing: rows offset by half spacing
    # For N=50, we can use about 7-8 rows
    r_est = 0.07
    dx = 2 * r_est
    dy = dx * np.sqrt(3) / 2
    
    # Try to fit as many as possible
    for row in range(10):
        y = -0.45 + row * dy
        if y > 0.45:
            break
        offset = 0 if row % 2 == 0 else dx / 2
        for col in range(10):
            if len(centers) >= N:
                break
            x = -0.45 + offset + col * dx
            if x < 0.45:
                centers.append([x, y])
    
    # If we don't have enough, add more
    while len(centers) < N:
        centers.append([0.0, 0.0])
    
    return np.array(centers[:N])

def _soft_min_objective(x, beta=50.0):
    """Soft-min objective for direct maximization of minimum distance."""
    centers = x.reshape(N, 2)
    
    # Wall distances (both x and y walls)
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    all_vals = np.concatenate([pair_dists, wx, wy])
    
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp < 1e-300:
        sm = min_val
    else:
        sm = min_val - (1.0 / beta) * np.log(sum_exp)
    
    return -sm

def _soft_min_grad(x, beta=50.0):
    """Gradient of soft-min objective."""
    centers = x.reshape(N, 2)
    
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    all_vals = np.concatenate([pair_dists, wx, wy])
    
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        return np.zeros_like(x)
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Pair distances gradient
    # d(pair_dist)/d(centers[i]) = (centers[i] - centers[j]) / pair_dist
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        w = weights[k]
        if w < 1e-10:
            continue
        d_ij = pair_dists[k]
        vec = centers[i] - centers[j]
        grad_vec = vec / d_ij
        
        # soft_min = min_val - 1/beta * log(sum_exp)
        # d(soft_min)/d(centers[i]) = -weights[k] * d(pair_dist)/d(centers[i])
        # because min_val is piecewise constant (ignore its gradient for simplicity)
        # Actually, this is an approximation. The true gradient involves the min_val derivative.
        # Let's use a simpler approach: just use the weighted average of gradients.
        
        grad[i] -= w * grad_vec
        grad[j] += w * grad_vec
    
    # Wall gradients
    sign_x = np.where(centers[:, 0] >= 0, 1.0, -1.0)
    sign_y = np.where(centers[:, 1] >= 0, 1.0, -1.0)
    
    # wx index starts at N*(N-1)/2
    n_pairs = len(iu[0])
    wx_weights = weights[n_pairs:n_pairs+N]
    wy_weights = weights[n_pairs+N:n_pairs+2*N]
    
    grad[:, 0] -= wx_weights * sign_x  # d(wx)/dx = -sign_x, so -w * (-sign_x) = w * sign_x... wait
    # soft_min derivative: d(soft_min)/dx = -sum(w_i * d(v_i)/dx)
    # d(wx)/dx = -sign_x
    # So contribution is -w_x * (-sign_x) = w_x * sign_x
    # But we're minimizing -soft_min, so gradient is sum(w_i * d(v_i)/dx)
    # d(v_x)/dx = d(wx)/dx = -sign_x
    # So grad[:, 0] += wx_weights * (-sign_x)
    
    grad[:, 0] = wx_weights * (-sign_x)
    grad[:, 1] = wy_weights * (-sign_y)
    
    # Add pair contributions
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        w = weights[k]
        if w < 1e-10:
            continue
        d_ij = pair_dists[k]
        vec = centers[i] - centers[j]
        grad_vec = vec / d_ij
        
        grad[i] += w * grad_vec
        grad[j] -= w * grad_vec
    
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    # Initial configurations
    grid_centers = _construct_initial_grid()
    hex_centers = _construct_hexagonal()
    
    best_centers = grid_centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try hexagonal as well
    hex_radius = _compute_radius(hex_centers)
    if hex_radius > best_radius:
        best_radius = hex_radius
        best_centers = hex_centers.copy()
    
    # Strategy 1: Binary search on radius with violation minimization
    r_low = best_radius
    r_high = 0.08
    
    current_centers = best_centers.copy()
    
    # Do a few binary search steps
    for step in range(12):
        if time.time() > deadline - 10.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        x0 = current_centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = opt.minimize(
                _violation_objective,
                x0,
                jac=_violation_grad,
                args=(r_mid,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_res = res.x
            r_actual = _compute_radius(x_res.reshape(N, 2))
            
            if r_actual >= r_mid - 1e-7:
                r_low = r_mid
                if r_actual > best_radius:
                    best_radius = r_actual
                    best_centers = x_res.reshape(N, 2).copy()
                current_centers = x_res.reshape(N, 2).copy()
            else:
                r_high = r_mid
                # Use result as starting point for next iteration
                current_centers = x_res.reshape(N, 2).copy()
                
        except Exception:
            r_high = r_mid
    
    # Strategy 2: Direct soft-min optimization from best point
    if time.time() < deadline - 5.0:
        try:
            res = opt.minimize(
                _soft_min_objective,
                best_centers.flatten(),
                jac=_soft_min_grad,
                args=(100.0,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_res = res.x
            r_res = _compute_radius(x_res.reshape(N, 2))
            if r_res > best_radius:
                best_radius = r_res
                best_centers = x_res.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with local optimization
    if time.time() < deadline - 5.0:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 3.0:
                break
            
            # Random perturbation of best solution
            x0 = best_centers.copy()
            x0 += np.random.randn(N, 2) * 0.01
            x0 = np.clip(x0, -0.49, 0.49)
            
            try:
                res = opt.minimize(
                    _soft_min_objective,
                    x0.flatten(),
                    jac=_soft_min_grad,
                    args=(100.0,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                x_res = res.x
                r_res = _compute_radius(x_res.reshape(N, 2))
                if r_res > best_radius:
                    best_radius = r_res
                    best_centers = x_res.reshape(N, 2).copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 2.0:
        try:
            res = opt.minimize(
                _soft_min_objective,
                best_centers.flatten(),
                jac=_soft_min_grad,
                args=(200.0,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            x_res = res.x
            r_res = _compute_radius(x_res.reshape(N, 2))
            if r_res > best_radius:
                best_radius = r_res
                best_centers = x_res.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    if not np.all(np.isfinite(best_centers)):
        return _construct_initial_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
