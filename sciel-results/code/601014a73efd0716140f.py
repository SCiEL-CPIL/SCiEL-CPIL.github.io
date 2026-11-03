import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _eval_packing(xy, r, tol=1e-9):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -tol:
            return 0.0, False
        if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return 0.0, False
    return float(np.sum(r)), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r

def _best_fit_local_search(xy, r, deadline, grid_size=100):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    This is the structural core from Inspiration 1.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 50
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

            # Refinement: if we moved, do a finer local search around the new position
            if new_r > old_r + 1e-10:
                # Create a finer grid around the best position
                step = 1.0 / grid_size
                fine_gx = np.linspace(max(0, new_x - 3*step), min(1, new_x + 3*step), 30)
                fine_gy = np.linspace(max(0, new_y - 3*step), min(1, new_y + 3*step), 30)
                FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
                fine_candidates = np.column_stack([FGX.ravel(), FGY.ravel()])
                FM = len(fine_candidates)
                
                if FM > 0:
                    fine_wall_clear = np.minimum(np.minimum(fine_candidates[:, 0], 1.0 - fine_candidates[:, 0]),
                                                 np.minimum(fine_candidates[:, 1], 1.0 - fine_candidates[:, 1]))
                    if len(others_xy) > 0:
                        fdx = fine_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                        fdy = fine_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                        fdists = np.sqrt(fdx * fdx + fdy * fdy)
                        fcircle_clear = np.min(fdists - others_r[np.newaxis, :], axis=1)
                        fr_candidates = np.minimum(fine_wall_clear, fcircle_clear)
                    else:
                        fr_candidates = fine_wall_clear.copy()
                    
                    fidx = int(np.argmax(fr_candidates))
                    fnew_x, fnew_y = fine_candidates[fidx, 0], fine_candidates[fidx, 1]
                    fnew_r = fr_candidates[fidx]
                    
                    if fnew_r > new_r + 1e-10:
                        xy[i, 0] = fnew_x
                        xy[i, 1] = fnew_y
                        r[i] = fnew_r

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid

def _smooth_objective(v, n):
    """
    Smooth objective: minimize -sum(r) + penalty for violations.
    v is a vector of length 3*n: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Uses smooth max via softplus for wall and pairwise violations.
    This is the refinement machinery from Inspiration 1.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall violations: r > min(x, 1-x, y, 1-y)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min  # can be negative
    # Smooth max(0, viol) via softplus: log(1 + exp(beta*viol)) / beta
    beta = 50.0
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    # Pairwise violations: r_i + r_j > dist_ij
    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    wall_pen_sum = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    return -np.sum(r) + 100.0 * (wall_pen_sum + total_viol)

def _smooth_grad(v, n):
    """
    Gradient of the smooth objective using softplus derivatives.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 50.0

    # Wall terms: softplus(beta * (r - wall_min)) / beta
    # d/d(r) = 100 * sigmoid(beta * (r - wall_min))
    # d/d(x) = 100 * sigmoid(...) * (-d(wall_min)/dx)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))  # sigmoid(beta * wall_viol)

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        # Determine which wall is binding (closest)
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = np.argmin(dists)
        
        # Gradient contribution from wall penalty
        # Wall penalty term: 100 * log(1+exp(beta*viol))/beta
        # d/dv = 100 * sigmoid(beta*viol) * beta * d(viol)/dv / beta = 100 * sigmoid * d(viol)/dv
        
        if min_idx == 0: # x
            # viol = r - x
            # d(viol)/dx = -1
            grad[i] += 100.0 * s * (-1.0)
            grad[2*n + i] += 100.0 * s * (1.0)
        elif min_idx == 1: # 1-x
            # viol = r - (1-x) = r - 1 + x
            # d(viol)/dx = 1
            grad[i] += 100.0 * s * (1.0)
            grad[2*n + i] += 100.0 * s * (1.0)
        elif min_idx == 2: # y
            # viol = r - y
            # d(viol)/dy = -1
            grad[n + i] += 100.0 * s * (-1.0)
            grad[2*n + i] += 100.0 * s * (1.0)
        elif min_idx == 3: # 1-y
            # viol = r - (1-y) = r - 1 + y
            # d(viol)/dy = 1
            grad[n + i] += 100.0 * s * (1.0)
            grad[2*n + i] += 100.0 * s * (1.0)

    # Objective: -sum(r)
    grad[2*n:3*n] += -1.0

    # Pairwise terms
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
            viol = r[i] + r[j] - dist
            sig = 1.0 / (1.0 + np.exp(-beta * viol))
            
            # d(viol)/dx_i = -dx/dist
            # d(viol)/dx_j = dx/dist
            # d(viol)/dy_i = -dy/dist
            # d(viol)/dy_j = dy/dist
            # d(viol)/dr_i = 1
            # d(viol)/dr_j = 1
            
            # Gradient term: 100 * sig * d(viol)/dv
            g_xi = 100.0 * sig * (-dx / dist)
            g_xj = 100.0 * sig * (dx / dist)
            g_yi = 100.0 * sig * (-dy / dist)
            g_yj = 100.0 * sig * (dy / dist)
            g_ri = 100.0 * sig * (1.0)
            g_rj = 100.0 * sig * (1.0)
            
            grad[i] += g_xi
            grad[j] += g_xj
            grad[n + i] += g_yi
            grad[n + j] += g_yj
            grad[2*n + i] += g_ri
            grad[2*n + j] += g_rj

    return grad

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill

    # 1. Generate multiple diverse initializations
    inits = []
    
    # Grid init
    grid_xy, grid_r = _grid_init(5)
    inits.append((grid_xy, grid_r))
    
    # Random inits
    for seed in range(10):
        xy, r = _random_init(N, seed=seed)
        inits.append((xy, r))

    best_total = 0.0
    best_xy = None
    best_r = None
    best_valid = False

    # 2. For each init, run local search to get a good local optimum
    for xy, r in inits:
        if time.time() > timeout:
            break
        
        # Local search
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(xy, r, timeout)
        
        if ls_valid and ls_total > best_total:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r
            best_valid = True

    if not best_valid:
        # Fallback to grid
        grid_xy, grid_r = _grid_init(5)
        return np.column_stack([grid_xy, grid_r])

    # 3. Refine the best solution with smooth gradient-based optimization
    # This helps "squeeze" the circles tighter and adjust radii slightly
    # to maximize the sum, using the smooth penalty to handle constraints.
    
    # We need to map from (xy, r) to v
    v = np.concatenate([best_xy.flatten(), best_r.flatten()])
    
    # Run several restarts of the optimizer from slightly perturbed best solutions
    for restart in range(5):
        if time.time() > timeout:
            break
            
        v_start = v.copy()
        if restart > 0:
            rng = np.random.RandomState(restart)
            v_start += rng.normal(0, 0.01, size=v_start.shape)
        
        try:
            res = minimize(
                _smooth_objective,
                v_start,
                args=(N,),
                jac=_smooth_grad,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N) + [(1e-6, 0.5)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            v_opt = res.x
            xy_opt = v_opt[:2*N].reshape(N, 2)
            r_opt = v_opt[2*N:3*N]
            
            total, valid = _eval_packing(xy_opt, r_opt)
            if valid and total > best_total:
                best_total = total
                best_xy = xy_opt
                best_r = r_opt
                best_valid = True
                v = v_opt # Update starting point for next restart
        except Exception:
            pass

    # 4. Final local search polish on the best solution found
    if best_valid:
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(best_xy, best_r, timeout, grid_size=150)
        if ls_valid and ls_total > best_total:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r

    # Construct final output
    result = np.column_stack([best_xy, best_r])
    
    # Final sanity check and fix
    # Ensure no negative radii
    result[:, 2] = np.maximum(result[:, 2], 1e-6)
    
    # If still invalid (shouldn't happen), return a known valid grid
    total, valid = _eval_packing(result[:, :2], result[:, 2])
    if not valid:
        grid_xy, grid_r = _grid_init(5)
        result = np.column_stack([grid_xy, grid_r])

    return result
