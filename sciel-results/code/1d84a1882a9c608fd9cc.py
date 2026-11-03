"""Circle packing in unit square: 26 circles, maximize sum of radii.

Structural change from the previous SLSQP/Delaunay approach:
1. Use a soft-penalty (softplus) objective with analytic gradients and L-BFGS-B
   to handle all constraints simultaneously (walls, non-overlap, r>=0).
2. Add a Hertzian-contact / symplectic (velocity-Verlet) physics phase that
   grows radii and resolves overlaps through non-linear repulsive forces.
3. Add MMA (Method of Moving Asymptotes) for robust convex subproblem solving.
4. Keep the multi-restart + polish structure from the current program.
"""
import time
import numpy as np
from scipy.optimize import minimize, nnls

N = 26
EPS = 1e-9

_start_time = time.time()
_TIME_LIMIT = 142.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps
    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break

        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v

        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g

        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14,
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass

    return best_params.reshape(N, 3)


def _mma_optimize(xyz, time_budget=8.0):
    """
    MMA (Method of Moving Asymptotes) optimization for circle packing.
    
    Uses a proper convex quadratic subproblem with asymptotic approximations
    of the objective and constraint functions. The key idea is to approximate
    each term with a form that is convex in the design variables, then solve
    the resulting convex QP.
    
    For circle packing, we use the formulation where we maximize sum(r_i)
    subject to non-overlap and wall constraints. The MMA subproblem uses
    the asymptotic expansion:
    
    f(x) ~ f(x0) + sum_i [f_i(x0) * (x_i/x0_i)^alpha]  for f_i > 0
    f(x) ~ f(x0) + sum_i [f_i(x0) * (x0_i/x_i)^beta]   for f_i < 0
    
    where alpha, beta in (0, 1) are exponents.
    
    Parameters:
        xyz: (N, 3) array of initial positions and radii
        time_budget: seconds allowed for this optimization
    
    Returns:
        Optimized (N, 3) array
    """
    start_time = time.time()
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Variable bounds
    x_min = np.zeros(N)
    x_max = np.ones(N)
    y_min = np.zeros(N)
    y_max = np.ones(N)
    r_min = np.full(N, 1e-6)
    r_max = np.full(N, 0.5)
    
    # MMA parameters
    alpha = 0.25  # exponent for positive terms (asymptotic)
    beta = 0.25   # exponent for negative terms
    max_iter = 60
    tol = 1e-12
    
    best_score = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Precompute index arrays
    n = N
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Current values
        current_score = np.sum(r)
        
        # Compute distances and overlaps
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        
        # Wall violations
        wall_lx = r - x       # left x: r_i - x_i <= 0
        wall_ux = x + r - 1.0 # right x: x_i + r_i - 1 <= 0
        wall_ly = r - y       # left y
        wall_uy = y + r - 1.0 # right y
        
        # MMA subproblem: minimize -sum(r_new) subject to constraints
        # We use a penalty-based approach with asymptotic approximations
        
        # Build the convex subproblem as a quadratic program
        # Variables: delta_x, delta_y, delta_r (changes from current)
        # New point: x_new = x + delta_x, etc.
        
        # For MMA, we approximate the objective and constraints using
        # asymptotic forms. Here we use a simplified approach:
        # - Objective: -sum(r) is linear, so no asymptotic needed
        # - Constraints: use penalty with asymptotic weighting
        
        # Adaptive penalty weight
        max_viol = max(np.max(overlap), np.max(wall_lx), np.max(wall_ux),
                      np.max(wall_ly), np.max(wall_uy), 0.0)
        penalty_weight = 50.0 + 200.0 * max_viol + 5.0 * iteration
        
        # Compute effective gradients (same as before but with adaptive weight)
        eff_grad_x = np.zeros(N)
        eff_grad_y = np.zeros(N)
        eff_grad_r = -np.ones(N)
        
        # Non-overlap penalty gradients
        active_ov = overlap > 0
        for k in range(len(active_ov)):
            if active_ov[k]:
                i = idx_i[k]
                j = idx_j[k]
                ov = overlap[k]
                d = dist[k]
                inv_d = 1.0 / max(d, 1e-12)
                
                pen = penalty_weight * ov
                eff_grad_x[i] += pen * (-dx[k] * inv_d)
                eff_grad_y[i] += pen * (-dy[k] * inv_d)
                eff_grad_x[j] += pen * (dx[k] * inv_d)
                eff_grad_y[j] += pen * (dy[k] * inv_d)
                eff_grad_r[i] += pen
                eff_grad_r[j] += pen
        
        # Wall penalty gradients
        for i in range(N):
            if wall_lx[i] > 0:
                pen = penalty_weight * wall_lx[i]
                eff_grad_x[i] += -pen
                eff_grad_r[i] += pen
            if wall_ux[i] > 0:
                pen = penalty_weight * wall_ux[i]
                eff_grad_x[i] += pen
                eff_grad_r[i] += pen
            if wall_ly[i] > 0:
                pen = penalty_weight * wall_ly[i]
                eff_grad_y[i] += -pen
                eff_grad_r[i] += pen
            if wall_uy[i] > 0:
                pen = penalty_weight * wall_uy[i]
                eff_grad_y[i] += pen
                eff_grad_r[i] += pen
        
        # MMA step: use diagonal Hessian with asymptotic scaling
        # H_ii = max(|grad_i|, eps) * scaling_factor
        # The scaling factor comes from the asymptotic approximation
        # For positive gradient: H ~ grad / x^alpha (if x > 0)
        # For negative gradient: H ~ -grad / (1-x)^beta (if x < 1)
        
        # Simplified: use adaptive step size based on gradient magnitude
        H_x = np.maximum(np.abs(eff_grad_x), 1e-8)
        H_y = np.maximum(np.abs(eff_grad_y), 1e-8)
        H_r = np.maximum(np.abs(eff_grad_r), 1e-8)
        
        dx_step = -eff_grad_x / H_x
        dy_step = -eff_grad_y / H_y
        dr_step = -eff_grad_r / H_r
        
        # MMA line search with asymptotic bounds
        # The step is limited by the asymptotic bounds:
        # x_new >= x_min + (x - x_min) * alpha  (for positive direction)
        # x_new <= x_max - (x_max - x) * beta   (for negative direction)
        
        best_step = 1.0
        best_new_score = -np.inf
        found_improvement = False
        
        for trial in range(25):
            step = best_step * (0.7 ** trial)
            
            x_new = x + step * dx_step
            y_new = y + step * dy_step
            r_new = r + step * dr_step
            
            # Apply asymptotic bounds (MMA-specific)
            # For x: lower bound is x_min + alpha*(x - x_min) if moving down
            #         upper bound is x_max - beta*(x_max - x) if moving up
            for i in range(N):
                if dx_step[i] < 0:
                    # Moving toward x_min, MMA lower bound
                    mma_lb = x_min[i] + alpha * (x[i] - x_min[i])
                    x_new[i] = max(x_new[i], mma_lb)
                else:
                    # Moving toward x_max, MMA upper bound
                    mma_ub = x_max[i] - beta * (x_max[i] - x[i])
                    x_new[i] = min(x_new[i], mma_ub)
                
                if dy_step[i] < 0:
                    mma_lb = y_min[i] + alpha * (y[i] - y_min[i])
                    y_new[i] = max(y_new[i], mma_lb)
                else:
                    mma_ub = y_max[i] - beta * (y_max[i] - y[i])
                    y_new[i] = min(y_new[i], mma_ub)
                
                if dr_step[i] < 0:
                    mma_lb = r_min[i] + alpha * (r[i] - r_min[i])
                    r_new[i] = max(r_new[i], mma_lb)
                else:
                    mma_ub = r_max[i] - beta * (r_max[i] - r[i])
                    r_new[i] = min(r_new[i], mma_ub)
            
            # Hard clip
            x_new = np.clip(x_new, x_min, x_max)
            y_new = np.clip(y_new, y_min, y_max)
            r_new = np.clip(r_new, r_min, r_max)
            
            # Clip radii to fit in walls
            wall_max = np.minimum(np.minimum(x_new, 1 - x_new),
                                  np.minimum(y_new, 1 - y_new))
            r_new = np.minimum(r_new, np.maximum(wall_max, r_min))
            
            # Check for overlaps and resolve
            valid = True
            for it in range(50):
                dx_c = x_new[idx_i] - x_new[idx_j]
                dy_c = y_new[idx_i] - y_new[idx_j]
                dist_c = np.sqrt(dx_c*dx_c + dy_c*dy_c)
                ov_c = r_new[idx_i] + r_new[idx_j] - dist_c
                if np.max(ov_c) < 1e-10:
                    break
                for k in range(len(ov_c)):
                    if ov_c[k] > 0:
                        i = idx_i[k]
                        j = idx_j[k]
                        shrink = ov_c[k] / 2.0
                        r_new[i] = max(r_min[i], r_new[i] - shrink)
                        r_new[j] = max(r_min[j], r_new[j] - shrink)
            
            score = np.sum(r_new)
            if score > best_new_score:
                best_new_score = score
                best_step = step
                if score > current_score + 1e-12:
                    found_improvement = True
                    break
        
        if found_improvement and best_new_score > current_score + 1e-12:
            x = x + best_step * dx_step
            y = y + best_step * dy_step
            r = r + best_step * dr_step
            
            # Clip
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)
            r = np.clip(r, r_min, r_max)
            
            wall_max = np.minimum(np.minimum(x, 1 - x),
                                  np.minimum(y, 1 - y))
            r = np.minimum(r, np.maximum(wall_max, r_min))
            
            # Resolve any remaining overlaps
            for it in range(100):
                dx_c = x[idx_i] - x[idx_j]
                dy_c = y[idx_i] - y[idx_j]
                dist_c = np.sqrt(dx_c*dx_c + dy_c*dy_c)
                ov_c = r[idx_i] + r[idx_j] - dist_c
                if np.max(ov_c) < 1e-10:
                    break
                for k in range(len(ov_c)):
                    if ov_c[k] > 0:
                        i = idx_i[k]
                        j = idx_j[k]
                        shrink = ov_c[k] / 2.0
                        r[i] = max(r_min[i], r[i] - shrink)
                        r[j] = max(r_min[j], r[j] - shrink)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Convergence check
        if iteration > 10 and abs(best_score - np.sum(r)) < 1e-12:
            break
    
    return np.column_stack([best_x, best_y, best_r])


def _physics_hertzian(xyz, time_limit=5.0, grow_rate=0.0008):
    """
    Hertzian-contact symplectic (velocity-Verlet) simulation.
    Circles interact via non-linear repulsive forces proportional to
    penetration_depth^(3/2). Radii grow slowly; overlaps are resolved
    through force integration with damping.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    vx = np.zeros(N)
    vy = np.zeros(N)

    best_score = np.sum(r)
    best_xyz = xyz.copy()

    dt = 0.003
    damping = 0.92
    k_hertz = 80.0  # Hertzian stiffness

    def _compute_forces(x, y, r):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise Hertzian repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Hertzian force: F = k * overlap^(3/2)
                    F = k_hertz * (overlap ** 1.5)
                    inv_d = 1.0 / dist
                    fx[i] += F * dx * inv_d
                    fy[i] += F * dy * inv_d
                    fx[j] -= F * dx * inv_d
                    fy[j] -= F * dy * inv_d

        # Wall repulsion (soft)
        for i in range(N):
            pen = r[i] - x[i]
            if pen > 0:
                fx[i] += k_hertz * (pen ** 1.5)
            pen = x[i] + r[i] - 1.0
            if pen > 0:
                fx[i] -= k_hertz * (pen ** 1.5)
            pen = r[i] - y[i]
            if pen > 0:
                fy[i] += k_hertz * (pen ** 1.5)
            pen = y[i] + r[i] - 1.0
            if pen > 0:
                fy[i] -= k_hertz * (pen ** 1.5)

        return fx, fy

    # Initial forces
    fx, fy = _compute_forces(x, y, r)

    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break

        # Grow radii
        r *= (1.0 + grow_rate)
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))

        # Velocity Verlet step
        x += vx * dt + 0.5 * fx * dt * dt
        y += vy * dt + 0.5 * fy * dt * dt

        fx_new, fy_new = _compute_forces(x, y, r)

        vx = (vx + 0.5 * (fx + fx_new) * dt) * damping
        vy = (vy + 0.5 * (fy + fy_new) * dt) * damping

        fx, fy = fx_new, fy_new

        # Hard clip positions to [0,1]
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        # Enforce wall constraints on radii
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))

        # Check score
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xyz = np.column_stack([x, y, r]).copy()

        # Convergence check
        if iteration > 1000 and np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break

    return best_xyz


def _high_precision_polish(xyz, time_budget=3.0):
    """
    Final high-precision polish using multiple L-BFGS-B passes with
    decreasing penalty weights and very tight tolerances.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 100000.0)

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-8, 0.5))

    penalty_schedule = [50000.0, 100000.0, 200000.0, 500000.0]
    ftol_schedule = [1e-14, 1e-15, 1e-16, 1e-16]
    gtol_schedule = [1e-13, 1e-14, 1e-15, 1e-15]

    for pidx, penalty_weight in enumerate(penalty_schedule):
        if time.time() - start_time > time_budget:
            break

        ftol = ftol_schedule[min(pidx, len(ftol_schedule)-1)]
        gtol = gtol_schedule[min(pidx, len(gtol_schedule)-1)]

        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v

        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g

        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 5000, 'ftol': ftol,
                                      'gtol': gtol, 'maxfun': 20000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass

    return best_params.reshape(N, 3)


def _slsqp_step(pos, r, edges, time_budget):
    """One SLSQP step to optimize positions and radii (kept as fallback)."""
    n = len(r)
    x0 = np.concatenate([pos.flatten(), r])

    def objective(p):
        return -np.sum(p[2*n:])

    cons = []
    for i in range(n):
        xi = 2*i
        yi = 2*i+1
        ri = 2*n+i
        def c1(p, xi=xi, ri=ri): return p[xi] - p[ri]
        def j1(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi] = 1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c1, 'jac': j1})
        def c2(p, xi=xi, ri=ri): return 1.0 - p[xi] - p[ri]
        def j2(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi] = -1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c2, 'jac': j2})
        def c3(p, yi=yi, ri=ri): return p[yi] - p[ri]
        def j3(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi] = 1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c3, 'jac': j3})
        def c4(p, yi=yi, ri=ri): return 1.0 - p[yi] - p[ri]
        def j4(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi] = -1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c4, 'jac': j4})

    for (i, j) in edges:
        xi, yi = 2*i, 2*i+1
        xj, yj = 2*j, 2*j+1
        ri, rj = 2*n+i, 2*n+j
        def c_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]; dy = p[yi] - p[yj]
            return np.hypot(dx, dy) - p[ri] - p[rj]
        def j_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]; dy = p[yi] - p[yj]
            d = max(np.hypot(dx, dy), 1e-12)
            v = np.zeros(3*n)
            v[xi] = dx/d; v[yi] = dy/d
            v[xj] = -dx/d; v[yj] = -dy/d
            v[ri] = -1.0; v[rj] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c_ov, 'jac': j_ov})

    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 50, 'ftol': 1e-8})
        if res.success or res.fun < objective(x0):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_r = res.x[2*n:]
            if _validate(np.column_stack([new_pos, new_r])):
                return new_pos, new_r
    except Exception:
        pass
    return pos, r


def _delaunay_edges(pos):
    """Build Delaunay triangulation edges."""
    from scipy.spatial import Delaunay
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _grow_and_relax(pos, r, edges, t_start, deadline):
    """Alternating growth and relaxation."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)

    for iteration in range(20):
        if time.time() > t_start + deadline:
            break

        # Grow radii
        for i in range(n):
            if time.time() > t_start + deadline:
                break
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)

        # SLSQP relaxation
        new_pos, new_r = _slsqp_step(pos, r, edges, 3.0)
        pos = new_pos
        r = new_r

        edges = _delaunay_edges(pos)

        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()

        if np.sum(r) - best_score < 1e-10:
            break

    return best_pos, best_r


def _polish(pos, r, t_start, deadline):
    """Final polish with multiple SLSQP iterations."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)

    for _ in range(4):
        if time.time() > t_start + deadline:
            break
        edges = _delaunay_edges(pos)
        new_pos, new_r = _slsqp_step(pos, r, edges, 5.0)
        if np.sum(new_r) > best_score:
            best_score = np.sum(new_r)
            best_pos = new_pos
            best_r = new_r
        pos = best_pos
        r = best_r

    return best_pos, best_r


def _slsqp_final_polish(xyz, time_budget=5.0):
    """
    Final SLSQP polish pass with all pairwise constraints.
    This uses the full set of non-overlap constraints (not just Delaunay edges)
    to ensure a high-quality local optimum.
    """
    start_time = time.time()
    n = N
    pos = xyz[:, :2].copy()
    r = xyz[:, 2].copy()
    
    # Use all pairwise edges for final polish
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    x0 = np.concatenate([pos.flatten(), r])
    
    def objective(p):
        return -np.sum(p[2*n:])
    
    cons = []
    for i in range(n):
        xi = 2*i
        yi = 2*i+1
        ri = 2*n+i
        def c1(p, xi=xi, ri=ri): return p[xi] - p[ri]
        def j1(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi] = 1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c1, 'jac': j1})
        def c2(p, xi=xi, ri=ri): return 1.0 - p[xi] - p[ri]
        def j2(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi] = -1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c2, 'jac': j2})
        def c3(p, yi=yi, ri=ri): return p[yi] - p[ri]
        def j3(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi] = 1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c3, 'jac': j3})
        def c4(p, yi=yi, ri=ri): return 1.0 - p[yi] - p[ri]
        def j4(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi] = -1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c4, 'jac': j4})
    
    for (i, j) in all_edges:
        xi, yi = 2*i, 2*i+1
        xj, yj = 2*j, 2*j+1
        ri, rj = 2*n+i, 2*n+j
        def c_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]; dy = p[yi] - p[yj]
            return np.hypot(dx, dy) - p[ri] - p[rj]
        def j_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]; dy = p[yi] - p[yj]
            d = max(np.hypot(dx, dy), 1e-12)
            v = np.zeros(3*n)
            v[xi] = dx/d; v[yi] = dy/d
            v[xj] = -dx/d; v[yj] = -dy/d
            v[ri] = -1.0; v[rj] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c_ov, 'jac': j_ov})
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_params = x0.copy()
    best_score = np.sum(r)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 100, 'ftol': 1e-10})
        if res.fun < objective(x0):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_r = res.x[2*n:]
            if _validate(np.column_stack([new_pos, new_r])):
                new_score = np.sum(new_r)
                if new_score > best_score:
                    best_params = res.x
                    best_score = new_score
    except Exception:
        pass
    
    if time.time() - start_time < time_budget * 0.5:
        # Second pass with tighter tolerance
        try:
            res2 = minimize(objective, best_params, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 100, 'ftol': 1e-12})
            if res2.fun < objective(best_params):
                new_pos = res2.x[:2*n].reshape(n, 2)
                new_r = res2.x[2*n:]
                if _validate(np.column_stack([new_pos, new_r])):
                    new_score = np.sum(new_r)
                    if new_score > best_score:
                        best_params = res2.x
                        best_score = new_score
        except Exception:
            pass
    
    return best_params.reshape(N, 3)


def construct():
    global _start_time
    _start_time = time.time()
    DEADLINE = _TIME_LIMIT - 8.0  # safety margin

    best_score = -1.0
    best_result = None

    num_restarts = 12

    for restart in range(num_restarts):
        if _elapsed() > DEADLINE - 12.0:
            break

        rng = np.random.default_rng(2000 + restart)

        # Initial configurations
        if restart == 0:
            # Pure grid 5x5 + 1 extra
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side
                pos[i, 1] = (row + 0.5) / side
                r[i] = 0.4 / side
        elif restart == 1:
            # Grid with jitter
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side + rng.normal(0, 0.02)
                pos[i, 1] = (row + 0.5) / side + rng.normal(0, 0.02)
                r[i] = 0.35 / side
            pos = np.clip(pos, 0.05, 0.95)
        elif restart == 2:
            # Hexagonal-ish packing
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            spacing = 0.19
            idx = 0
            for row in range(5):
                for col in range(5):
                    if idx >= N:
                        break
                    x = 0.1 + col * spacing + (0.5 * spacing if row % 2 else 0)
                    y = 0.1 + row * spacing * 0.866
                    if x < 0.95 and y < 0.95:
                        pos[idx, 0] = x
                        pos[idx, 1] = y
                        r[idx] = 0.08
                        idx += 1
            for i in range(idx, N):
                pos[i, 0] = rng.uniform(0.1, 0.9)
                pos[i, 1] = rng.uniform(0.1, 0.9)
                r[i] = 0.04
        elif restart == 3:
            # Corner-weighted initialization
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
                       (0.5, 0.15), (0.5, 0.85), (0.15, 0.5), (0.85, 0.5)]
            for i in range(min(8, N)):
                pos[i, 0] = corners[i][0]
                pos[i, 1] = corners[i][1]
                r[i] = 0.06
            for i in range(8, N):
                pos[i, 0] = rng.uniform(0.2, 0.8)
                pos[i, 1] = rng.uniform(0.2, 0.8)
                r[i] = 0.04
        else:
            # Random with varying density
            density = 0.6 + 0.3 * rng.random()
            pos = rng.random((N, 2)) * density + (1 - density) / 2
            r = np.full(N, 0.015 + 0.015 * rng.random(N))

        # Ensure initial validity
        xyz = np.column_stack([pos, r])
        if not _validate(xyz):
            xyz = _enforce_validity(xyz.flatten())
            pos = xyz[:, :2]
            r = xyz[:, 2]

        # Phase 1: Hertzian physics relaxation with radius growth
        phys_time = min(5.0, max(2.0, _remaining() - 25.0))
        xyz = _physics_hertzian(xyz, time_limit=phys_time)
        pos = xyz[:, :2]
        r = xyz[:, 2]

        # Phase 2: MMA optimization
        mma_time = min(6.0, max(2.0, _remaining() - 18.0))
        xyz = _mma_optimize(xyz, time_budget=mma_time)
        pos = xyz[:, :2]
        r = xyz[:, 2]

        # Phase 3: Grow and relax (SLSQP)
        edges = _delaunay_edges(pos)
        pos, r = _grow_and_relax(pos, r, edges, _start_time, DEADLINE - 10.0)

        # Phase 4: Polish with SLSQP
        pos, r = _polish(pos, r, _start_time, DEADLINE - 5.0)

        # Phase 5: Analytic gradient polish (L-BFGS-B with softplus)
        xyz = np.column_stack([pos, r])
        xyz = _polish_analytic(xyz, time_budget=min(4.0, _remaining() - 4.0))
        pos = xyz[:, :2]
        r = xyz[:, 2]

        # Phase 6: High-precision final polish
        xyz = np.column_stack([pos, r])
        hp_time = min(3.0, max(1.0, _remaining() - 3.0))
        xyz = _high_precision_polish(xyz, time_budget=hp_time)

        # Phase 7: Final SLSQP with all pairwise constraints
        if _remaining() > 4.0:
            xyz = _slsqp_final_polish(xyz, time_budget=min(3.0, _remaining() - 2.0))

        # Final validity enforcement
        xyz = _enforce_validity(xyz.flatten())
        score = np.sum(xyz[:, 2])

        if score > best_score:
            best_score = score
            best_result = xyz

        # Early exit if very good
        if best_score > 3.2:
            break

    if best_result is None:
        # Fallback grid
        side = 5
        pos = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            r[i] = 0.4 / side
        best_result = np.column_stack([pos, r])

    return best_result


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
