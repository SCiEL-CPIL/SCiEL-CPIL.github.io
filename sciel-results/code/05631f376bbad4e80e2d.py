"""Circle packing in perimeter-4 rectangle: SLSQP + Newton Contact Polish + Greedy Expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _softplus(x, beta=100.0):
    """Numerically stable softplus approximation for max(x, 0)."""
    return np.logaddexp(0.0, beta * x) / beta - np.log(2.0) / beta


def _vectorized_objective(z, penalty_weight=1000.0, beta=100.0):
    """
    Vectorized objective with smooth penalties.
    z: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN, w]
    h = 2 - w
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    # Basic validity check for early termination
    if w < 0.1 or h < 0.1 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Only compute upper triangle
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap_raw = r_sum_pairs - d_pairs
    overlap = _softplus(overlap_raw, beta)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)
    obj += overlap_penalty

    # Boundary penalties
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )
    obj += boundary_penalty

    # Aspect ratio penalty to keep rectangle reasonable
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _get_active_constraints(x, y, r, w, h, tol=1e-4):
    """
    Identify active constraints (contacts) for Newton's method.
    Returns a list of constraint types and indices.
    Types: 'wall_left', 'wall_right', 'wall_bottom', 'wall_top', 'circle'
    """
    constraints = []
    
    # Wall contacts
    for i in range(N):
        if abs(x[i] - r[i]) < tol:
            constraints.append(('wall_left', i))
        if abs(x[i] + r[i] - w) < tol:
            constraints.append(('wall_right', i))
        if abs(y[i] - r[i]) < tol:
            constraints.append(('wall_bottom', i))
        if abs(y[i] + r[i] - h) < tol:
            constraints.append(('wall_top', i))
            
    # Circle contacts
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            r_sum = r[i] + r[j]
            if abs(d - r_sum) < tol:
                constraints.append(('circle', (i, j)))
                
    return constraints


def _newton_polish(z, max_iter=50, tol=1e-10):
    """
    Newton's method to refine the solution by solving the system of equations
    for active constraints.
    Variables: x, y, r, w (h is dependent)
    We fix w and adjust x, y, r to satisfy contacts exactly.
    """
    z = z.copy()
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w
    
    for _ in range(max_iter):
        constraints = _get_active_constraints(x, y, r, w, h, tol=1e-3)
        if not constraints:
            break
            
        # We have N*3 variables (x,y,r for each circle). w is fixed.
        # Let's pick a subset of constraints to make the system square or use least squares.
        # Actually, we can use all active constraints in a least-squares sense if they are over-determined,
        # or pick the most critical ones.
        
        # Let's construct J and g for the equations F_i(x,y,r) = 0
        # F_i = 0 for each active constraint.
        
        # We will solve for delta_x, delta_y, delta_r.
        # This is a linear system J * delta = -g.
        
        m = len(constraints)
        if m == 0:
            break
            
        # If m > 3N, we might have an over-determined system. We'll use np.linalg.lstsq.
        J = np.zeros((m, 3*N))
        g = np.zeros(m)
        
        for idx, (ctype, indices) in enumerate(constraints):
            if ctype == 'wall_left':
                i = indices
                # x_i - r_i = 0
                J[idx, 3*i] = 1.0
                J[idx, 3*i+2] = -1.0
                g[idx] = x[i] - r[i]
            elif ctype == 'wall_right':
                i = indices
                # x_i + r_i - w = 0
                J[idx, 3*i] = 1.0
                J[idx, 3*i+2] = 1.0
                g[idx] = x[i] + r[i] - w
            elif ctype == 'wall_bottom':
                i = indices
                # y_i - r_i = 0
                J[idx, 3*i+1] = 1.0
                J[idx, 3*i+2] = -1.0
                g[idx] = y[i] - r[i]
            elif ctype == 'wall_top':
                i = indices
                # y_i + r_i - h = 0
                J[idx, 3*i+1] = 1.0
                J[idx, 3*i+2] = 1.0
                g[idx] = y[i] + r[i] - h
            elif ctype == 'circle':
                i, j = indices
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 = 0
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                r_sum = r[i] + r[j]
                d2 = dx*dx + dy*dy
                g[idx] = d2 - r_sum*r_sum
                
                # Derivatives
                # d/dx_i = 2*dx, d/dy_i = 2*dy, d/dr_i = -2*r_sum
                # d/dx_j = -2*dx, d/dy_j = -2*dy, d/dr_j = -2*r_sum
                J[idx, 3*i] = 2*dx
                J[idx, 3*i+1] = 2*dy
                J[idx, 3*i+2] = -2*r_sum
                J[idx, 3*j] = -2*dx
                J[idx, 3*j+1] = -2*dy
                J[idx, 3*j+2] = -2*r_sum
                
        # Solve J * delta = -g
        try:
            delta, residuals, rank, s = np.linalg.lstsq(J, -g, rcond=None)
        except np.linalg.LinAlgError:
            break
            
        # Apply step with damping
        step_size = 1.0
        x_new = x
        y_new = y
        r_new = r
        
        for _ in range(20):
            x_trial = x + step_size * delta[0::3]
            y_trial = y + step_size * delta[1::3]
            r_trial = r + step_size * delta[2::3]
            
            # Check validity roughly
            if np.any(r_trial <= 0):
                step_size *= 0.5
                continue
                
            # Check if overlaps are reduced
            # Simple check: just ensure no massive violations
            valid = True
            for i in range(N):
                if x_trial[i] < r_trial[i] - 1e-3 or x_trial[i] > w - r_trial[i] + 1e-3:
                    valid = False
                    break
                if y_trial[i] < r_trial[i] - 1e-3 or y_trial[i] > h - r_trial[i] + 1e-3:
                    valid = False
                    break
            if not valid:
                step_size *= 0.5
                continue
                
            # If step is small enough, accept
            if step_size < 1e-6:
                break
                
            x_new = x_trial
            y_new = y_trial
            r_new = r_trial
            
            # Check convergence
            if np.max(np.abs(delta * step_size)) < tol:
                x = x_new
                y = y_new
                r = r_new
                break
                
            step_size *= 0.5
            # If we didn't break, we try smaller step next iteration? 
            # Actually, lstsq gives the exact Newton step. Damping is for numerical stability.
            # Let's just accept the damped step if it's valid.
            break
            
        if not _is_valid(x_new, y_new, r_new, w, h, tol=1e-8):
             # If newton step made it worse, revert
             continue

        x, y, r = x_new, y_new, r_new
        
        # Update z
        z[0::3][:N] = x
        z[1::3][:N] = y
        z[2::3][:N] = r
        
        if np.max(np.abs(delta)) < tol:
            break
            
    return z


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _generate_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    rng = np.random.default_rng(42)
    
    # Grid-based layouts
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.5]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.4:
            continue
            
        # Simple grid
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cw = w0 / cols
        ch = h0 / rows
        
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cw + rng.uniform(-0.01, 0.01)
            y[i] = (row + 0.5) * ch + rng.uniform(-0.01, 0.01)
            r[i] = min(cw, ch) / 2.0 * 0.95
            
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x
        z[1::3][:N] = y
        z[2::3][:N] = r
        z[-1] = w0
        guesses.append(z)
        
    # Random layouts
    for _ in range(5):
        w0 = rng.uniform(0.7, 1.3)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.05, 0.15)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)
        
    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    guesses = _generate_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # SLSQP Optimization
        try:
            # Use a smaller penalty weight initially to find a feasible region,
            # then increase it in polish steps if needed.
            res = minimize(_vectorized_objective, z0, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 400, 'ftol': 1e-12, 'maxls': 100})
            
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                # Newton Polish
                z_polished = _newton_polish(z_opt, max_iter=20)
                valid_p, val_p = check_validity(z_polished)
                if valid_p and val_p > best_value:
                    best_value = val_p
                    best_result = z_polished.copy()
                elif valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            elif not valid:
                # Try shrinking radii slightly to make valid
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.9
                valid_r, val_r = check_validity(z_repair)
                if valid_r and val_r > best_value:
                    best_value = val_r
                    best_result = z_repair.copy()

        except Exception as e:
            continue

    # Polish loop on best result
    if best_result is not None:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Newton Polish
            z_polished = _newton_polish(best_result, max_iter=50)
            valid, val = check_validity(z_polished)
            if valid and val > best_value:
                best_value = val
                best_result = z_polished.copy()
            else:
                break
                
            # Greedy Expansion
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            
            x_new, y_new, r_new = _greedy_expand(x, y, r, w, h, deadline - 5)
            
            if _is_valid(x_new, y_new, r_new, w, h):
                val_new = float(np.sum(r_new))
                if val_new > best_value:
                    z_new = np.empty(3 * N + 1)
                    z_new[0::3][:N] = x_new
                    z_new[1::3][:N] = y_new
                    z_new[2::3][:N] = r_new
                    z_new[-1] = w
                    best_value = val_new
                    best_result = z_new
                else:
                    break
            else:
                break

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h, tol=1e-12):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
