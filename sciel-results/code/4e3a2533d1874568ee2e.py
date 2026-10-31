"""Circle packing in perimeter-4 rectangle: Constrained SLSQP + Simulated Annealing Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _slsqp_objective(z):
    """Objective: maximize sum of radii (negative for minimization)."""
    r = z[2::3][:N]
    return -np.sum(r)


def _make_overlap_constraint(i, j):
    """Create a constraint function for non-overlap of circles i and j."""
    def constraint(p):
        dx = p[3*i] - p[3*j]
        dy = p[3*i+1] - p[3*j+1]
        r_sum = p[3*i+2] + p[3*j+2]
        return dx*dx + dy*dy - r_sum*r_sum
    return constraint


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if circles are valid (non-overlapping and within bounds)."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to (new_x, new_y) is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _feasibility_check(x, y, r, w, h, tol=1e-9):
    """Check if configuration is feasible (no overlaps, within bounds)."""
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx*dx + dy*dy < (r[i] + r[j])**2 - tol:
                return False
    return True


def _simulated_annealing_polish(x, y, r, w, h, time_budget):
    """Simulated annealing local search to improve packing."""
    n = N
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = float(np.sum(r))
    
    T = 0.05
    T_min = 1e-6
    cooling = 0.98
    rng = np.random.default_rng(123)
    
    while time.time() < time_budget and T > T_min:
        # Pick a random circle to move
        i = rng.integers(0, n)
        
        # Try random small perturbation
        dx = rng.normal(0, 0.02)
        dy = rng.normal(0, 0.02)
        
        new_x = x[i] + dx
        new_y = y[i] + dy
        
        # Clamp to bounds
        new_x = np.clip(new_x, r[i], w - r[i])
        new_y = np.clip(new_y, r[i], h - r[i])
        
        # Check feasibility
        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
            x[i] = new_x
            y[i] = new_y
            
            # Try to grow this circle
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0, r_max)
            
            if r_max > r[i] + 1e-8:
                r[i] = r_max
            
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        else:
            # Accept with probability exp(-delta/T) if it's worse
            # For simplicity, reject invalid moves
            pass
        
        T *= cooling
    
    return best_x, best_y, best_r


def _expand_and_optimize(z, time_budget):
    """Expand circles greedily, then use SLSQP to re-optimize positions."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Greedy expansion pass
    for rnd in range(10):
        if time.time() > time_budget - 5:
            break
        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0, r_max)
            if r_max > r[i] + 1e-6:
                r[i] = r_max
                improved = True
        if not improved:
            break

    # SLSQP re-optimization with only overlap constraints
    z_opt = np.empty(3*N + 1)
    z_opt[0::3][:n] = x
    z_opt[1::3][:n] = y
    z_opt[2::3][:n] = r
    z_opt[-1] = w

    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': _make_overlap_constraint(i, j)})
    
    bounds = [(0, 2)] * (3*N) + [(0.3, 1.7)]
    
    try:
        res = minimize(_slsqp_objective, z_opt, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50})
        if _feasibility_check(res.x[0::3][:n], res.x[1::3][:n], res.x[2::3][:n], 
                              res.x[-1], 2.0 - res.x[-1]):
            return res.x
    except Exception:
        pass
    
    return None


def _aspect_search(z, time_budget):
    """Search over aspect ratios to find better packing."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_val = float(np.sum(r))
    best_result = z.copy()

    for w_try in np.linspace(0.7, 1.3, 7):
        if time.time() > time_budget - 10:
            break
        h_try = 2.0 - w_try
        if h_try < 0.4 or w_try < 0.4:
            continue

        # Scale to new aspect
        x_try = x * (w_try / w)
        y_try = y * (h_try / h)
        scale = min(w_try / w, h_try / h)
        r_try = r * scale

        if not _feasibility_check(x_try, y_try, r_try, w_try, h_try):
            continue

        z_try = np.empty(3*N + 1)
        z_try[0::3][:n] = x_try
        z_try[1::3][:n] = y_try
        z_try[2::3][:n] = r_try
        z_try[-1] = w_try

        # Expand and optimize
        result = _expand_and_optimize(z_try, time_budget - 15)
        if result is not None:
            val = float(np.sum(result[2::3][:n]))
            if val > best_val:
                best_val = val
                best_result = result

    if best_val > float(np.sum(r)) + 1e-6:
        return best_result
    return None


def _final_refine(z, time_budget):
    """Final SLSQP refinement with overlap constraints only."""
    n = N
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': _make_overlap_constraint(i, j)})
    
    bounds = [(0, 2)] * (3*N) + [(0.3, 1.7)]
    
    try:
        res = minimize(_slsqp_objective, z, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 100})
        if _feasibility_check(res.x[0::3][:n], res.x[1::3][:n], res.x[2::3][:n], 
                              res.x[-1], 2.0 - res.x[-1]):
            return res.x
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Constrained SLSQP from multiple starts
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': _make_overlap_constraint(i, j)})
    
    bounds = [(0, 2)] * (3*N) + [(0.3, 1.7)]

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(_slsqp_objective, z0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()

        except Exception:
            continue

    # Phase 2: Polish on best result
    if best_result is not None:
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
        w = best_result[-1]
        h = 2.0 - w

        # Simulated annealing polish
        if time.time() < deadline - 20:
            bx, by, br = _simulated_annealing_polish(x, y, r, w, h, deadline - 15)
            z_sa = np.empty(3*N + 1)
            z_sa[0::3][:N] = bx
            z_sa[1::3][:N] = by
            z_sa[2::3][:N] = br
            z_sa[-1] = w
            valid, val = check_validity(z_sa)
            if valid and val > best_value:
                best_value = val
                best_result = z_sa

        # Expand and optimize
        if time.time() < deadline - 15:
            expanded = _expand_and_optimize(best_result, deadline - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # Aspect search
        if time.time() < deadline - 12:
            aspect_res = _aspect_search(best_result, deadline - 8)
            if aspect_res is not None:
                valid, val = check_validity(aspect_res)
                if valid and val > best_value:
                    best_value = val
                    best_result = aspect_res

        # Final refine
        if time.time() < deadline - 6:
            refined = _final_refine(best_result, deadline - 2)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final micro-growth
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
