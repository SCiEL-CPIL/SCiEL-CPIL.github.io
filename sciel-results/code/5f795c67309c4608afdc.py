"""Circle packing in perimeter-4 rectangle: unified penalty SLSQP + hard-constraint refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 132.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Objective for penalty-based SLSQP (Phase 1) ---
    penalty_weight = 800.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty (vectorized pairwise)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                dist_req = r[i] + r[j]
                if d < dist_req:
                    obj += penalty_weight * (dist_req - d) ** 2

        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def check_validity(z, tol=1e-8):
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

    # --- Generate diverse initial guesses ---
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9]
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

    # --- Phase 1: Penalty-based SLSQP from multiple starts ---
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 8:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=150)

        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 8
        maxiter = int(min(250, max(50, remaining * 25)))

        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Hard-constraint SLSQP refinement (structural improvement) ---
    if best_result is not None and time.time() < deadline - 15:
        refined = _hard_constraint_refine(best_result, deadline - (time.time() - start_time))
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # --- Phase 3: COBYLA refinement from best (different solver family) ---
    if best_result is not None and time.time() < deadline - 10:
        cobyla_refined = _cobyla_refine(best_result, deadline - (time.time() - start_time))
        if cobyla_refined is not None:
            valid, val = check_validity(cobyla_refined)
            if valid and val > best_value:
                best_value = val
                best_result = cobyla_refined

    # --- Phase 4: Force-directed polish with radius growth ---
    if best_result is not None and time.time() < deadline - 5:
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

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

    # Safety shrink
    r = r * (1 - 1e-6)

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=150):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005

    for _ in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 50
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 50
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 50
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 50

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _hard_constraint_refine(z_init, time_budget):
    """Refine using SLSQP with explicit hard inequality constraints."""
    if time_budget < 3:
        return None

    z = z_init.copy().astype(float)
    n = N

    # Build constraint list
    constraints = []

    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3 * idx] - p[3 * idx + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3 * idx] - p[3 * idx + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3 * idx + 1] - p[3 * idx + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3 * i] - p[3 * j]) ** 2 +
                    (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                    (p[3 * i + 2] + p[3 * j + 2]) ** 2
                )
            })

    # Objective: maximize sum of radii
    def obj(p):
        r = p[2::3][:n]
        return -np.sum(r)

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

    elapsed = time.time()
    maxiter = int(min(300, max(50, (time_budget - 3) * 30)))

    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

        if res.success or res.fun < obj(z):
            z_new = res.x
            # Verify validity
            x = z_new[0::3][:n]
            y = z_new[1::3][:n]
            r = z_new[2::3][:n]
            w = z_new[-1]
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                return z_new
            # If slightly invalid, try small shrink
            r_shrunk = r * 0.99
            if _is_valid(x, y, r_shrunk, w, h):
                z_new[2::3][:n] = r_shrunk
                return z_new
    except Exception:
        pass

    return None


def _cobyla_refine(z_init, time_budget):
    """Refine using COBYLA (derivative-free) to escape SLSQP local minima."""
    if time_budget < 3:
        return None

    z = z_init.copy().astype(float)
    n = N

    # COBYLA uses a penalty-based objective (it doesn't handle constraints natively well)
    # but with a high penalty it can still find valid solutions
    penalty = 500.0

    def obj_cobyla(p):
        x = p[0::3][:n]
        y = p[1::3][:n]
        r = p[2::3][:n]
        w = p[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        val = -np.sum(r)

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                req = r[i] + r[j]
                if d < req:
                    val += penalty * (req - d) ** 2

        for i in range(n):
            if x[i] < r[i]:
                val += penalty * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                val += penalty * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                val += penalty * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                val += penalty * (y[i] - (h - r[i])) ** 2

        return val

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

    # COBYLA doesn't accept bounds directly; use a small perturbation start
    # and rely on penalty to keep in bounds
    maxiter = int(min(500, max(100, (time_budget - 3) * 40)))

    try:
        res = minimize(obj_cobyla, z, method='COBYLA',
                       options={'maxiter': maxiter, 'rhobeg': 0.01, 'catol': 1e-8})

        z_new = res.x
        x = z_new[0::3][:n]
        y = z_new[1::3][:n]
        r = z_new[2::3][:n]
        w = z_new[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_new
        r_shrunk = r * 0.98
        if _is_valid(x, y, r_shrunk, w, h):
            z_new[2::3][:n] = r_shrunk
            return z_new
    except Exception:
        pass

    return None


def _polish_result(z, time_budget):
    """Polish with force-directed relaxation + iterative radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Force-directed relaxation to settle positions
    dt = 0.003
    max_iter = int(min(400, time_budget * 40))

    for it in range(max_iter):
        if time.time() > time_budget + 0:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 40
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 40
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 40
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 40

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Iterative radius growth: try to grow all radii by a small factor
    # If it fails, try growing subsets (circles with most slack)
    grow_factor = 1.001
    for _ in range(200):
        if time.time() > time_budget + 0:
            break

        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            # Re-relax positions briefly after growth
            for _it in range(20):
                fx = np.zeros(n)
                fy = np.zeros(n)
                for i in range(n):
                    if x[i] - r[i] < 0:
                        fx[i] += (x[i] - r[i]) * 40
                    if x[i] + r[i] > w:
                        fx[i] -= (x[i] + r[i] - w) * 40
                    if y[i] - r[i] < 0:
                        fy[i] += (y[i] - r[i]) * 40
                    if y[i] + r[i] > h:
                        fy[i] -= (y[i] + r[i] - h) * 40
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        min_d = r[i] + r[j]
                        if d < min_d and d > 1e-12:
                            overlap = min_d - d
                            fx[i] += (dx / d) * overlap * 20
                            fy[i] += (dy / d) * overlap * 20
                            fx[j] -= (dx / d) * overlap * 20
                            fy[j] -= (dy / d) * overlap * 20
                x += 0.002 * fx
                y += 0.002 * fy
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
        else:
            # Try growing only the smallest radii (most slack)
            sorted_idx = np.argsort(r)
            for k in range(3, 0, -1):
                r_trial2 = r.copy()
                r_trial2[sorted_idx[:k]] *= grow_factor
                if _is_valid(x, y, r_trial2, w, h):
                    r = r_trial2
                    break
            else:
                break

    # Final validity check
    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
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
