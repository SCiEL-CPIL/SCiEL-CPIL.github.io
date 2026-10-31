"""Circle packing in perimeter-4 rectangle: unified SLSQP with free aspect ratio."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Phase 1: Unified SLSQP optimization with free aspect ratio ---
    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w

    n_starts = 12
    penalty_weight = 500.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Smooth overlap penalty using log-sum-exp for numerical stability
        # For each pair, penalty = softplus(overlap) where overlap = r_i + r_j - d
        # Use a smooth approximation: penalty = -log(1 + exp(-beta * d + beta * (r_i+r_j))) / beta
        # This is smooth everywhere and approximates max(0, r_i+r_j-d) for large beta
        beta = 50.0  # sharpness of the smooth max
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - d
                # Smooth penalty: use log-sum-exp form
                arg = beta * overlap
                # softplus(arg) = log(1 + exp(arg)), but for large arg use arg + log(1+exp(-arg))
                if arg > 30:
                    soft = arg
                elif arg < -30:
                    soft = 0.0
                else:
                    soft = np.log1p(np.exp(arg))
                obj += penalty_weight * soft / beta

        # Smooth boundary penalty
        for i in range(N):
            for margin in [x[i] - r[i], (w - r[i]) - x[i], y[i] - r[i], (h - r[i]) - y[i]]:
                arg = beta * (-margin)  # penalty when margin < 0
                if arg > 30:
                    soft = arg
                elif arg < -30:
                    soft = 0.0
                else:
                    soft = np.log1p(np.exp(arg))
                obj += penalty_weight * soft / beta

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []

        # Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
        side = int(np.ceil(np.sqrt(N)))

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(n_starts - len(guesses)):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 5:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=200)

        # SLSQP optimization
        maxiter = int(min(300, (deadline - (time.time() - start_time)) * 30))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair

        except Exception:
            continue

    # --- Phase 2: Local polish on best result ---
    if best_result is not None and time.time() < deadline - 10:
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # --- Phase 3: High-precision SLSQP polish with tighter tolerances ---
    if best_result is not None and time.time() < deadline - 5:
        try:
            res_hp = minimize(objective, best_result, method='SLSQP',
                              bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                              options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 100})
            z_hp = res_hp.x
            valid_hp, val_hp = check_validity(z_hp)
            if valid_hp and val_hp > best_value:
                best_value = val_hp
                best_result = z_hp
        except Exception:
            pass

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


def _force_relax_unified(z, max_iter=200):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003

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
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed relaxation and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Adaptive step size: start larger, decay over iterations
    dt_start = 0.003
    dt_end = 0.0005
    max_iter = int(min(1200, time_budget * 80))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # Decay step size
        frac = it / max(1, max_iter)
        dt = dt_start * (dt_end / dt_start) ** frac

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 30
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 30
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 30
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 30

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 15
                    fy[i] += (dy / d) * overlap * 15
                    fx[j] -= (dx / d) * overlap * 15
                    fy[j] -= (dy / d) * overlap * 15

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly with smaller steps for precision
    grow_factor = 1.0002
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Final micro-adjustment: try tiny per-circle radius increases
    for i in range(n):
        if time.time() > time_budget:
            break
        for _ in range(50):
            r_trial = r.copy()
            r_trial[i] *= 1.0001
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
            else:
                break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
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
