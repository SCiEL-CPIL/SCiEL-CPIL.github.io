"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _surrogate_energy(z):
    """
    Surrogate energy: penalizes overlaps, wall violations, and encourages
    circles to be well-separated (gap penalty) and centered.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0.01):
        return 1e6

    energy = 0.0

    # Overlap penalty (quadratic)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    energy += 1000.0 * np.sum(overlap * overlap)

    # Wall violation penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    energy += 1000.0 * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Gap penalty: encourage circles to be close to each other (reduce wasted space)
    # This acts as a "pressure" to pack tightly
    gap = np.maximum(d_pairs - r_sum_pairs, 0.0)
    energy += 0.5 * np.sum(gap * gap)

    # Aspect ratio penalty to keep rectangle reasonable
    if w < 0.5:
        energy += 100.0 * (0.5 - w) ** 2
    elif w > 1.5:
        energy += 100.0 * (w - 1.5) ** 2

    return energy


def _dynamics_simulate(z, time_budget, dt=0.002, max_iter=5000):
    """
    Rigid-body dynamics simulation with impulse-based collision resolution.
    Mass proportional to radius squared.
    """
    x = z[0::3][:N].copy().astype(float)
    y = z[1::3][:N].copy().astype(float)
    r = z[2::3][:N].copy().astype(float)
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = r * r

    # Ensure valid start
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    damping = 0.995
    start_time = time.time()

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 500.0
        fx += right_viol * 500.0
        fy -= bottom_viol * 500.0
        fy += top_viol * 500.0

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 200.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Update velocities
        vx += dt * fx / mass
        vy += dt * fy / mass

        # Damping
        vx *= damping
        vy *= damping

        # Update positions
        x += dt * vx
        y += dt * vy

        # Clamp to walls
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Impulse-based collision resolution (post-step correction)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-9:
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                m_i = mass[i_idx]
                m_j = mass[j_idx]
                total_m = m_i + m_j
                correction = overlap[idx] * 0.5
                x[i_idx] += ux[idx] * correction * (m_j / total_m)
                y[i_idx] += uy[idx] * correction * (m_j / total_m)
                x[j_idx] -= ux[idx] * correction * (m_i / total_m)
                y[j_idx] -= uy[idx] * correction * (m_i / total_m)

            # Re-clamp after correction
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Final validity check
    if not _is_valid(x, y, r, w, h):
        return None

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:N] = x
    z_new[1::3][:N] = y
    z_new[2::3][:N] = r
    z_new[-1] = w
    return z_new


def _expand_radii(z, time_budget):
    """
    Iteratively expand radii using binary search, moving circles to make space.
    """
    x = z[0::3][:N].copy().astype(float)
    y = z[1::3][:N].copy().astype(float)
    r = z[2::3][:N].copy().astype(float)
    w = z[-1]
    h = 2.0 - w

    start_time = time.time()
    improved = True
    rounds = 0

    while improved and time.time() - start_time < time_budget and rounds < 50:
        improved = False
        rounds += 1

        for i in range(N):
            # Max radius limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall

            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed

            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid

                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        # If no improvement, try to move circles to create space
        if not improved:
            for i in range(N):
                # Find direction of least resistance
                best_move = None
                best_gain = 0.0

                # Try 8 directions
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    for step in [0.02, 0.01, 0.005]:
                        new_x = x[i] + np.cos(angle) * step
                        new_y = y[i] + np.sin(angle) * step

                        if new_x < r[i] or new_x > w - r[i]:
                            continue
                        if new_y < r[i] or new_y > h - r[i]:
                            continue

                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = new_x - x[j]
                            dy = new_y - y[j]
                            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                                valid = False
                                break

                        if valid:
                            # Estimate gain: how much can we expand after moving?
                            r_wall = min(new_x, w - new_x, new_y, h - new_y)
                            r_max = r_wall
                            for j in range(N):
                                if i == j:
                                    continue
                                dx = new_x - x[j]
                                dy = new_y - y[j]
                                d = np.sqrt(dx * dx + dy * dy)
                                r_allowed = d - r[j]
                                if r_allowed < r_max:
                                    r_max = r_allowed
                            gain = max(0.0, r_max - r[i])
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (new_x, new_y)

                if best_move is not None and best_gain > 1e-6:
                    x[i] = best_move[0]
                    y[i] = best_move[1]
                    improved = True

    # Final uniform expansion
    for _ in range(200):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w
        return z_new
    return None


def _optimize_aspect(z0, time_budget):
    """
    Optimize aspect ratio (width) using SLSQP on surrogate energy.
    """
    def obj(w):
        w = np.clip(w, 0.3, 1.7)
        h = 2.0 - w
        z = z0.copy()
        # Scale positions to fit new rectangle
        scale_x = w / z0[-1]
        scale_y = h / (2.0 - z0[-1])
        z[0::3][:N] *= scale_x
        z[1::3][:N] *= scale_y
        z[2::3][:N] *= min(scale_x, scale_y)
        z[-1] = w
        return _surrogate_energy(z)

    try:
        res = minimize(obj, z0[-1], method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        w_opt = np.clip(res.x, 0.3, 1.7)
        return w_opt
    except Exception:
        return z0[-1]


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

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

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]
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
        for _ in range(10):
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

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Step 1: Dynamics simulation to resolve overlaps
        sim_time = min(3.0, (deadline - (time.time() - start_time)) * 0.3)
        z_sim = _dynamics_simulate(z0, sim_time)
        if z_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        w_opt = _optimize_aspect(z_sim, 1.0)
        if w_opt != z_sim[-1]:
            h_old = 2.0 - z_sim[-1]
            h_new = 2.0 - w_opt
            scale_x = w_opt / z_sim[-1]
            scale_y = h_new / h_old
            z_sim[0::3][:N] *= scale_x
            z_sim[1::3][:N] *= scale_y
            z_sim[2::3][:N] *= min(scale_x, scale_y)
            z_sim[-1] = w_opt

        # Step 3: Expand radii
        expand_time = min(5.0, (deadline - (time.time() - start_time)) * 0.4)
        z_expanded = _expand_radii(z_sim, expand_time)
        if z_expanded is not None:
            valid, val = check_validity(z_expanded)
            if valid and val > best_value:
                best_value = val
                best_result = z_expanded.copy()
        else:
            valid, val = check_validity(z_sim)
            if valid and val > best_value:
                best_value = val
                best_result = z_sim.copy()

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 5:
        final_time = deadline - (time.time() - start_time) - 2
        z_final = _expand_radii(best_result, final_time)
        if z_final is not None:
            valid, val = check_validity(z_final)
            if valid and val > best_value:
                best_value = val
                best_result = z_final

    if best_result is None:
        # Fallback
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

    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


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
