import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
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

    # Keep aspect ratio in reasonable range
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _smooth_objective(z, beta=50.0):
    """
    Smooth log-sum-exp based objective for better gradient flow.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Smooth pairwise overlap via log-sum-exp
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    overlap_vals = r_sum_pairs - d_pairs

    shifted = beta * overlap_vals
    shifted = np.clip(shifted, -500, 500)
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_max_overlap = lse / beta

    hard_overlap = np.maximum(overlap_vals, 0.0)
    quad_penalty = 100.0 * np.sum(hard_overlap * hard_overlap)

    obj += 500.0 * smooth_max_overlap + quad_penalty

    # Smooth boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    shifted_b = beta * all_viol
    shifted_b = np.clip(shifted_b, -500, 500)
    max_b = np.max(shifted_b)
    lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
    smooth_max_viol = lse_b / beta

    hard_viol = np.maximum(all_viol, 0.0)
    obj += 500.0 * smooth_max_viol + 100.0 * np.sum(hard_viol * hard_viol)

    if w < 0.4:
        obj += 200.0 * (0.4 - w) ** 2
    elif w > 1.6:
        obj += 200.0 * (w - 1.6) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

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

    def get_initial_guesses():
        """Generate diverse initial guesses using grid-based decomposition."""
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # Grid-based initializations with various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8, 1.1, 0.9, 1.4]
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

        # Staggered/hexagonal grid
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
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

    # Run optimization from multiple initial guesses
    for guess in get_initial_guesses():
        if time.time() > deadline - 5.0:
            break

        try:
            res = minimize(
                objective,
                guess,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )

            if res.success or res.fun < 1e5:
                valid, val = check_validity(res.x)
                if valid and val > best_value:
                    best_value = val
                    best_result = res.x.copy()
        except Exception:
            continue

    # If no valid result found, try smooth objective
    if best_result is None:
        for guess in get_initial_guesses()[:5]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    lambda z: _smooth_objective(z, beta=30.0),
                    guess,
                    method='SLSQP',
                    options={'maxiter': 150, 'ftol': 1e-10, 'disp': False}
                )
                valid, val = check_validity(res.x)
                if valid and val > best_value:
                    best_value = val
                    best_result = res.x.copy()
            except Exception:
                continue

    # Fallback: return a valid grid-based solution
    if best_result is None:
        # Simple uniform grid
        C, R = 5, 5
        k = 2.0 / (C + R)
        W = C * k
        H = R * k
        r = (k / 2.0) * 0.999
        circles = []
        count = 0
        for j in range(R):
            for i in range(C):
                if count >= N:
                    break
                x = (i + 0.5) * (W / C)
                y = (j + 0.5) * (H / R)
                circles.append([x, y, r])
                count += 1
            if count >= N:
                break
        return np.array(circles)

    # Extract circles from best result
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Slight shrinkage for numerical safety
    r = r * 0.9995

    circles = np.column_stack([x, y, r])
    return circles
