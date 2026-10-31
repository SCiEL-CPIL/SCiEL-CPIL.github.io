"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

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
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax_unified(z, max_iter=500, tol=1e-6):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    This helps SLSQP start from a better initial point.
    """
    z = z.copy()
    for _ in range(max_iter):
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)

        # Boundary forces
        for i in range(N):
            # Left
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i])
            # Right
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w)
            # Bottom
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i])
            # Top
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h)

        # Pairwise forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d:
                    d = np.sqrt(d_sq)
                    if d < 1e-9:
                        # Random nudge
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        d = 1e-6
                    f = (min_d - d) / d
                    fx[i] += f * dx
                    fy[i] += f * dy
                    fx[j] -= f * dx
                    fy[j] -= f * dy
                    # Slightly shrink radii if overlapping significantly
                    shrink = 0.01 * (min_d - d)
                    fr[i] -= shrink
                    fr[j] -= shrink

        # Update
        step = 0.1
        x += step * fx
        y += step * fy
        r += step * fr

        # Clamp
        r = np.maximum(r, 0.01)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        z[0::3][:N] = x
        z[1::3][:N] = y
        z[2::3][:N] = r

        # Check convergence
        if np.max(np.abs(fx)) < tol and np.max(np.abs(fy)) < tol:
            break

    return z


def _polish_result(z, time_budget=10.0):
    """
    Local polish using SLSQP with tight tolerance.
    """
    start = time.time()
    try:
        res = minimize(
            _vectorized_objective,
            z,
            method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 50}
        )
        return res.x
    except Exception:
        return None


def _expansion_relaxation_polish(z, time_budget=10.0):
    """
    Try to expand radii slightly while maintaining validity.
    """
    z = z.copy()
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w

    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for i in range(N):
            # Try expanding radius i by small amount
            delta = 0.001
            r_new = r[i] + delta
            # Check boundaries
            if r_new > x[i] or r_new > w - x[i] or r_new > y[i] or r_new > h - y[i]:
                continue
            # Check overlaps
            valid = True
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                min_d = r_new + r[j]
                if d_sq < min_d * min_d - 1e-9:
                    valid = False
                    break
            if valid:
                r[i] = r_new
                improved = True

    z[2::3][:N] = r
    return z


def _constraint_refine(z, time_budget=10.0):
    """
    Refine using SLSQP with explicit constraints if possible,
    or just another SLSQP pass with different settings.
    """
    start = time.time()
    try:
        # Increase penalty weight for stricter constraints
        def strict_obj(z_):
            return _vectorized_objective(z_, penalty_weight=10000.0)

        res = minimize(
            strict_obj,
            z,
            method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 50}
        )
        return res.x
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def objective(z):
        return _vectorized_objective(z, penalty_weight=1000.0)

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

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
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

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

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

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 18)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 18:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 14:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 8)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Second expansion pass after SLSQP
        if best_result is not None and time.time() < deadline - 10:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 5. Final quick polish
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Convert to (cx, cy, r) format
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    return np.column_stack((x, y, r))

# EVOLVE-BLOCK-END
