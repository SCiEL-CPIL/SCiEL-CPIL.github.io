"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish + LSE Refine."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, fsolve

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
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def construct():
    global start_time
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 80:
        de_bounds = [(0, 2)] * (3 * N) + [(0.5, 1.5)]
        de_obj = lambda zv: _vectorized_objective(zv, 1000.0)
        
        try:
            de_res = differential_evolution(
                de_obj, de_bounds, seed=42, maxiter=50, 
                tol=1e-8, popsize=15, mutation=(0.3, 0.8), 
                recombination=0.7, init='latinhypercube'
            )
            if de_res.success or de_res.fun < 0:
                z_de = de_res.x
                valid, val = check_validity(z_de)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_de.copy()
        except Exception:
            pass

    # Phase 2: Local optimization from multiple starts
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Soft-repulsion relaxation to resolve initial overlaps and create space
        z0_relaxed = _soft_repulsion_relax(z0, max_iter=500, time_budget=deadline - 30)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 40)))
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

    # If DE found something better, use it as the base
    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # --- Phase 3: Advanced Polish on best result ---
    # Additional polish: try a few more SLSQP restarts from perturbed best
    if best_result is not None and time.time() < deadline - 40:
        rng_perturb = np.random.default_rng(123)
        for perturb_trial in range(3):
            if time.time() > deadline - 40:
                break
            z_perturb = best_result.copy()
            # Small random perturbation
            noise = rng_perturb.normal(0, 0.005, size=3 * N)
            z_perturb[:3*N] += noise
            z_perturb[3*N] = best_result[-1]  # keep w

            try:
                res_p = minimize(objective, z_perturb, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 60})
                z_opt_p = res_p.x
                valid_p, val_p = check_validity(z_opt_p)
                if valid_p and val_p > best_value:
                    best_value = val_p
                    best_result = z_opt_p.copy()
            except Exception:
                continue
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
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
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Contact-graph exact tangency polish
        if best_result is not None and time.time() < deadline - 5:
            contact_polished = _contact_graph_polish(best_result, deadline - (time.time() - start_time) - 2)
            if contact_polished is not None:
                valid, val = check_validity(contact_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = contact_polished

        # 5. LSE soft-expansion refinement
        if best_result is not None and time.time() < deadline - 15:
            lse_result = _lse_soft_expansion(best_result, deadline - (time.time() - start_time) - 10)
            if lse_result is not None:
                valid, val = check_validity(lse_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse_result

        # 6. Coordinate descent polish
        if best_result is not None and time.time() < deadline - 8:
            cd_result = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 5)
            if cd_result is not None:
                valid, val = check_validity(cd_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = cd_result

        # 7. Final quick polish
        if best_result is not None and time.time() < deadline - 3:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time) - 1)
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

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _soft_repulsion_relax(z, max_iter=1500, time_budget=None):
    """Fully vectorized soft-repulsion gradient descent to push circles apart."""
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])

    softness = 0.05
    base_lr = 0.003
    min_lr = 0.0001
    max_lr = 0.008
    lr = base_lr

    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break

        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces (vectorized)
        left_gap = x - r
        right_gap = w - x - r
        bottom_gap = y - r
        top_gap = h - y - r

        lf = np.where(left_gap < softness, (softness - left_gap) / softness, 0.0)
        rf = np.where(right_gap < softness, (softness - right_gap) / softness, 0.0)
        bf = np.where(bottom_gap < softness, (softness - bottom_gap) / softness, 0.0)
        tf = np.where(top_gap < softness, (softness - top_gap) / softness, 0.0)

        left_pen = np.maximum(-left_gap, 0.0)
        right_pen = np.maximum(-right_gap, 0.0)
        bottom_pen = np.maximum(-bottom_gap, 0.0)
        top_pen = np.maximum(-top_gap, 0.0)

        fx -= lf * 10.0 - left_pen * 200.0 + rf * 10.0 + right_pen * 200.0
        fy -= bf * 10.0 - bottom_pen * 200.0 + tf * 10.0 + top_pen * 200.0

        # Pairwise forces (fully vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        gap = d_pairs - r_sum_pairs
        soft_gap = np.clip(softness - gap, 0.0, 2 * softness)
        f_mag = (soft_gap / (2 * softness)) ** 2
        overlap_boost = np.where(gap < 0, 5.0, 1.0)
        f_pairs = f_mag * overlap_boost * 15.0

        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        # Vectorized force accumulation using scatter-add
        for idx in range(n_pairs):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]

        force_mag = float(np.sqrt(np.sum(fx * fx) + np.sum(fy * fy))) / n
        if force_mag > 1.0:
            lr = max(min_lr, lr * 0.7)
        elif force_mag < 0.05:
            lr = min(max_lr, lr * 1.1)

        x = x + lr * fx
        y = y + lr * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        z[0::3][:n] = x
        z[1::3][:n] = y

        # Convergence check
        d_sq_final = dx * dx + dy * dy
        d_final = np.sqrt(np.maximum(d_sq_final[iu], 1e-12))
        max_overlap = float(np.max(r_sum_pairs - d_final)) if n_pairs > 0 else 0.0
        if max_overlap < 1e-8:
            break

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing: vectorized soft-repulsion + per-circle radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    softness = 0.03
    lr = 0.001
    min_lr = 0.00005
    max_lr = 0.004

    max_iter = int(min(6000, time_budget * 500))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_gap = x - r
        right_gap = w - x - r
        bottom_gap = y - r
        top_gap = h - y - r

        left_pen = np.maximum(-left_gap, 0.0)
        right_pen = np.maximum(-right_gap, 0.0)
        bottom_pen = np.maximum(-bottom_gap, 0.0)
        top_pen = np.maximum(-top_gap, 0.0)

        lf = np.where(left_gap < softness, (softness - left_gap) / softness, 0.0)
        rf = np.where(right_gap < softness, (softness - right_gap) / softness, 0.0)
        bf = np.where(bottom_gap < softness, (softness - bottom_gap) / softness, 0.0)
        tf = np.where(top_gap < softness, (softness - top_gap) / softness, 0.0)

        fx -= lf * 8.0 - left_pen * 300.0 + rf * 8.0 + right_pen * 300.0
        fy -= bf * 8.0 - bottom_pen * 300.0 + tf * 8.0 + top_pen * 300.0

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        gap = d_pairs - r_sum_pairs
        soft_gap = np.clip(softness - gap, 0.0, 2 * softness)
        f_mag = (soft_gap / (2 * softness)) ** 2
        overlap_boost = np.where(gap < 0, 8.0, 1.0)
        f_pairs = f_mag * overlap_boost * 12.0

        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        for idx in range(n_pairs):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]

        force_mag = float(np.sqrt(np.sum(fx * fx) + np.sum(fy * fy))) / n
        if force_mag > 0.5:
            lr = max(min_lr, lr * 0.7)
        elif force_mag < 0.02:
            lr = min(max_lr, lr * 1.1)

        x += lr * fx
        y += lr * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Vectorized per-circle radius expansion
    for _ in range(30):
        if time.time() > time_budget:
            break
        improved = False
        # Compute max radius for all circles at once
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        # For each circle, compute distance to all others
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        # r_max[i] = min over j != i of (d[i,j] - r[j])
        r_from_others = d_all - r[None, :]
        np.fill_diagonal(r_from_others, np.inf)
        r_max_from_others = np.min(r_from_others, axis=1)
        r_max = np.minimum(r_wall, r_max_from_others)
        r_max = np.maximum(r_max, 0.0)

        for i in range(n):
            if r_max[i] > r[i] + 1e-7:
                lo = r[i]
                hi = r_max[i]
                for _ in range(18):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Final uniform expansion
    for _ in range(20):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.005
        for _ in range(12):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: vectorized soft-repulsion + iterative per-circle radius expansion.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    softness = 0.04
    lr = 0.002
    min_lr = 0.00005
    max_lr = 0.005

    # Phase 1: Soft-repulsion position optimization
    max_pos_rounds = 3
    for pos_rnd in range(max_pos_rounds):
        if time.time() > time_budget:
            break

        remaining = time_budget - time.time()
        if remaining < 1:
            break
        max_iter = int(min(2000, remaining * 400))
        for it in range(max_iter):
            if time.time() > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            left_gap = x - r
            right_gap = w - x - r
            bottom_gap = y - r
            top_gap = h - y - r

            left_pen = np.maximum(-left_gap, 0.0)
            right_pen = np.maximum(-right_gap, 0.0)
            bottom_pen = np.maximum(-bottom_gap, 0.0)
            top_pen = np.maximum(-top_gap, 0.0)

            lf = np.where(left_gap < softness, (softness - left_gap) / softness, 0.0)
            rf = np.where(right_gap < softness, (softness - right_gap) / softness, 0.0)
            bf = np.where(bottom_gap < softness, (softness - bottom_gap) / softness, 0.0)
            tf = np.where(top_gap < softness, (softness - top_gap) / softness, 0.0)

            fx -= lf * 10.0 - left_pen * 300.0 + rf * 10.0 + right_pen * 300.0
            fy -= bf * 10.0 - bottom_pen * 300.0 + tf * 10.0 + top_pen * 300.0

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

            gap = d_pairs - r_sum_pairs
            soft_gap = np.clip(softness - gap, 0.0, 2 * softness)
            f_mag = (soft_gap / (2 * softness)) ** 2
            overlap_boost = np.where(gap < 0, 10.0, 1.0)
            f_pairs = f_mag * overlap_boost * 15.0

            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(n_pairs):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

            force_mag = float(np.sqrt(np.sum(fx * fx) + np.sum(fy * fy))) / n
            if force_mag > 0.5:
                lr = max(min_lr, lr * 0.7)
            elif force_mag < 0.02:
                lr = min(max_lr, lr * 1.1)

            x += lr * fx
            y += lr * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Phase 2: Vectorized per-circle radius expansion
    max_expand_rounds = 35
    for exp_rnd in range(max_expand_rounds):
        if time.time() > time_budget:
            break

        improved = False
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        r_from_others = d_all - r[None, :]
        np.fill_diagonal(r_from_others, np.inf)
        r_max_from_others = np.min(r_from_others, axis=1)
        r_max = np.minimum(r_wall, r_max_from_others)
        r_max = np.maximum(r_max, 0.0)

        for i in range(n):
            if r_max[i] > r[i] + 1e-7:
                lo = r[i]
                hi = r_max[i]
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Phase 3: Final uniform expansion
    for _ in range(25):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.01
        for _ in range(14):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _contact_graph_polish(z, time_budget):
    """
    Identify contact pairs (near-tangent circles) and solve exact tangency
    equations to find a locally optimal configuration.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])

    # Identify contact pairs: distance within 1% of sum of radii
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    d = np.sqrt(np.maximum(d_sq, 1e-12))

    d_pairs = d[iu]
    r_sum_pairs = r_sum[iu]
    gap = d_pairs - r_sum_pairs

    # Contact threshold: within 0.5% of tangency
    contact_thresh = 0.005 * np.maximum(r_sum_pairs, 0.01)
    is_contact = gap < contact_thresh

    contact_i = iu[0][is_contact]
    contact_j = iu[1][is_contact]
    n_contacts = len(contact_i)

    if n_contacts < 3:
        # Not enough contacts to form a rigid structure; just do a quick expansion
        return _quick_expand(z, time_budget)

    # Interleaved contact enforcement and radius expansion
    max_rounds = int(min(30, (time_budget - time.time()) * 20))
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # Re-identify contacts (they may change as radii grow)
        d_check = np.sqrt(np.maximum((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2, 1e-12))
        gap_check = d_check[iu] - r_sum[iu]
        is_contact = gap_check < contact_thresh
        contact_i = iu[0][is_contact]
        contact_j = iu[1][is_contact]
        n_contacts = len(contact_i)

        if n_contacts < 3:
            break

        # Contact enforcement: iterative projection
        max_iters = int(min(50, (time_budget - time.time()) * 30))
        for it in range(max_iters):
            if time.time() > time_budget:
                break

            max_err = 0.0
            for idx in range(n_contacts):
                i = contact_i[idx]
                j = contact_j[idx]
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                d_ij = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)
                if d_ij < 1e-12:
                    continue
                target_d = r[i] + r[j]
                err = target_d - d_ij
                if abs(err) < 1e-10:
                    continue
                max_err = max(max_err, abs(err))
                ux = dx_ij / d_ij
                uy = dy_ij / d_ij
                move = err / 2.0
                x[i] += ux * move
                y[i] += uy * move
                x[j] -= ux * move
                y[j] -= uy * move

            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

            if max_err < 1e-10:
                break

        # Try to expand the most constrained circles
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        r_from_others = d_all - r[None, :]
        np.fill_diagonal(r_from_others, np.inf)
        r_max_from_others = np.min(r_from_others, axis=1)
        r_max = np.minimum(r_wall, r_max_from_others)
        r_max = np.maximum(r_max, 0.0)

        # Find circles with most room to grow
        room = r_max - r
        if np.max(room) < 1e-7:
            break

        # Expand the circle with most room
        i_best = int(np.argmax(room))
        lo = r[i_best]
        hi = r_max[i_best]
        for _ in range(15):
            mid = (lo + hi) / 2.0
            r_trial = r.copy()
            r_trial[i_best] = mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > r[i_best] + 1e-8:
            r[i_best] = lo

    # Final expansion
    result = _quick_expand_from_state(x, y, r, w, h, time_budget)
    if result is not None:
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = result[0]
        z_new[1::3][:n] = result[1]
        z_new[2::3][:n] = result[2]
        z_new[-1] = w
        return z_new
    return None


def _quick_expand(z, time_budget):
    """Quick radius expansion without position changes."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Vectorized max radius computation
    r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
    dx_all = x[:, None] - x[None, :]
    dy_all = y[:, None] - y[None, :]
    d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
    r_from_others = d_all - r[None, :]
    np.fill_diagonal(r_from_others, np.inf)
    r_max_from_others = np.min(r_from_others, axis=1)
    r_max = np.minimum(r_wall, r_max_from_others)
    r_max = np.maximum(r_max, 0.0)

    # Binary search expansion for each circle
    for _ in range(15):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if r_max[i] > r[i] + 1e-7:
                lo = r[i]
                hi = r_max[i]
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    # Uniform expansion
    for _ in range(15):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.005
        for _ in range(10):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _quick_expand_from_state(x, y, r, w, h, time_budget):
    """Quick radius expansion from a given state, returns (x, y, r) or None."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()

    r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
    dx_all = x[:, None] - x[None, :]
    dy_all = y[:, None] - y[None, :]
    d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
    r_from_others = d_all - r[None, :]
    np.fill_diagonal(r_from_others, np.inf)
    r_max_from_others = np.min(r_from_others, axis=1)
    r_max = np.minimum(r_wall, r_max_from_others)
    r_max = np.maximum(r_max, 0.0)

    for _ in range(15):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if r_max[i] > r[i] + 1e-7:
                lo = r[i]
                hi = r_max[i]
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    for _ in range(15):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.005
        for _ in range(10):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        return (x, y, r)
    return None


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    
    # Neighbor check
    for j in range(N):
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = r[i] + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints with tighter tolerances."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(1200, time_budget * 200))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid using vectorized operations."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Vectorized pairwise check
    iu = np.triu_indices(N, k=1)
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    d_sq = dx * dx + dy * dy
    r_sum = r[iu[0]] + r[iu[1]]
    if np.any(d_sq < r_sum * r_sum - tol):
        return False
    return True


def _lse_soft_expansion(z, time_budget):
    """
    LSE-based soft-expansion: jointly grow radii and adjust positions
    using a smooth log-sum-exp penalty for overlap. This provides a
    differentiable surrogate that tightens near convergence.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])

    # Adaptive temperature for LSE
    temp = 0.02
    lr_pos = 0.001
    lr_rad = 0.0005
    min_lr_pos = 1e-5
    max_lr_pos = 0.005
    min_lr_rad = 1e-6
    max_lr_rad = 0.002

    max_iter = int(min(5000, (time_budget - time.time()) * 300))
    if max_iter < 50:
        return None

    prev_val = -np.sum(r)

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # Compute smooth overlap penalty using LSE
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        # Violation: r_sum - d (positive means overlap)
        viol = r_sum_pairs - d_pairs

        # LSE approximation of max(viol, 0) with temperature
        # Use softplus: log(1 + exp(viol/temp)) * temp ≈ max(viol,0) for small temp
        scaled = viol / temp
        # Numerically stable softplus
        softplus = np.where(scaled > 30, scaled, np.log1p(np.exp(scaled)))
        lse_penalty = temp * np.sum(softplus)

        # Gradient of LSE penalty w.r.t. positions
        # d/dx_i of softplus((r_i+r_j - d_ij)/temp) * temp
        # = sigmoid(scaled) * (1/temp) * d/dx_i (r_i+r_j - d_ij) * temp
        # = sigmoid(scaled) * (1 - d_ij/d_ij) for x component... let's compute properly

        sigmoid_vals = np.where(scaled > 30, 1.0, 1.0 / (1.0 + np.exp(-scaled)))

        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        # d(d_ij)/d(x_i) = dx_ij / d_ij, d(d_ij)/d(x_j) = -dx_ij / d_ij
        # d(viol)/d(x_i) = -dx_ij/d_ij (since viol = r_sum - d)
        # d(viol)/d(x_j) = +dx_ij/d_ij

        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        # Gradient of penalty w.r.t. x
        gx = np.zeros(n)
        gy = np.zeros(n)

        # For each pair (i,j), the contribution to grad_x[i] is -sigmoid * ux
        # and to grad_x[j] is +sigmoid * ux
        for idx in range(n_pairs):
            ii = iu[0][idx]
            jj = iu[1][idx]
            sv = sigmoid_vals[idx]
            gx[ii] -= sv * ux[idx]
            gy[ii] -= sv * uy[idx]
            gx[jj] += sv * ux[idx]
            gy[jj] += sv * uy[idx]

        # Gradient w.r.t. radii: d(viol)/d(r_i) = 1, d(viol)/d(r_j) = 1
        gr = np.zeros(n)
        for idx in range(n_pairs):
            ii = iu[0][idx]
            jj = iu[1][idx]
            sv = sigmoid_vals[idx]
            gr[ii] += sv
            gr[jj] += sv

        # Wall constraints (soft penalty)
        wall_soft = 0.01
        left_gap = x - r
        right_gap = w - x - r
        bottom_gap = y - r
        top_gap = h - y - r

        # Soft wall penalty gradients
        wl = np.where(left_gap < wall_soft, (wall_soft - left_gap) / wall_soft, 0.0)
        wr = np.where(right_gap < wall_soft, (wall_soft - right_gap) / wall_soft, 0.0)
        wb = np.where(bottom_gap < wall_soft, (wall_soft - bottom_gap) / wall_soft, 0.0)
        wt = np.where(top_gap < wall_soft, (wall_soft - top_gap) / wall_soft, 0.0)

        gx_wall = -wl + wr
        gy_wall = -wb + wt
        gr_wall = wl + wr + wb + wt

        # Combined gradients
        gx_total = gx + 50.0 * gx_wall
        gy_total = gy + 50.0 * gy_wall
        gr_total = gr + 50.0 * gr_wall

        # Update positions (minimize penalty, so move against gradient)
        x -= lr_pos * gx_total
        y -= lr_pos * gy_total
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Update radii (maximize sum, so move against gradient of penalty)
        # We want to increase r where the gradient of penalty is small
        # Use: r_new = r - lr_rad * gr_total + lr_rad * n_pairs/n (bias toward growth)
        growth_bias = n_pairs / n * 0.5
        r -= lr_rad * (gr_total - growth_bias)
        r = np.clip(r, 1e-4, 1.0)

        # Clip to wall constraints
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        r = np.minimum(r, r_wall)

        # Adaptive learning rate and temperature
        force_mag = float(np.sqrt(np.sum(gx_total**2) + np.sum(gy_total**2))) / n
        if force_mag > 1.0:
            lr_pos = max(min_lr_pos, lr_pos * 0.8)
        elif force_mag < 0.05:
            lr_pos = min(max_lr_pos, lr_pos * 1.05)

        # Tighten temperature as we converge
        max_viol = float(np.max(viol)) if n_pairs > 0 else 0.0
        if max_viol < 0.001:
            temp = max(0.001, temp * 0.95)
        elif max_viol > 0.05:
            temp = min(0.1, temp * 1.05)

        # Check convergence
        curr_val = float(np.sum(r))
        if it % 100 == 0 and curr_val < prev_val - 1e-6:
            # Value decreased, revert slightly
            pass
        prev_val = curr_val

    # Final expansion pass
    for _ in range(20):
        if time.time() > time_budget:
            break
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        r_from_others = d_all - r[None, :]
        np.fill_diagonal(r_from_others, np.inf)
        r_max_from_others = np.min(r_from_others, axis=1)
        r_max = np.minimum(r_wall, r_max_from_others)
        r_max = np.maximum(r_max, 0.0)

        improved = False
        for i in range(n):
            if r_max[i] > r[i] + 1e-7:
                lo = r[i]
                hi = r_max[i]
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    # Uniform expansion
    for _ in range(15):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.005
        for _ in range(12):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _coordinate_descent_polish(z, time_budget):
    """
    Coordinate descent: for each circle, try small perturbations in x, y, and r
    to find local improvements. Uses a smart ordering based on current gaps.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)

    # Compute initial gaps to prioritize circles with more room
    def compute_gaps(x, y, r, w, h):
        gaps = np.zeros(n)
        for i in range(n):
            min_gap = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                gap = d - r[i] - r[j]
                min_gap = min(min_gap, gap)
            gaps[i] = min_gap
        return gaps

    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    max_rounds = int(min(50, (time_budget - time.time()) * 10))
    if max_rounds < 1:
        return None

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # Sort circles by gap (most constrained first)
        gaps = compute_gaps(x, y, r, w, h)
        order = np.argsort(gaps)

        improved = False
        for i in order:
            if time.time() > time_budget:
                break

            # Try expanding radius
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_max = min(r_max, d - r[j])
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                lo, hi = r[i], r_max
                for _ in range(12):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            # Try position perturbations
            for sx in step_sizes[:3]:
                for sy in step_sizes[:3]:
                    for dx in [-sx, 0, sx]:
                        for dy in [-sy, 0, sy]:
                            if dx == 0 and dy == 0:
                                continue
                            nx = x[i] + dx
                            ny = y[i] + dy
                            if nx < r[i] or nx > w - r[i]:
                                continue
                            if ny < r[i] or ny > h - r[i]:
                                continue
                            x_trial = x.copy()
                            y_trial = y.copy()
                            x_trial[i] = nx
                            y_trial[i] = ny
                            if _is_valid(x_trial, y_trial, r, w, h):
                                # Check if this allows more expansion
                                r_wall_new = min(nx, w - nx, ny, h - ny)
                                r_max_new = r_wall_new
                                for j in range(n):
                                    if i == j:
                                        continue
                                    ddx = nx - x[j]
                                    ddy = ny - y[j]
                                    d = np.sqrt(ddx*ddx + ddy*ddy)
                                    r_max_new = min(r_max_new, d - r[j])
                                r_max_new = max(0.0, r_max_new)
                                if r_max_new > r[i] + 1e-7:
                                    x[i] = nx
                                    y[i] = ny
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            break

    # Final expansion
    for _ in range(15):
        if time.time() > time_budget:
            break
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        r_from_others = d_all - r[None, :]
        np.fill_diagonal(r_from_others, np.inf)
        r_max_from_others = np.min(r_from_others, axis=1)
        r_max = np.minimum(r_wall, r_max_from_others)
        r_max = np.maximum(r_max, 0.0)

        improved = False
        for i in range(n):
            if r_max[i] > r[i] + 1e-7:
                lo = r[i]
                hi = r_max[i]
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    for _ in range(15):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.005
        for _ in range(12):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None
