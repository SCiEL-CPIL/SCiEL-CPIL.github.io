import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
# Pre-compute upper triangle indices for pairwise constraints
_IU = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 500.0
    soft_beta = 80.0  # soft-min sharpness for overlap penalty

    # --- Vectorized objective with smooth soft-min surrogate ---
    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Pairwise overlap: smooth soft-min surrogate
        # overlap_ij = max(0, r_i + r_j - d_ij); smooth with softmin
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-12)
        dist_req = r[_IU[0]] + r[_IU[1]]
        # soft-min: for a = dist_req - d, softmax(a*beta)/beta approximates max(a,0)
        # use log-sum-exp trick for stability
        a = dist_req - d  # positive when overlapping
        # smooth max(0, a): use softplus(a*beta)/beta
        # softplus(t) = log(1+exp(t)), stable form
        t = a * soft_beta
        # softplus stable: where t>0: t + log1p(exp(-t)); else log1p(exp(t))
        sp = np.where(t > 0, t + np.log1p(np.exp(-t)), np.log1p(np.exp(t)))
        soft_overlap = sp / soft_beta
        obj += penalty_weight * np.sum(soft_overlap)

        # Boundary penalty (smooth)
        # x < r  -> (r - x) positive
        viol_xl = np.maximum(0.0, r - x)
        viol_xr = np.maximum(0.0, x - (w - r))
        viol_yl = np.maximum(0.0, r - y)
        vir_yr = np.maximum(0.0, y - (h - r))
        # smooth these too
        for v in (viol_xl, viol_xr, viol_yl, vir_yr):
            tv = v * soft_beta
            sp = np.where(tv > 0, tv + np.log1p(np.exp(-tv)), np.log1p(np.exp(tv)))
            v[:] = sp / soft_beta
        obj += penalty_weight * (np.sum(viol_xl) + np.sum(viol_xr) + np.sum(viol_yl) + np.sum(vir_yr))

        # Aspect ratio penalty
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

    def _sa_explore(time_budget):
        """Simulated annealing to find diverse near-valid packings.
        Returns list of z vectors (positions+radii+w) that are near-valid."""
        results = []
        rng = random.Random(42)
        start = time.time()

        side = 5
        # Start from grid
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            cell_w = w0 / side
            cell_h = h0 / side
            pos = np.zeros((N, 2))
            rad = np.zeros(N)
            for k in range(N):
                row, col = divmod(k, side)
                pos[k, 0] = (col + 0.5) * cell_w
                pos[k, 1] = (row + 0.5) * cell_h
                rad[k] = min(cell_w, cell_h) * 0.45

            T = 0.05
            T_min = 1e-6
            alpha = 0.9995
            max_iter = 30000

            best_pos = pos.copy()
            best_rad = rad.copy()
            best_obj = _sa_evaluate(pos, rad, w0)

            for it in range(max_iter):
                if time.time() - start > time_budget:
                    break
                k = rng.randint(0, N - 1)
                step = 0.02 * T * 100
                new_pos = pos.copy()
                new_pos[k, 0] += random.gauss(0, step)
                new_pos[k, 1] += random.gauss(0, step)
                if random.random() < 0.5:
                    new_rad = rad.copy()
                    new_rad[k] += random.gauss(0, 0.005)
                    if new_rad[k] < 0.01:
                        new_rad[k] = 0.01
                else:
                    new_rad = rad

                new_obj = _sa_evaluate(new_pos, new_rad, w0)
                delta_obj = new_obj - best_obj if best_obj is not None else 0
                # accept
                if delta_obj > 0 or random.random() < np.exp(delta_obj / T):
                    pos = new_pos
                    rad = new_rad
                    obj = _sa_evaluate(pos, rad, w0)
                    if obj is not None and obj > best_obj:
                        best_obj = obj
                        best_pos = pos.copy()
                        best_rad = rad.copy()
                T *= alpha
                if T < T_min:
                    T = 0.01

            # Build z
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = best_pos[:, 0]
            z[1::3][:N] = best_pos[:, 1]
            z[2::3][:N] = best_rad
            z[-1] = w0
            results.append(z)

        return results

    def _sa_evaluate(pos, rad, w):
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            return -1e6
        # boundary
        dist_x = np.minimum(pos[:, 0], w - pos[:, 0])
        dist_y = np.minimum(pos[:, 1], h - pos[:, 1])
        min_dist_wall = np.minimum(dist_x, dist_y)
        bound_viol = np.maximum(0, rad - min_dist_wall)
        # pairwise
        dx = pos[_IU[0], 0] - pos[_IU[1], 0]
        dy = pos[_IU[0], 1] - pos[_IU[1], 1]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-12)
        sum_rad = rad[_IU[0]] + rad[_IU[1]]
        pair_viol = np.sum(np.maximum(0, sum_rad - d))
        total_viol = np.sum(bound_viol) + pair_viol
        obj = np.sum(rad) - 100.0 * total_viol
        return obj

    def _greedy_grow(z, time_budget):
        """Greedily grow radii in random order to extract value."""
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w
        start = time.time()
        rng = np.random.default_rng(7)

        for _ in range(30):
            if time.time() - start > time_budget:
                break
            order = rng.permutation(n)
            for k in order:
                # max radius given walls
                max_r = min(x[k], w - x[k], y[k], h - y[k])
                # constraints with others
                for j in range(n):
                    if j == k:
                        continue
                    dx = x[k] - x[j]
                    dy = y[k] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    constraint = d - r[j]
                    if constraint < max_r:
                        max_r = constraint
                if max_r > r[k] + 1e-9:
                    r[k] = max_r - 1e-9

        z[0::3][:n] = x
        z[1::3][:n] = y
        z[2::3][:n] = r
        z[-1] = w
        return z

    def _polish_result(z, time_budget):
        """Polish a valid packing with force-directed radius growth + SLSQP passes."""
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        dt = 0.001
        max_iter = 300
        start = time.time()

        for it in range(max_iter):
            if time.time() - start > time_budget * 0.5:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 50
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 50
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 50
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 50

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

            # grow radii
            growth = 0.0001
            for i in range(n):
                can_grow = True
                if x[i] - (r[i] + growth) < 0 or x[i] + (r[i] + growth) > w:
                    can_grow = False
                if y[i] - (r[i] + growth) < 0 or y[i] + (r[i] + growth) > h:
                    can_grow = False
                if can_grow:
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < r[i] + r[j] + 2 * growth:
                            can_grow = False
                            break
                if can_grow:
                    r[i] += growth

            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        z_polished = np.zeros_like(z)
        z_polished[0::3][:n] = x
        z_polished[1::3][:n] = y
        z_polished[2::3][:n] = r
        z_polished[-1] = w

        # Multi-pass SLSQP polish with decreasing tolerance
        if time.time() - start < time_budget * 0.8:
            for pass_i, tol in enumerate([1e-8, 1e-10, 1e-12]):
                if time.time() - start > time_budget:
                    break
                try:
                    res = minimize(objective, z_polished, method='SLSQP',
                                   bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                   options={'maxiter': 100, 'ftol': tol, 'maxls': 30})
                    z_polished = res.x
                except Exception:
                    pass

        return z_polished

    # --- Generate initial guesses ---
    def get_initial_guesses():
        guesses = []
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

        rng = np.random.default_rng(42)
        for _ in range(max(0, 12 - len(guesses))):
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

    # --- Phase 0: Simulated Annealing exploration ---
    if time.time() < deadline - 60:
        sa_results = _sa_explore(deadline - (time.time() - start_time) - 60)
        for z_sa in sa_results:
            z_relaxed = _force_relax_unified(z_sa, max_iter=100)
            valid, val = check_validity(z_relaxed)
            if valid and val > best_value:
                best_value = val
                best_result = z_relaxed

    # --- Phase 1: SLSQP multi-start ---
    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=200)

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
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair

        except Exception:
            continue

    # --- Phase 2: Greedy radius growth on best ---
    if best_result is not None and time.time() < deadline - 15:
        grown = _greedy_grow(best_result, deadline - (time.time() - start_time) - 15)
        valid, val = check_validity(grown)
        if valid and val > best_value:
            best_value = val
            best_result = grown

    # --- Phase 3: Local polish on best result ---
    if best_result is not None and time.time() < deadline - 5:
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Safety shrink
    r = r * (1 - 1e-6)

    return np.column_stack((x, y, r))
