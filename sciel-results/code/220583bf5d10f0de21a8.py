"""Circle packing in perimeter-4 rectangle: SIS Rigid-Body + Contact-Graph Local Search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _sequential_impulse_solve(z0, time_budget, n_impulse_iters=200, dt=0.02, damping=0.98):
    """
    Sequential Impulse Solver (SIS): treats circles as dynamic particles.
    Enforces non-penetration via velocity-level impulses (no penalty terms).
    
    Each step:
      1. Apply gravity-like force toward center to create pressure
      2. Compute relative velocities at contact points
      3. Apply corrective impulses to eliminate approach velocity
      4. Integrate positions
      5. Project onto walls (position correction)
    
    This naturally finds a compressed, valid configuration.
    """
    n = N
    x = z0[0::3][:n].copy().astype(float)
    y = z0[1::3][:n].copy().astype(float)
    r = z0[2::3][:n].copy().astype(float)
    w = float(z0[-1])
    h = 2.0 - w

    # Velocities
    vx = np.zeros(n)
    vy = np.zeros(n)
    # Masses (uniform)
    m = np.ones(n)

    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])

    best_val = float(np.sum(r)) if _is_valid(x, y, r, w, h) else -1e10
    best_state = (x.copy(), y.copy(), r.copy(), w)

    max_steps = int(min(5000, time_budget * 200))
    for step in range(max_steps):
        if time.time() > time_budget:
            break

        # --- Force phase: apply pressure toward center to compress ---
        cx_center = w / 2.0
        cy_center = h / 2.0
        for i in range(n):
            dx_c = cx_center - x[i]
            dy_c = cy_center - y[i]
            dist_c = np.sqrt(dx_c * dx_c + dy_c * dy_c)
            if dist_c > 1e-8:
                # Pressure force proportional to inverse distance (creates compression)
                f_mag = 0.5 / (dist_c + 0.1)
                vx[i] += dt * f_mag * dx_c / dist_c
                vy[i] += dt * f_mag * dy_c / dist_c

        # --- Impulse phase: resolve contacts ---
        for _imp in range(10):  # Multiple impulse iterations for convergence
            # Circle-circle contacts
            dx_mat = x[:, None] - x[None, :]
            dy_mat = y[:, None] - y[None, :]
            d_sq_mat = dx_mat * dx_mat + dy_mat * dy_mat
            d_pairs = np.sqrt(np.maximum(d_sq_mat[iu], 1e-12))
            r_sum = r[iu[0]] + r[iu[1]]

            # Find contacts (penetration > 0 or near-contact)
            pen = r_sum - d_pairs  # positive = overlap
            contact_mask = pen > -0.001  # include near-contacts

            for idx in np.where(contact_mask)[0]:
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                d = d_pairs[idx]
                if d < 1e-10:
                    continue
                nx = dx / d
                ny = dy / d

                # Relative velocity at contact
                rel_vx = vx[i_idx] - vx[j_idx]
                rel_vy = vy[i_idx] - vy[j_idx]
                rel_vn = rel_vx * nx + rel_vy * ny

                # Only correct if approaching (rel_vn < 0)
                if rel_vn < 0:
                    # Impulse magnitude (equal mass)
                    j_imp = -(1.0 + 0.0) * rel_vn / (1.0 / m[i_idx] + 1.0 / m[j_idx])
                    vx[i_idx] += j_imp * nx / m[i_idx]
                    vy[i_idx] += j_imp * ny / m[i_idx]
                    vx[j_idx] -= j_imp * nx / m[j_idx]
                    vy[j_idx] -= j_imp * ny / m[j_idx]

            # Wall contacts (position-level impulse)
            for i in range(n):
                # Left wall
                if x[i] < r[i]:
                    vx[i] = max(vx[i], 0.0)
                # Right wall
                if x[i] > w - r[i]:
                    vx[i] = min(vx[i], 0.0)
                # Bottom wall
                if y[i] < r[i]:
                    vy[i] = max(vy[i], 0.0)
                # Top wall
                if y[i] > h - r[i]:
                    vy[i] = min(vy[i], 0.0)

        # --- Integration ---
        x += dt * vx
        y += dt * vy

        # Damping
        vx *= damping
        vy *= damping

        # --- Position projection onto walls ---
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # --- Position correction for circle-circle overlaps (Baumgarte) ---
        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        d_sq_mat = dx_mat * dx_mat + dy_mat * dy_mat
        d_pairs = np.sqrt(np.maximum(d_sq_mat[iu], 1e-12))
        r_sum = r[iu[0]] + r[iu[1]]
        pen = r_sum - d_pairs

        for idx in np.where(pen > 0)[0]:
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            d = d_pairs[idx]
            if d < 1e-10:
                continue
            nx = dx / d
            ny = dy / d
            corr = pen[idx] * 0.5 * 0.8  # Baumgarte factor
            x[i_idx] += corr * nx
            y[i_idx] += corr * ny
            x[j_idx] -= corr * nx
            y[j_idx] -= corr * ny

        # Re-project after correction
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # Track best valid state
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_state = (x.copy(), y.copy(), r.copy(), w)

    if best_state is not None:
        x_best, y_best, r_best, w_best = best_state
        if _is_valid(x_best, y_best, r_best, w_best, 2.0 - w_best):
            z_out = np.empty(3 * N + 1)
            z_out[0::3][:n] = x_best
            z_out[1::3][:n] = y_best
            z_out[2::3][:n] = r_best
            z_out[-1] = w_best
            return z_out
    return None


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


def _contact_graph_local_search(x, y, r, w, h, time_budget, max_rounds=50):
    """
    Contact-graph-aware local search: identifies the active constraint manifold
    (which circles are touching walls or each other) and performs targeted
    position adjustments to free up space for radius growth.
    
    Key insight: in a tight packing, each circle is constrained by its
    "contacts" (circles or walls it touches). By identifying the contact graph,
    we can determine which circles have the most freedom and grow them first,
    then propagate adjustments through the contact network.
    """
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    tol = 0.005  # Contact tolerance

    best_val = float(np.sum(r))

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # Build contact graph
        # For each circle, find its contacts (walls and other circles)
        contacts = [[] for _ in range(n)]  # list of (type, index) tuples
        contact_count = np.zeros(n, dtype=int)

        # Wall contacts
        for i in range(n):
            if abs(x[i] - r[i]) < tol:
                contacts[i].append(('wall', 'left'))
                contact_count[i] += 1
            if abs(x[i] - (w - r[i])) < tol:
                contacts[i].append(('wall', 'right'))
                contact_count[i] += 1
            if abs(y[i] - r[i]) < tol:
                contacts[i].append(('wall', 'bottom'))
                contact_count[i] += 1
            if abs(y[i] - (h - r[i])) < tol:
                contacts[i].append(('wall', 'top'))
                contact_count[i] += 1

        # Circle-circle contacts
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if abs(d - (r[i] + r[j])) < tol:
                    contacts[i].append(('circle', j))
                    contacts[j].append(('circle', i))
                    contact_count[i] += 1
                    contact_count[j] += 1

        # Find the circle with the fewest contacts (most freedom)
        # and try to grow it
        # Weight by potential gain: circles with fewer contacts can grow more
        freedom = np.zeros(n)
        for i in range(n):
            # Max possible radius given current position
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            freedom[i] = r_max - r[i]  # How much it can grow in place

        # Priority: high freedom, low contact count
        priority = freedom - 0.01 * contact_count
        target = int(np.argmax(priority))

        if freedom[target] < 1e-5:
            # No circle can grow in place; try to move the most constrained
            # circle to create space
            target = int(np.argmin(freedom))
            if freedom[target] < 1e-5:
                break

        # Try to grow target circle
        r_trial = r.copy()
        r_trial[target] = r[target] + min(freedom[target], 0.01)
        if _is_valid(x, y, r_trial, w, h):
            r[target] = r_trial[target]
        else:
            # Can't grow in place; try to move target circle
            moved = False
            # Find direction that maximizes new freedom
            best_gain = 0.0
            best_dx, best_dy = 0, 0
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                dx_d = np.cos(angle) * 0.02
                dy_d = np.sin(angle) * 0.02
                nx = x[target] + dx_d
                ny = y[target] + dy_d
                if nx < r[target] or nx > w - r[target]:
                    continue
                if ny < r[target] or ny > h - r[target]:
                    continue
                # Check validity of move
                valid_move = True
                for j in range(n):
                    if j == target:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    d2 = dx * dx + dy * dy
                    if d2 < (r[target] + r[j]) ** 2 - 1e-8:
                        valid_move = False
                        break
                if not valid_move:
                    continue
                # Compute new freedom after move
                r_wall = min(nx, w - nx, ny, h - ny)
                r_max = r_wall
                for j in range(n):
                    if j == target:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                new_freedom = max(0.0, r_max - r[target])
                if new_freedom > best_gain:
                    best_gain = new_freedom
                    best_dx, best_dy = dx_d, dy_d

            if best_gain > 1e-5:
                x[target] += best_dx
                y[target] += best_dy
                # Now try to grow
                r_wall = min(x[target], w - x[target], y[target], h - y[target])
                r_max = r_wall
                for j in range(n):
                    if j == target:
                        continue
                    dx = x[target] - x[j]
                    dy = y[target] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                grow = min(r_max - r[target], 0.01)
                if grow > 1e-6:
                    r[target] += grow
                moved = True

            if not moved and freedom[target] < 1e-4:
                # Try moving a different circle to create space for target
                # Move the circle that's blocking target the most
                for j in range(n):
                    if j == target:
                        continue
                    dx = x[target] - x[j]
                    dy = y[target] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < (r[target] + r[j]) + tol:
                        # j is blocking target; try to move j away
                        nx = x[j] - dx / d * 0.02
                        ny = y[j] - dy / d * 0.02
                        if nx < r[j] or nx > w - r[j] or ny < r[j] or ny > h - r[j]:
                            continue
                        valid_move = True
                        for k in range(n):
                            if k == j:
                                continue
                            ddx = nx - x[k]
                            ddy = ny - y[k]
                            d2 = ddx * ddx + ddy * ddy
                            if d2 < (r[j] + r[k]) ** 2 - 1e-8:
                                valid_move = False
                                break
                        if valid_move:
                            x[j] = nx
                            y[j] = ny
                            moved = True
                            break

        # Uniform growth attempt
        for _ in range(5):
            if time.time() > time_budget:
                break
            lo, hi = 1.0, 1.02
            for _ in range(10):
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-9:
                r = r * lo
            else:
                break

        val = float(np.sum(r))
        if val > best_val:
            best_val = val

    if _is_valid(x, y, r, w, h):
        z_out = np.empty(3 * N + 1)
        z_out[0::3][:n] = x
        z_out[1::3][:n] = y
        z_out[2::3][:n] = r
        z_out[-1] = w
        return z_out
    return None


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

        # SLSQP optimization (kept for initial exploration)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 60})

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

        # Also try SIS from this starting point for diverse basins
        if time.time() < deadline - 15:
            try:
                sis_init = _sequential_impulse_solve(z0_relaxed, deadline - (time.time() - start_time) - 12, dt=0.03, damping=0.97)
                if sis_init is not None:
                    valid, val = check_validity(sis_init)
                    if valid and val > best_value:
                        best_value = val
                        best_result = sis_init.copy()
            except Exception:
                pass

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Sequential Impulse Solver: compress and find tight valid config
        if time.time() < deadline - 25:
            sis_result = _sequential_impulse_solve(best_result, deadline - (time.time() - start_time) - 20)
            if sis_result is not None:
                valid, val = check_validity(sis_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = sis_result

        # 2. Contact-graph local search: exploit constraint manifold structure
        if best_result is not None and time.time() < deadline - 20:
            cg_result = _contact_graph_local_search(
                best_result[0::3][:N].copy(),
                best_result[1::3][:N].copy(),
                best_result[2::3][:N].copy(),
                best_result[-1],
                2.0 - best_result[-1],
                deadline - (time.time() - start_time) - 15,
                max_rounds=60
            )
            if cg_result is not None:
                valid, val = check_validity(cg_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result

        # 3. Force-directed position polish
        if best_result is not None and time.time() < deadline - 18:
            polished = _polish_result(best_result, deadline - (time.time() - start_time) - 13)
            if polished is not None:
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

        # 4. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 14:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 5. Second SIS pass (different dynamics)
        if best_result is not None and time.time() < deadline - 10:
            sis_result2 = _sequential_impulse_solve(best_result, deadline - (time.time() - start_time) - 7, dt=0.01, damping=0.95)
            if sis_result2 is not None:
                valid, val = check_validity(sis_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = sis_result2

        # 6. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 7:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 7. Second contact-graph pass
        if best_result is not None and time.time() < deadline - 4:
            cg_result2 = _contact_graph_local_search(
                best_result[0::3][:N].copy(),
                best_result[1::3][:N].copy(),
                best_result[2::3][:N].copy(),
                best_result[-1],
                2.0 - best_result[-1],
                deadline - (time.time() - start_time) - 2,
                max_rounds=40
            )
            if cg_result2 is not None:
                valid, val = check_validity(cg_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result2

        # 8. Final quick polish
        if best_result is not None and time.time() < deadline - 2:
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

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
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

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
    max_iter = int(min(5000, time_budget * 200))
    prev_best = float(np.sum(r))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        # Only accept if valid
        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Try to grow radii uniformly with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        # Binary search for max uniform growth
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
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


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Improved: 8-direction moves, binary search for max radius, two expansion passes.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 8 directions
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    def try_expand_circle(i):
        """Try to expand circle i to max possible radius."""
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        r_max = max(0.0, r_max)
        
        if r_max > r[i] + 1e-6:
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
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        """Try to move circle i in various directions to create space."""
        # Find the direction of greatest constraint
        best_dir = None
        best_score = -float('inf')
        
        for dx_d, dy_d in dirs:
            # Score: how much space would be created
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue
            
            # Check if this move creates more space
            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
            
            # Also consider wall distances
            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)
            
            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)
        
        if best_dir is None:
            return False
        
        dx_d, dy_d = best_dir
        
        # Try multiple step sizes
        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True
        
        return False

    # Pass 1: Expand and move
    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            
            # Try to expand
            if try_expand_circle(i):
                improved = True
            
            # If still limited, try to move
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            
            if r_max - r[i] < 0.01:  # Limited expansion
                if try_move_circle(i):
                    improved = True
        
        if not improved:
            break

    # Pass 2: Second expansion after repositioning
    for rnd in range(20):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True
        
        if not improved:
            break

    # Final uniform expansion with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
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
    """High-precision SLSQP refinement using explicit inequality constraints."""
    z = z.copy().astype(float)
    n = N
    n_pairs = n * (n - 1) // 2

    def obj_refine(zv):
        """Objective: minimize negative sum of radii."""
        r = zv[2::3][:n]
        return -np.sum(r)

    def make_circle_constraint(i):
        """Circle i must not overlap with any other circle."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            # For each pair (i, j), enforce dist(i,j) >= r[i] + r[j]
            # Equivalent: dist^2 - (r[i]+r[j])^2 >= 0
            # But SLSQP needs g(z) >= 0, so we use dist - (r[i]+r[j]) >= 0
            # Actually, let's use the squared form for smoothness but it's not convex
            # Better: use dist - (r[i]+r[j]) >= 0 which is smooth when dist > 0
            
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            # For all j != i: dist_j - (r[i] + r[j]) >= 0
            result = dist - (r[i] + r)
            result[i] = 1.0  # Self-constraint is always satisfied
            return result
        return constraint

    def make_wall_constraint(axis, side):
        """Wall constraint: circle must stay inside rectangle."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w
            
            if axis == 'x':
                pos = x
            else:
                pos = y
            
            if side == 'left' or side == 'bottom':
                # pos - r >= 0
                return pos - r
            else:
                # (w or h) - pos - r >= 0
                if axis == 'x':
                    return w - pos - r
                else:
                    return h - pos - r
        return constraint

    # Build constraints
    constraints = []
    
    # Circle-circle constraints (one per circle, each returning n-1 values)
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_circle_constraint(i)})
    
    # Wall constraints (4 per circle, but we can vectorize)
    # Left wall: x[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'left')})
    # Right wall: w - x[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'right')})
    # Bottom wall: y[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'bottom')})
    # Top wall: h - y[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'top')})

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        # Try to repair if slightly invalid
        if not _is_valid(x, y, r, w, h):
            # Shrink radii slightly
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break
        
        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
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
