"""Circle packing in perimeter-4 rectangle: SLSQP with free aspect ratio + constraint-based optimization + radius-growth polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill


def _force_relax(z, max_iter=150):
    """Force-directed relaxation in unified parameter space to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003

    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
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
    """
    Polish a valid packing by iteratively growing each radius to its local slack
    and re-relaxing positions. This extracts the last bit of room that SLSQP
    may have left on the table.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    start = time.time()
    max_outer = 30

    for outer in range(max_outer):
        if time.time() - start > time_budget:
            break

        # Try to grow each radius to its local slack
        grew = False
        for i in range(n):
            # Boundary slack
            slack = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])

            # Pairwise slack
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                slack = min(slack, d - r[i] - r[j])

            if slack > 1e-7:
                growth = min(0.002, slack * 0.5)
                r[i] += growth
                grew = True

        # After growing, do a light position relaxation to resolve any new overlaps
        for _ in range(20):
            if time.time() - start > time_budget:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)

            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 20
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 20
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 20
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 20

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

            dt = 0.001
            x += dt * fx
            y += dt * fy

            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        if not grew:
            break

    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r

    return z


def construct():
    best_value = 0.0
    best_result = None

    # --- Phase 1: SLSQP with explicit inequality constraints ---
    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w

    n_starts = 12

    def objective(z):
        r = z[2::3][:N]
        return -np.sum(r)

    def cons_inside(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        return np.concatenate([
            x - r,
            w - x - r,
            y - r,
            h - y - r
        ])

    def cons_non_overlap(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)

    def cons_positive_r(z):
        r = z[2::3][:N]
        return r

    def cons_aspect(z):
        w = z[-1]
        h = 2.0 - w
        return np.array([w - 0.4, 1.6 - w])

    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r},
        {'type': 'ineq', 'fun': cons_aspect}
    ]

    def check_validity(z, tol=1e-7):
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
        if time.time() > DEADLINE - 10:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=150)

        # SLSQP optimization with explicit constraints
        maxiter = int(min(200, (DEADLINE - (time.time() - START)) * 20))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.4, 1.6)],
                           constraints=constraints,
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

    # --- Phase 2: Local polish on best result ---
    if best_result is not None and time.time() < DEADLINE - 5:
        polished = _polish_result(best_result, DEADLINE - (time.time() - START) - 2)
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
