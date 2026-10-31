"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Expand-to-Max polish."""
import numpy as np
import time
import math

N = 21
_IU = np.triu_indices(N, k=1)  # precomputed upper-triangle indices


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # ---------- helpers ----------
    def check_validity(x, y, r, w, h, tol=1e-8):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def expand_all(x, y, r, w, h, max_rounds=40):
        """Push every radius to its geometric maximum, iterating to convergence."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(max_rounds):
            improved = False
            for i in range(N):
                # Max radius from walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                # Max radius from other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = math.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def sa_search(x0, y0, r0, w0, time_budget=20.0):
        """Simulated annealing over (x, y, r, w)."""
        n = N
        # State vector: [x0..xN-1, y0..yN-1, r0..rN-1, w]
        z = np.zeros(3 * n + 1)
        z[0:n] = x0
        z[n:2*n] = y0
        z[2*n:3*n] = r0
        z[3*n] = w0

        T0 = 0.05
        T_min = 1e-7
        # Number of iterations based on time budget
        # Estimate ~10000 iterations per second for this size
        max_iter = int(time_budget * 8000)
        # Geometric cooling
        if max_iter > 1:
            alpha = (T_min / T0) ** (1.0 / max_iter)
        else:
            alpha = 1.0

        T = T0
        best_z = z.copy()
        best_obj = 1e10
        current_obj = 1e10

        rng = np.random.default_rng(42)

        # Precompute index arrays for speed
        iu_i, iu_j = _IU

        for it in range(max_iter):
            if time.time() > start_time + time_budget:
                break

            # --- Generate a random perturbation ---
            # Move a random subset of variables
            n_perturb = rng.integers(1, 4)  # 1 to 3 variables
            for _ in range(n_perturb):
                var_idx = rng.integers(0, 3 * n + 1)
                if var_idx < n:
                    # x coordinate
                    scale = 0.02
                    z[var_idx] += rng.normal(0, scale)
                elif var_idx < 2 * n:
                    # y coordinate
                    scale = 0.02
                    z[var_idx] += rng.normal(0, scale)
                elif var_idx < 3 * n:
                    # radius
                    scale = 0.005
                    z[var_idx] += rng.normal(0, scale)
                else:
                    # width
                    scale = 0.01
                    z[var_idx] += rng.normal(0, scale)

            # --- Compute objective (energy) ---
            x = z[0:n]
            y = z[n:2*n]
            r = z[2*n:3*n]
            w = z[3*n]
            h = 2.0 - w

            if w < 0.1 or h < 0.1 or np.any(r < 0):
                obj = 1e6
            else:
                obj = 0.0
                # Overlap penalty (quadratic)
                for i, j in zip(iu_i, iu_j):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = math.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        obj += overlap * overlap
                # Wall penalty (quadratic)
                for i in range(n):
                    if x[i] < r[i]:
                        obj += (r[i] - x[i]) ** 2
                    if x[i] > w - r[i]:
                        obj += (x[i] - (w - r[i])) ** 2
                    if y[i] < r[i]:
                        obj += (r[i] - y[i]) ** 2
                    if y[i] > h - r[i]:
                        obj += (y[i] - (h - r[i])) ** 2
                # Subtract sum of radii (we want to maximize)
                obj -= 2.0 * np.sum(r)

            # --- Acceptance criterion ---
            if obj < current_obj:
                current_obj = obj
                z_new = z.copy()
                if obj < best_obj:
                    best_obj = obj
                    best_z = z.copy()
            else:
                # Metropolis acceptance
                if T > T_min:
                    delta = obj - current_obj
                    if delta < 0:
                        z_new = z.copy()
                    else:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            z_new = z.copy()
                        else:
                            z_new = None
                else:
                    z_new = None

            if z_new is not None:
                z = z_new
            # else: keep z as is (rejection)

            # Cool
            T *= alpha

        # Return best state
        x = best_z[0:n]
        y = best_z[n:2*n]
        r = best_z[2*n:3*n]
        w = best_z[3*n]
        h = 2.0 - w
        return x, y, r, w, h

    def get_initial_guesses():
        guesses = []
        side = int(math.ceil(math.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.65, 1.4, 0.6, 1.6, 0.55]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.40
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0))

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(6):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.09)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0))

        return guesses

    # ---------- Main search loop ----------
    best_value = 0.0
    best_result = None

    guesses = get_initial_guesses()

    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        # Allocate time for SA based on remaining budget
        remaining = deadline - time.time()
        sa_time = min(15.0, remaining - 10.0)
        if sa_time < 3.0:
            break

        x, y, r, w, h = sa_search(x0, y0, r0, w0, time_budget=sa_time)

        # Expand radii to maximum
        x, y, r = expand_all(x, y, r, w, h, max_rounds=50)

        # Check validity
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # ---------- Final polish ----------
    if best_result is not None and time.time() < deadline - 5:
        x, y, r, w, h = best_result
        # One more expansion pass
        x, y, r = expand_all(x, y, r, w, h, max_rounds=50)
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
