"""Circle packing in perimeter-4 rectangle: Dual-Primal Alternating Scheme."""
import numpy as np
import time
import math

N = 21
_IU = np.triu_indices(N, k=1)  # precomputed upper-triangle indices


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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

    def dual_primal_inner(x0, y0, r0, w0, time_budget=20.0):
        """
        Inner problem: Fix aspect ratio w0, solve for circle positions and radii
        using proximal point iteration on the non-overlap constraints.
        """
        n = N
        x = x0.copy().astype(np.float64)
        y = y0.copy().astype(np.float64)
        r = r0.copy().astype(np.float64)
        w = float(w0)
        h = 2.0 - w

        # Parameters
        step_size = 0.01
        decay = 0.999
        max_iter = 10000

        iu_i, iu_j = _IU

        for it in range(max_iter):
            if time.time() > start_time + time_budget:
                break

            # Compute constraint violations and gradients
            gx = np.zeros(n)
            gy = np.zeros(n)
            gr = np.zeros(n)

            # Circle-circle interactions
            for i, j in zip(iu_i, iu_j):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                r_sum = r[i] + r[j]
                diff = d - r_sum  # positive if gap, negative if overlap

                if diff < 0:
                    # Overlap: strong penalty
                    penalty = diff * diff
                    dpen_dd = 2 * diff
                else:
                    # Gap: mild penalty to encourage closeness
                    penalty = 0.01 * diff * diff
                    dpen_dd = 0.02 * diff

                # Gradient w.r.t. distance
                if d > 1e-8:
                    dx_dd = dx / d
                    dy_dd = dy / d
                    dr_i_dd = 1.0
                    dr_j_dd = 1.0
                else:
                    dx_dd, dy_dd, dr_i_dd, dr_j_dd = 0, 0, 0, 0

                # Chain rule
                gx[i] += dpen_dd * dx_dd
                gy[i] += dpen_dd * dy_dd
                gx[j] -= dpen_dd * dx_dd
                gy[j] -= dpen_dd * dy_dd
                gr[i] += dpen_dd * dr_i_dd
                gr[j] += dpen_dd * dr_j_dd

            # Wall interactions
            for i in range(n):
                # Left wall
                diff_l = x[i] - r[i]
                if diff_l < 0:
                    gx[i] += 2 * diff_l
                    gr[i] += -2 * diff_l
                # Right wall
                diff_r = (w - x[i]) - r[i]
                if diff_r < 0:
                    gx[i] -= 2 * diff_r
                    gr[i] += -2 * diff_r
                # Bottom wall
                diff_b = y[i] - r[i]
                if diff_b < 0:
                    gy[i] += 2 * diff_b
                    gr[i] += -2 * diff_b
                # Top wall
                diff_t = (h - y[i]) - r[i]
                if diff_t < 0:
                    gy[i] -= 2 * diff_t
                    gr[i] += -2 * diff_t

            # Radius reward: maximize sum(r)
            gr -= 1.0

            # Proximal point update
            x -= step_size * gx
            y -= step_size * gy
            r -= step_size * gr

            # Clamp to valid ranges
            r = np.maximum(r, 0.01)
            x = np.clip(x, 0.01, w - 0.01)
            y = np.clip(y, 0.01, h - 0.01)

            # Decay step size
            step_size *= decay

        return x, y, r

    def get_initial_guesses():
        guesses = []
        # Hexagonal lattice initialization
        side = int(math.ceil(math.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.65, 1.4, 0.6, 1.6, 0.55, 0.9, 1.1]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            # Hexagonal packing
            r0 = 0.08
            dx = 2 * r0
            dy = dx * math.sqrt(3) / 2
            for i in range(N):
                row = i // int(math.ceil(w0 / dx))
                col = i % int(math.ceil(w0 / dx))
                cx = (col + 0.5) * dx
                cy = (row + 0.5) * dy + (0.5 * dy if row % 2 else 0)
                if cx + r0 > w0:
                    cx = w0 - r0
                if cy + r0 > h0:
                    cy = h0 - r0
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0))

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(8):
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

        # Allocate time for inner problem based on remaining budget
        remaining = deadline - time.time()
        inner_time = min(15.0, remaining - 10.0)
        if inner_time < 3.0:
            break

        x, y, r = dual_primal_inner(x0, y0, r0, w0, time_budget=inner_time)
        h = 2.0 - w0

        # Expand radii to maximum
        x, y, r = expand_all(x, y, r, w0, h, max_rounds=50)

        # Check validity
        valid, val = check_validity(x, y, r, w0, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w0, h)

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
