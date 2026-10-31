"""Circle packing in perimeter-4 rectangle: unified penalty-based NLP with force-directed relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    best_value = 0.0
    best_result = None

    # Parameters
    penalty_weight = 500.0

    def unpack(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        return x, y, r, w, h

    def objective(z):
        x, y, r, w, h = unpack(z)

        if w < 0.3 or h < 0.3 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty: smooth
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                dist_req = r[i] + r[j]
                if d_sq < dist_req * dist_req:
                    d = np.sqrt(d_sq)
                    overlap = dist_req - d
                    obj += penalty_weight * overlap * overlap

        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2

        return obj

    def check_validity(z, tol=1e-8):
        x, y, r, w, h = unpack(z)

        if w <= 0.2 or h <= 0.2:
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
                d_sq = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d_sq < r_sum * r_sum - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def force_relax(z, max_iter=200, dt=0.005):
        """Force-directed relaxation to resolve overlaps and boundary violations."""
        z = z.copy().astype(float)
        n = N

        for _ in range(max_iter):
            x, y, r, w, h = unpack(z)

            if w < 0.2 or h < 0.2:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall repulsion
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 50
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 50
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 50
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 50

            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d_sq = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        f = overlap * 20
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f

            # Update positions
            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            # Clamp positions
            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def make_constraints(z):
        """Generate SLSQP constraints for the unified variable vector."""
        x, y, r, w, h = unpack(z)
        constraints = []

        # Boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: (2.0 - p[-1]) - p[3*idx+1] - p[3*idx+2]})

        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (
                        (p[3*i] - p[3*j])**2 +
                        (p[3*i+1] - p[3*j+1])**2 -
                        (p[3*i+2] + p[3*j+2])**2
                    )
                })

        # Positive radius
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+2]})

        # Width bounds
        constraints.append({'type': 'ineq', 'fun': lambda p: p[-1] - 0.3})
        constraints.append({'type': 'ineq', 'fun': lambda p: 1.7 - p[-1]})

        return constraints

    def local_refine(z_init, time_budget):
        """Refine using SLSQP with hard constraints."""
        if time_budget < 3:
            return z_init

        z = z_init.copy()
        constraints = make_constraints(z)

        try:
            result = minimize(
                objective, z, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if result.fun < objective(z):
                z = result.x
        except Exception:
            pass

        # Force relax to clean up any residual violations
        z = force_relax(z, max_iter=100)

        # Final polish
        if time_budget > 10:
            try:
                constraints2 = make_constraints(z)
                result2 = minimize(
                    objective, z, method='SLSQP',
                    constraints=constraints2,
                    options={'maxiter': 2000, 'ftol': 1e-12}
                )
                if result2.fun < objective(z):
                    z = result2.x
                z = force_relax(z, max_iter=50)
            except Exception:
                pass

        return z

    def generate_initial_guesses():
        guesses = []
        rng = np.random.default_rng(12345)

        # Aspect ratios to try
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.9, 1.1, 0.8, 1.4, 0.7]

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue

            # Grid-based initial placement
            n_cols = int(np.round(np.sqrt(N * w0 / h0)))
            n_rows = int(np.ceil(N / n_cols))
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            z = np.zeros(3 * N + 1)
            cell_w = w0 / n_cols
            cell_h = h0 / n_rows
            est_r = 0.42 * min(cell_w, cell_h)

            for i in range(N):
                col = i % n_cols
                row = i // n_cols
                cx = (col + 0.5) * cell_w
                cy = (row + 0.5) * cell_h
                # Stagger odd rows
                if row % 2 == 1:
                    cx += 0.5 * cell_w
                    cx = np.clip(cx, est_r, w0 - est_r)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = est_r
            z[-1] = w0
            guesses.append(z)

            # Random jittered variants
            for _ in range(3):
                zj = z.copy()
                jitter = 0.1 * min(cell_w, cell_h)
                zj[0::3][:N] += rng.normal(0, jitter, N)
                zj[1::3][:N] += rng.normal(0, jitter, N)
                zj[0::3][:N] = np.clip(zj[0::3][:N], est_r, w0 - est_r)
                zj[1::3][:N] = np.clip(zj[1::3][:N], est_r, h0 - est_r)
                guesses.append(zj)

        # Pure random starts
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.1)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    # Main search loop
    guesses = generate_initial_guesses()
    rng = np.random.default_rng(99999)

    # Shuffle guesses for diversity
    order = rng.permutation(len(guesses))
    guesses = [guesses[i] for i in order]

    for idx, z0 in enumerate(guesses):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break

        # Force relax the initial guess to make it more feasible
        z_relaxed = force_relax(z0, max_iter=150)

        # Local refinement
        z_refined = local_refine(z_relaxed, time_budget=remaining - 2)

        # Validate
        valid, val = check_validity(z_refined)
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack([z_refined[0::3][:N], z_refined[1::3][:N], z_refined[2::3][:N]])

        # Also check the relaxed version in case refinement broke something
        if not valid:
            valid2, val2 = check_validity(z_relaxed)
            if valid2 and val2 > best_value:
                best_value = val2
                best_result = np.column_stack([z_relaxed[0::3][:N], z_relaxed[1::3][:N], z_relaxed[2::3][:N]])

    # Fallback: simple grid in square
    if best_result is None:
        w = h = 1.0
        n_cols = 5
        n_rows = 5
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(cells) < N:
                    cx = (j + 0.5) * w / n_cols
                    cy = (i + 0.5) * h / n_rows
                    r = 0.45 * min(w / n_cols, h / n_rows)
                    cells.append((cx, cy, r))
        best_result = np.array(cells)

    return best_result
