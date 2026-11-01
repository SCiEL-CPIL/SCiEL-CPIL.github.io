"""Circle packing in unit square, n=32.

Structural change: formulate as a single-objective NLP maximizing a global
radius variable R with per-circle scaling, using SLSQP interior-point
mechanism. This drives toward densest uniform packing rather than
maximizing sum of heterogeneous radii.
"""

N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    best_result = None
    best_score = -np.inf

    # ------------------------------------------------------------------
    # Unpack helper: params = [x0..xN-1, y0..yN-1, s0..sN-1, R]
    # where s_i is a scaling factor and R is global radius
    # Actual radius of circle i = s_i * R
    # ------------------------------------------------------------------
    def unpack(params):
        x = params[:N]
        y = params[N:2*N]
        s = params[2*N:3*N]
        R = params[3*N]
        r = s * R
        return x, y, r, s, R

    # ------------------------------------------------------------------
    # Objective: maximize R (so minimize -R)
    # ------------------------------------------------------------------
    def objective(params):
        return -params[3*N]

    # ------------------------------------------------------------------
    # Constraints:
    # 1. Pairwise: dist(i,j) >= r_i + r_j  =>  dist^2 >= (r_i+r_j)^2
    #    We use: dist^2 - (r_i + r_j)^2 >= 0
    # 2. Boundary: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    # 3. Scaling: s_i >= 0.5 (minimum relative size), s_i <= 2.0
    # ------------------------------------------------------------------
    def make_constraints():
        cons = []
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                def pair_con(params, i=i, j=j):
                    x = params[:N]
                    y = params[N:2*N]
                    s = params[2*N:3*N]
                    R = params[3*N]
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    r_i = s[i] * R
                    r_j = s[j] * R
                    return dx*dx + dy*dy - (r_i + r_j)**2
                cons.append({'type': 'ineq', 'fun': pair_con})
        # Boundary constraints
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]*p[3*N]})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2*N+i]*p[3*N]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]*p[3*N]})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[N+i] - p[2*N+i]*p[3*N]})
            # s_i >= 0.5
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i] - 0.5})
            # s_i <= 2.0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 2.0 - p[2*N+i]})
        return cons

    constraints = make_constraints()

    # ------------------------------------------------------------------
    # Bounds: x,y in [0,1], s in [0.5, 2.0], R in [0.01, 0.5]
    # ------------------------------------------------------------------
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.5, 2.0)] * N + [(0.01, 0.5)]

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------
    def is_valid(params, tol=1e-9):
        x = params[:N]
        y = params[N:2*N]
        s = params[2*N:3*N]
        R = params[3*N]
        r = s * R
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(1.0 - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(1.0 - y - r < -tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx*dx + dy*dy
        min_dist2 = (r[:, None] + r[None, :])**2
        iu = np.triu_indices(N, k=1)
        if np.any(dist2[iu] < min_dist2[iu] - tol):
            return False
        return True

    def get_score(params):
        _, _, r, _, _ = unpack(params)
        return float(np.sum(r))

    # ------------------------------------------------------------------
    # Initial configurations
    # ------------------------------------------------------------------
    def get_initial_config(restart, rng):
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        s = np.ones(N)
        R = 0.08

        if restart % 10 == 0:
            # Regular grid
            for i in range(N):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
            R = 0.5 / side - 0.002
        elif restart % 10 == 1:
            # Perturbed grid
            for i in range(N):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
            x += rng.uniform(-0.02, 0.02, size=N)
            y += rng.uniform(-0.02, 0.02, size=N)
            R = 0.5 / side - 0.002
        elif restart % 10 == 2:
            # Hexagonal-like
            for i in range(N):
                row = i // 6
                col = i % 6
                x[i] = (col + 0.5) / 6 + (0.25 if row % 2 == 1 else 0.0)
                y[i] = (row + 0.5) / 6
            R = 0.08
        elif restart % 10 == 3:
            # Random
            x = rng.uniform(0.1, 0.9, size=N)
            y = rng.uniform(0.1, 0.9, size=N)
            R = 0.07
        elif restart % 10 == 4:
            # 8x4 grid
            for i in range(N):
                row = i // 8
                col = i % 8
                x[i] = (col + 0.5) / 8
                y[i] = (row + 0.5) / 4
            R = 0.06
        elif restart % 10 == 5:
            # 4x8 grid
            for i in range(N):
                row = i // 4
                col = i % 4
                x[i] = (col + 0.5) / 4
                y[i] = (row + 0.5) / 8
            R = 0.12
        elif restart % 10 == 6:
            # Hexagonal variant
            for i in range(N):
                row = i // 6
                col = i % 6
                x[i] = (col + 0.5) / 6 + (0.2 if row % 2 == 0 else 0.0)
                y[i] = (row + 0.5) / 6
            R = 0.075
        elif restart % 10 == 7:
            # Clustered
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(N):
                group = i // 8
                cx, cy = centers[group]
                sub = i % 8
                sx = (sub % 3) / 3.0 + 0.5/3.0
                sy = (sub // 3) / 3.0 + 0.5/3.0
                x[i] = cx + (sx - 0.5) * 0.4
                y[i] = cy + (sy - 0.5) * 0.4
            R = 0.06
        elif restart % 10 == 8:
            # Random medium
            x = rng.uniform(0.1, 0.9, size=N)
            y = rng.uniform(0.1, 0.9, size=N)
            R = 0.08
        else:
            # Random small
            x = rng.uniform(0.05, 0.95, size=N)
            y = rng.uniform(0.05, 0.95, size=N)
            R = 0.05

        return np.concatenate([x, y, s, [R]])

    # ------------------------------------------------------------------
    # Push-apart for center-only (preserves radii)
    # ------------------------------------------------------------------
    def push_apart(params, n_iters=50):
        x = params[:N].copy()
        y = params[N:2*N].copy()
        s = params[2*N:3*N].copy()
        R = params[3*N]
        r = s * R

        for _ in range(n_iters):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        if d > 1e-12:
                            overlap = min_d - d
                            px = (dx/d) * overlap
                            py = (dy/d) * overlap
                            x[i] -= px * 0.5
                            y[i] -= py * 0.5
                            x[j] += px * 0.5
                            y[j] += py * 0.5
                            moved = True
                        else:
                            angle = np.random.uniform(0, 2*np.pi)
                            x[i] -= 0.01 * np.cos(angle)
                            y[i] -= 0.01 * np.sin(angle)
                            x[j] += 0.01 * np.cos(angle)
                            y[j] += 0.01 * np.sin(angle)
                            moved = True
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            if not moved:
                break

        return np.concatenate([x, y, s, [R]])

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    rng = np.random.RandomState(42)

    n_restarts = 40
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=40)

        if is_valid(params):
            s_val = get_score(params)
            if s_val > best_score:
                best_score = s_val
                best_result = params.copy()

        # SLSQP optimization
        try:
            res = scipy.optimize.minimize(
                objective, params, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.success or res.fun < objective(params):
                x_opt = res.x
                if is_valid(x_opt):
                    s_val = get_score(x_opt)
                    if s_val > best_score:
                        best_score = s_val
                        best_result = x_opt.copy()
                else:
                    # Try push-apart to recover
                    x_fixed = push_apart(x_opt, n_iters=30)
                    if is_valid(x_fixed):
                        s_val = get_score(x_fixed)
                        if s_val > best_score:
                            best_score = s_val
                            best_result = x_fixed.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final polish: local perturbation + SLSQP on best
    # ------------------------------------------------------------------
    if best_result is not None and time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            pert_scale = 0.003 / (trial + 1)
            x_pert = best_result + np.random.normal(0, pert_scale, size=best_result.shape)
            x_pert[:N] = np.clip(x_pert[:N], 0, 1)
            x_pert[N:2*N] = np.clip(x_pert[N:2*N], 0, 1)
            x_pert[2*N:3*N] = np.clip(x_pert[2*N:3*N], 0.5, 2.0)
            x_pert[3*N] = np.clip(x_pert[3*N], 0.01, 0.5)
            try:
                res = scipy.optimize.minimize(
                    objective, x_pert, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-13}
                )
                if is_valid(res.x):
                    s_val = get_score(res.x)
                    if s_val > best_score:
                        best_score = s_val
                        best_result = res.x.copy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Safety fallback
    # ------------------------------------------------------------------
    if best_result is None:
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        s = np.ones(N)
        R = 0.5/side - 0.002
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
        best_result = np.concatenate([x, y, s, [R]])

    # ------------------------------------------------------------------
    # Format output
    # ------------------------------------------------------------------
    x = best_result[:N]
    y = best_result[N:2*N]
    s = best_result[2*N:3*N]
    R = best_result[3*N]
    r = s * R

    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r

    return result

# EVOLVE-BLOCK-END
