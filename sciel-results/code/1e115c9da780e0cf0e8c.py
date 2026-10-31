"""Circle packing in perimeter-4 rectangle: Structural NLP with SLSQP + repair loop."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit

    best_value = 0.0
    best_xy = None
    best_r = None
    best_w = None
    best_h = None

    # ---- Feasibility check ----
    def is_valid(x, y, r, w, h, tol=1e-9):
        if w <= 1e-6 or h <= 1e-6:
            return False
        if np.any(r <= 0):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > w + tol) or np.any(y + r > h + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    # ---- Force relaxation to get feasible start ----
    def relax_to_feasible(x, y, r, w, h, max_iter=2000):
        x = x.copy()
        y = y.copy()
        r = r.copy()
        h = 2.0 - w
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            dt = 0.01 * (0.999 ** it)
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 100
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 100
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 100
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 100
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    ds = r[i] + r[j]
                    if d2 < ds * ds:
                        d = np.sqrt(max(d2, 1e-12))
                        overlap = ds - d
                        if overlap > 0:
                            fx[i] += (dx / d) * overlap * 50
                            fy[i] += (dy / d) * overlap * 50
                            fx[j] -= (dx / d) * overlap * 50
                            fy[j] -= (dy / d) * overlap * 50
            x += dt * fx
            y += dt * fy
            # clip to boundary
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            if is_valid(x, y, r, w, h, tol=1e-7):
                break
        return x, y, r

    # ---- NLP: maximize sum(r) with constraints ----
    def nlp_optimize(x0, y0, r0, w0, max_iter=500, ftol=1e-14):
        h0 = 2.0 - w0

        def pack(x, y, r, w):
            h = 2.0 - w
            z = np.empty(3 * N + 1)
            z[0::3][:N] = x
            z[1::3][:N] = y
            z[2::3][:N] = r
            z[-1] = w
            return z

        def unpack(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            return x, y, r, w

        def objective(z):
            r = z[2::3][:N]
            return -np.sum(r)

        constraints = []
        # boundary constraints
        for i in range(N):
            def _mk(axis, side, idx=i):
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
            constraints.append({'type': 'ineq', 'fun': _mk(0, 0)})
            constraints.append({'type': 'ineq', 'fun': _mk(0, 1)})
            constraints.append({'type': 'ineq', 'fun': _mk(1, 0)})
            constraints.append({'type': 'ineq', 'fun': _mk(1, 1)})

        # non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                def _ov(idx1, idx2):
                    return lambda p: (
                        (p[3 * idx1] - p[3 * idx2]) ** 2 +
                        (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                        (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                    )
                constraints.append({'type': 'ineq', 'fun': _ov(i, j)})

        bounds = []
        for i in range(N):
            bounds.extend([(0, 2), (0, 2), (1e-6, 1.0)])
        bounds.append((0.3, 1.7))

        z0 = pack(x0, y0, r0, w0)

        try:
            res = minimize(objective, z0, method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 100})
        except Exception:
            return None

        if not res.success and res.fun >= 0:
            return None

        z_opt = res.x
        x, y, r, w = unpack(z_opt)
        h = 2.0 - w
        if is_valid(x, y, r, w, h, tol=1e-9):
            val = float(np.sum(r))
            return x, y, r, w, val
        else:
            return None

    # ---- Initial configurations ----
    def get_initials():
        inits = []
        # Grid-based: 5x5 with 21 of 25 cells
        side = 5
        for w0 in [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = 0.45 * min(w0 / side, h0 / side)
            inits.append((x.copy(), y.copy(), r.copy(), w0))

        # Random with seed
        rng = np.random.default_rng(123)
        for w0 in [0.9, 1.1, 0.8, 1.2]:
            h0 = 2.0 - w0
            x = rng.uniform(0.1, w0 - 0.1, N)
            y = rng.uniform(0.1, h0 - 0.1, N)
            r = rng.uniform(0.04, 0.08, N)
            inits.append((x, y, r, w0))

        return inits

    inits = get_initials()

    for idx, (x0, y0, r0, w0) in enumerate(inits):
        if time.time() > deadline - 30:
            break

        # Relax to feasible
        xr, yr, rr = relax_to_feasible(x0, y0, r0, w0, max_iter=500)
        hr = 2.0 - w0

        if not is_valid(xr, yr, rr, w0, hr, tol=1e-6):
            continue

        # NLP optimize
        res = nlp_optimize(xr, yr, rr, w0, max_iter=400, ftol=1e-14)
        if res is None:
            continue

        x_opt, y_opt, r_opt, w_opt, val = res
        if val > best_value:
            best_value = val
            best_xy = (x_opt, y_opt)
            best_r = r_opt
            best_w = w_opt
            best_h = 2.0 - w_opt

        # Try second NLP pass with tighter tolerance
        if time.time() < deadline - 35:
            res2 = nlp_optimize(x_opt, y_opt, r_opt, w_opt, max_iter=200, ftol=1e-16)
            if res2 is not None:
                x2, y2, r2, w2, val2 = res2
                if val2 > best_value:
                    best_value = val2
                    best_xy = (x2, y2)
                    best_r = r2
                    best_w = w2
                    best_h = 2.0 - w2

    # Fallback: grid solution
    if best_xy is None:
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.45 / side
        w = 1.0
        h = 1.0
        return np.column_stack((x, y, r))

    x, y = best_xy
    r = best_r
    w = best_w
    h = best_h

    # Final shrink to ensure strict validity
    for factor in [1.0, 0.999, 0.998, 0.995, 0.99]:
        r_try = r * factor
        if is_valid(x, y, r_try, w, h, tol=1e-10):
            r = r_try
            break

    return np.column_stack((x, y, r))


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
