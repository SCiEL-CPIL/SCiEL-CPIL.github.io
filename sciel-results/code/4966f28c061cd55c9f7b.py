"""Circle packing in perimeter-4 rectangle: maximize sum of 21 radii via constrained optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

def _constraints(vars, w, h):
    """Return list of inequality constraints (>= 0)."""
    x = vars[0:N]
    y = vars[N:2*N]
    r = vars[2*N:3*N]
    cons = []
    # boundary: x_i - r_i >= 0
    cons.append((x - r).min())
    cons.append((w - x - r).min())
    cons.append((y - r).min())
    cons.append((h - y - r).min())
    # non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rr = (r[i] + r[j])**2
            cons.append(d2 - rr)
    return cons

def _neg_sum_r(vars, w, h):
    return -np.sum(vars[2*N:3*N])

def _try_aspect(w, h, deadline, rng):
    """Try to pack circles in rectangle w x h. Return (sum_r, vars) or None."""
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) >= N:
                break
            xs.append((i + 0.5) * w / side)
            ys.append((j + 0.5) * h / side)
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    # Initial radius: small, will grow
    r0 = 0.5 * min(w, h) / side * 0.9
    r0 = np.full(N, r0)

    x0 = np.concatenate([xs, ys, r0])
    lb = np.concatenate([np.zeros(N), np.zeros(N), np.full(N, 1e-4)])
    ub = np.concatenate([np.full(N, w), np.full(N, h), np.full(N, min(w, h))])

    best_sum = 0.0
    best_vars = None

    # Multi-start with small perturbations
    n_starts = 3
    for s in range(n_starts):
        if time.time() > deadline:
            break
        if s == 0:
            x0s = x0.copy()
        else:
            x0s = x0 + 0.05 * (w, h, r0[0]) * rng.standard_normal(3*N)
            x0s = np.clip(x0s, lb + 1e-6, ub - 1e-6)

        res = minimize(
            _neg_sum_r, x0s, args=(w, h),
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': 500, 'maxls': 20, 'ftol': 1e-10}
        )
        if res.success or res.fun < best_sum * -1 if best_sum > 0 else True:
            # Check feasibility
            cons_vals = _constraints(res.x, w, h)
            min_con = min(cons_vals)
            if min_con > -1e-5:
                # Clamp tiny violations
                s_r = np.sum(res.x[2*N:3*N])
                if s_r > best_sum:
                    best_sum = s_r
                    best_vars = res.x.copy()

    if best_vars is not None:
        # Final clamp to ensure feasibility
        x = best_vars[0:N].copy()
        y = best_vars[N:2*N].copy()
        r = best_vars[2*N:3*N].copy()
        # Clamp to boundaries
        r = np.clip(r, 1e-4, None)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        # Re-check overlaps and shrink if needed
        for _ in range(50):
            ok = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i]-x[j]; dy = y[i]-y[j]
                    d = np.sqrt(dx*dx+dy*dy)
                    rr = r[i]+r[j]
                    if d < rr - 1e-9:
                        # shrink the smaller radius a bit
                        k = np.min([i, j])
                        r[k] *= 0.98
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        best_vars = np.concatenate([x, y, r])
        best_sum = np.sum(r)
        return best_sum, best_vars
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    rng = np.random.default_rng(42)

    best_sum = 0.0
    best_result = None

    # Try various aspect ratios w x h with 2(w+h)=4 => w+h=2
    # w ranges from 0.5 to 1.5 (h=2-w)
    aspects = [1.0, 0.8, 1.2, 0.6, 1.4, 0.7, 1.3, 0.9, 1.1, 0.5, 1.5]
    for w in aspects:
        if time.time() > deadline:
            break
        h = 2.0 - w
        if h <= 0:
            continue
        result = _try_aspect(w, h, deadline, rng)
        if result is not None:
            s, vars_ = result
            if s > best_sum:
                best_sum = s
                best_result = vars_.copy()

    if best_result is None:
        # Fallback: seed-like grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_sum = np.sum(best_result[:, 2])

    # Format output: (x, y, r)
    x = best_result[0:N]
    y = best_result[N:2*N]
    r = best_result[2*N:3*N]
    out = np.column_stack([x, y, r])
    return out
