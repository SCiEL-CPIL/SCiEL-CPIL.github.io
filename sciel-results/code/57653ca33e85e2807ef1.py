"""Optimized circle packing in perimeter-4 rectangle for 21 circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _objective_and_grad(x):
    """
    x = [r0..r20, cx0..cx20, cy0..cy20, W, H]
    Maximize sum(r) = -minimize -sum(r)
    Constraints (as penalties):
      - perimeter: W + H <= 2
      - each circle inside: cx_i - r_i >= 0, cy_i - r_i >= 0, cx_i + r_i <= W, cy_i + r_i <= H
      - pairwise: dist(i,j) >= r_i + r_j
    """
    r = x[:N]
    cx = x[N:2*N]
    cy = x[2*N:3*N]
    W = x[3*N]
    H = x[3*N+1]

    # Soft penalties (smooth, differentiable)
    # Perimeter penalty
    per_viol = max(0.0, W + H - 2.0)

    # Boundary violations
    bnd_viol = 0.0
    for i in range(N):
        bnd_viol += max(0.0, r[i] - cx[i])           # left
        bnd_viol += max(0.0, r[i] - cy[i])           # bottom
        bnd_viol += max(0.0, cx[i] + r[i] - W)       # right
        bnd_viol += max(0.0, cy[i] + r[i] - H)       # top

    # Pairwise overlap violations
    pair_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            pair_viol += max(0.0, r[i] + r[j] - d)

    penalty = 100.0 * (per_viol + bnd_viol + pair_viol)
    obj = -np.sum(r) + penalty
    return obj

def _objective_and_grad_vec(x):
    """Vectorized objective for speed."""
    r = x[:N]
    cx = x[N:2*N]
    cy = x[2*N:3*N]
    W = x[3*N]
    H = x[3*N+1]

    per_viol = max(0.0, W + H - 2.0)

    # Boundary
    bnd_viol = (np.sum(np.maximum(0.0, r - cx)) +
                np.sum(np.maximum(0.0, r - cy)) +
                np.sum(np.maximum(0.0, cx + r - W)) +
                np.sum(np.maximum(0.0, cy + r - H)))

    # Pairwise
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx*dx + dy*dy + 1e-12)
    dist_mat = r[:, None] + r[None, :] - d
    pair_viol = 0.5 * np.sum(np.maximum(0.0, dist_mat))

    penalty = 100.0 * (per_viol + bnd_viol + pair_viol)
    return -np.sum(r) + penalty

def _grad_vec(x):
    """Numerical gradient (central difference) - robust fallback."""
    n = len(x)
    g = np.zeros(n)
    h = 1e-7
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_objective_and_grad_vec(xp) - _objective_and_grad_vec(xm)) / (2*h)
    return g

def _grad_analytic(x):
    """Analytical gradient of the smooth penalty objective."""
    r = x[:N]
    cx = x[N:2*N]
    cy = x[2*N:3*N]
    W = x[3*N]
    H = x[3*N+1]

    g = np.zeros(3*N + 2)

    # Objective: -sum(r) + 100*(per + bnd + pair)
    # d(-sum(r))/dr = -1
    g[:N] = -1.0

    # Perimeter: max(0, W+H-2)
    per_viol = max(0.0, W + H - 2.0)
    if per_viol > 0:
        g[3*N] += 100.0   # dW
        g[3*N+1] += 100.0 # dH

    # Boundary terms
    # left: max(0, r_i - cx_i)
    for i in range(N):
        # r_i - cx_i > 0
        v = r[i] - cx[i]
        if v > 0:
            g[i] += 100.0      # dr_i
            g[N+i] -= 100.0    # dcx_i
        # r_i - cy_i > 0
        v = r[i] - cy[i]
        if v > 0:
            g[i] += 100.0
            g[2*N+i] -= 100.0
        # cx_i + r_i - W > 0
        v = cx[i] + r[i] - W
        if v > 0:
            g[i] += 100.0
            g[N+i] += 100.0
            g[3*N] -= 100.0
        # cy_i + r_i - H > 0
        v = cy[i] + r[i] - H
        if v > 0:
            g[i] += 100.0
            g[2*N+i] += 100.0
            g[3*N+1] -= 100.0

    # Pairwise: 0.5 * sum_{i<j} max(0, r_i+r_j-d_ij)
    # For each pair (i,j) with violation:
    #   d/d r_i = +100, d/d r_j = +100
    #   d/d cx_i = 100 * (cx_i-cx_j)/d, d/d cx_j = 100*(cx_j-cx_i)/d
    #   similarly for cy
    for i in range(N):
        for j in range(i+1, N):
            dxi = cx[i] - cx[j]
            dyi = cy[i] - cy[j]
            d = np.sqrt(dxi*dxi + dyi*dyi + 1e-12)
            viol = r[i] + r[j] - d
            if viol > 0:
                g[i] += 100.0
                g[j] += 100.0
                g[N+i] += 100.0 * dxi / d
                g[N+j] += 100.0 * (-dxi) / d
                g[2*N+i] += 100.0 * dyi / d
                g[2*N+j] += 100.0 * (-dyi) / d

    return g

def _is_valid(x):
    """Check if configuration is valid (perimeter <= 2, no overlaps, inside)."""
    r = x[:N]
    cx = x[N:2*N]
    cy = x[2*N:3*N]
    W = x[3*N]
    H = x[3*N+1]

    if W + H > 2.0 + 1e-9:
        return False
    if W < 1e-6 or H < 1e-6:
        return False
    if np.any(r < -1e-9):
        return False
    r = np.maximum(r, 0.0)

    # Inside
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False

    # Pairwise
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            if d < r[i] + r[j] - 1e-9:
                return False
    return True

def _repair(x):
    """Clip radii to non-negative and ensure circles fit in rectangle."""
    r = np.maximum(x[:N], 0.0)
    cx = x[N:2*N].copy()
    cy = x[2*N:3*N].copy()
    W = max(x[3*N], 1e-4)
    H = max(x[3*N+1], 1e-4)

    # Scale W, H to satisfy perimeter
    per = W + H
    if per > 2.0:
        scale = 2.0 / per
        W *= scale
        H *= scale
        cx *= scale
        cy *= scale
        r *= scale

    # Clip radii to fit in box
    for i in range(N):
        max_r = min(cx[i], cy[i], W - cx[i], H - cy[i])
        r[i] = min(r[i], max(0.0, max_r))

    return np.concatenate([r, cx, cy, [W, H]])

def _seed_grid(n_cols, n_rows, W, H):
    """Generate a grid seed configuration."""
    total = n_cols * n_rows
    if total < N:
        n_rows += 1
        total = n_cols * n_rows

    xs = (np.arange(n_cols) + 0.5) * W / n_cols
    ys = (np.arange(n_rows) + 0.5) * H / n_rows
    cx = np.tile(xs, n_rows)
    cy = np.repeat(ys, n_cols)
    r = np.full(N, min(W/n_cols, H/n_rows) * 0.45)
    return np.concatenate([r, cx[:N], cy[:N], [W, H]])

def _optimize_from(x0, deadline, max_iter=300):
    """Run L-BFGS-B from x0, return best valid result or None."""
    x = x0.copy()
    best_val = -np.inf
    best_x = None

    def fun(xv):
        return _objective_and_grad_vec(xv)

    def jac(xv):
        return _grad_analytic(xv)

    res = None
    try:
        res = minimize(fun, x, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxls': 50, 'maxcor': 10})
    except Exception:
        pass

    if res is not None and res.x is not None:
        x = res.x
        val = _objective_and_grad_vec(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    # Try a few local refinements
    for _ in range(3):
        if time.time() > deadline:
            break
        try:
            res2 = minimize(fun, x, method='L-BFGS-B', jac=jac,
                            options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13})
            if res2.x is not None:
                x = res2.x
                val = _objective_and_grad_vec(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
        except Exception:
            break

    if best_x is None:
        return None

    # Repair and check validity
    repaired = _repair(best_x)
    if _is_valid(repaired):
        return repaired
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    best_x = None
    best_score = -np.inf

    # Multiple aspect ratios for the rectangle
    aspect_ratios = [1.0, 0.8, 1.25, 0.66, 1.5, 0.5, 2.0, 0.75, 1.33, 0.6, 1.66]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break

        # W + H = 2, W/H = ar => W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)

        # Try different grid layouts
        layouts = [
            (5, 5),   # 25 cells, use 21
            (6, 4),   # 24 cells
            (7, 3),   # 21 cells
            (4, 6),
            (3, 7),
        ]

        for (nc, nr) in layouts:
            if time.time() > deadline:
                break

            x0 = _seed_grid(nc, nr, W, H)
            result = _optimize_from(x0, deadline, max_iter=200)
            if result is not None:
                score = np.sum(result[:N])
                if score > best_score:
                    best_score = score
                    best_x = result.copy()

    # Final polish on best
    if best_x is not None and time.time() < deadline:
        # High-precision polish
        def fun(xv):
            return _objective_and_grad_vec(xv)
        def jac(xv):
            return _grad_analytic(xv)
        try:
            res = minimize(fun, best_x, method='L-BFGS-B', jac=jac,
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.x is not None:
                repaired = _repair(res.x)
                if _is_valid(repaired):
                    score = np.sum(repaired[:N])
                    if score > best_score:
                        best_x = repaired
                        best_score = score
        except Exception:
            pass

    # Format output: (cx, cy, r) for each circle
    if best_x is None:
        # Fallback: simple grid
        side = 5
        W = 1.0
        H = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i+0.5)/side * W, (j+0.5)/side * H, 0.5/side * 0.9))
        return np.array(cells[:N])

    cx = best_x[N:2*N]
    cy = best_x[2*N:3*N]
    r = best_x[:N]
    return np.column_stack([cx, cy, r])

# EVOLVE-BLOCK-END
