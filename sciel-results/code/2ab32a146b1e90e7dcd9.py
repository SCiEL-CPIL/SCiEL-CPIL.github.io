"""Circle packing in perimeter-4 rectangle.
Maximize sum of 21 disjoint circle radii inside a rectangle with perimeter <= 4.

Combined strategy:
1. Parametric layouts (grid, hexagonal, mixed-row) -> tight bounding rectangle.
2. Rectangle-shape optimization: for a given configuration, optimize aspect ratio.
3. Vectorized local optimization (Nelder-Mead on [r, x, y]) to improve a layout.
4. Randomized restarts of the optimizer to escape local optima.
5. Explicit deadline guard; return best found before the wall-clock limit.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin before the 150s kill


def _hexagonal_layout(n_cols, n_rows):
    xs = []
    ys = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 2.0 + (1.0 if row % 2 == 1 else 0.0)
            y = row * np.sqrt(3.0)
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _grid_layout(n_cols, n_rows):
    xs = np.array([col * 2.0 + 1.0 for row in range(n_rows) for col in range(n_cols)])
    ys = np.array([row * 2.0 + 1.0 for row in range(n_rows) for col in range(n_cols)])
    W = (n_cols - 1) * 2.0 + 2.0
    H = (n_rows - 1) * 2.0 + 2.0
    return xs, ys, W, H


def _mixed_row_layout(row_counts):
    xs = []
    ys = []
    for row_idx, n_circ in enumerate(row_counts):
        y = row_idx * np.sqrt(3.0)
        offset = 1.0 if row_idx % 2 == 1 else 0.0
        for col in range(n_circ):
            xs.append(col * 2.0 + offset)
            ys.append(y)
    if len(xs) == 0:
        return None
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - xs.min() + 1.0
    ys = ys - ys.min() + 1.0
    W = xs.max() - xs.min() + 2.0
    H = ys.max() - ys.min() + 2.0
    return xs, ys, W, H


def _tight_rect_for_config(x, y, r):
    """Given circle centers and radius, compute tight bounding rectangle.
    w = (xmax - xmin) + 2r, h = (ymax - ymin) + 2r.
    Scale so w + h = 2."""
    W = (x.max() - x.min()) + 2.0 * r
    H = (y.max() - y.min()) + 2.0 * r
    if W + H <= 0:
        return None
    w = 2.0 * W / (W + H)
    h = 2.0 * H / (W + H)
    return w, h


def _is_valid(x, y, r, w, h, tol=1e-7):
    n = len(x)
    if r <= 0 or w + h > 2.0 + tol:
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    if d.min() < 2.0 * r - tol:
        return False
    return True


def _smooth_penalty(params, w, h, beta=50.0):
    """
    Smooth penalty using log-sum-exp for a differentiable objective.
    params = [r, x1, y1, x2, y2, ..., xN, yN]
    """
    r = params[0]
    if r <= 1e-10:
        return 1e6
    x = params[1::2]
    y = params[2::2]
    n = len(x)
    
    # Boundary violations (smooth)
    # For each circle: min(r-x, x-(w-r)) and min(r-y, y-(h-r))
    # Use soft-min via log-sum-exp
    bx_left = r - x
    bx_right = x - (w - r)
    by_bottom = r - y
    by_top = y - (h - r)
    
    # Soft penalty: sum of max(violation, 0)^2 approximated smoothly
    # Use: penalty = sum(softplus(violation)^2) where softplus(z) = log(1+exp(z/beta))*beta
    def softplus(z, b):
        return b * np.log1p(np.exp(z / b))
    
    b = 0.05
    pen_bx = softplus(bx_left, b) + softplus(bx_right, b)
    pen_by = softplus(by_bottom, b) + softplus(by_top, b)
    pen_boundary = np.sum(pen_bx**2) + np.sum(pen_by**2)
    
    # Pair violations
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    d = np.sqrt(d2 + 1e-12)
    np.fill_diagonal(d, np.inf)
    gap = 2.0 * r - d
    # Smooth max(gap, 0)^2
    pen_pair = np.sum(softplus(gap, b)**2)
    
    return -r + beta * (pen_boundary + pen_pair)


def _objective(params, w, h):
    r = params[0]
    if r <= 0:
        return 1e6
    x = params[1::2]
    y = params[2::2]
    n = len(x)
    # boundary violations
    bx = np.minimum(np.minimum(r - x, x - (w - r)), 0.0)
    by = np.minimum(np.minimum(r - y, y - (h - r)), 0.0)
    pen = np.sum(bx * bx) + np.sum(by * by)
    # pair violations
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    gap = 2.0 * r - d
    viol = np.maximum(gap, 0.0)
    pen += np.sum(viol * viol)
    return -r + 50.0 * pen


def _optimize_positions(x, y, r, w, h, maxiter=300, method='Nelder-Mead'):
    n = len(x)
    params0 = np.concatenate([[r], x, y])
    try:
        if method == 'SLSQP':
            # Define bounds
            bounds = [(r*0.5, min(w, h)/2 + 0.01)]
            for i in range(n):
                bounds.append((0.01, w - 0.01))
                bounds.append((0.01, h - 0.01))
            
            res = minimize(_smooth_penalty, params0, args=(w, h), method='SLSQP',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-8})
        else:
            res = minimize(_objective, params0, args=(w, h), method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-5})
        r2 = res.x[0]
        x2 = res.x[1::2]
        y2 = res.x[2::2]
        if r2 > r + 1e-9 and _is_valid(x2, y2, r2, w, h):
            return x2, y2, r2
    except Exception:
        pass
    return x, y, r


def _optimize_aspect(x, y, r):
    """For a given circle configuration, find the best (w,h) with w+h=2
    such that all circles fit. Returns improved (x, y, r, w, h) or None."""
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    W0 = xmax - xmin  # width of center span
    H0 = ymax - ymin  # height of center span

    # Compute min pair distance
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()

    def max_r_for_w(w):
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 0.0
        r_boundary = min(w / 2.0, h / 2.0)
        r_span_w = (w - W0) / 2.0
        r_span_h = (h - H0) / 2.0
        r_pair = min_d / 2.0
        return max(0.0, min(r_boundary, r_span_w, r_span_h, r_pair))

    # Optimize w to maximize max_r_for_w(w)
    w_min = max(W0 + 1e-6, 0.01)
    w_max = min(2.0 - H0 - 1e-6, 1.99)
    if w_min >= w_max:
        return None

    best_r = 0.0
    best_w = None
    # Coarse grid
    for w_try in np.linspace(w_min, w_max, 100):
        r_try = max_r_for_w(w_try)
        if r_try > best_r:
            best_r = r_try
            best_w = w_try

    if best_w is None:
        return None

    # Fine grid around best
    delta = (w_max - w_min) / 500.0
    for w_try in np.linspace(max(w_min, best_w - 5*delta), min(w_max, best_w + 5*delta), 100):
        r_try = max_r_for_w(w_try)
        if r_try > best_r:
            best_r = r_try
            best_w = w_try

    if best_r <= r + 1e-10:
        return None

    h = 2.0 - best_w
    shift_x = (best_w - W0) / 2.0 - xmin
    shift_y = (h - H0) / 2.0 - ymin

    x_new = x + shift_x
    y_new = y + shift_y

    if _is_valid(x_new, y_new, best_r, best_w, h):
        return x_new, y_new, best_r, best_w, h
    return None


def construct():
    start = time.time()
    best_sum = 0.0
    best_result = None

    def consider(x, y, r, w, h):
        nonlocal best_sum, best_result
        if _is_valid(x, y, r, w, h):
            s = N * r
            if s > best_sum + 1e-12:
                best_sum = s
                best_result = np.column_stack([x, y, np.full(N, r)])

    # ---- Phase 1: analytic layouts ----
    candidates = []
    for nc in range(1, 12):
        for nr in range(1, 12):
            if nc * nr >= N:
                candidates.append(('grid', nc, nr))
                candidates.append(('hex', nc, nr))
    base_counts = [
        [4, 4, 4, 4, 3, 2], [5, 4, 4, 4, 3, 1], [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2], [5, 5, 4, 3, 2, 2], [6, 4, 4, 3, 2, 2],
        [4, 5, 4, 4, 3, 1], [3, 4, 4, 5, 3, 2], [5, 4, 5, 3, 2, 2],
        [4, 4, 5, 4, 3, 1], [3, 3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4], [6, 4, 4, 4, 3], [7, 3, 3, 3, 3, 2],
        [3, 4, 4, 4, 4, 2], [4, 4, 4, 3, 4, 2], [5, 4, 4, 3, 4, 1],
        [4, 3, 4, 3, 4, 3], [3, 4, 3, 4, 3, 4], [4, 4, 3, 4, 3, 3],
    ]
    for counts in base_counts:
        if sum(counts) >= N:
            candidates.append(('mixed', counts))

    for cand in candidates:
        if time.time() > DEADLINE - 30.0:
            break
        try:
            if cand[0] == 'grid':
                xs, ys, W, H = _grid_layout(cand[1], cand[2])
            elif cand[0] == 'hex':
                xs, ys, W, H = _hexagonal_layout(cand[1], cand[2])
            else:
                res = _mixed_row_layout(cand[1])
                if res is None:
                    continue
                xs, ys, W, H = res

            # Use first N circles
            xs_sub = xs[:N]
            ys_sub = ys[:N]

            W_sub = xs_sub.max() - xs_sub.min()
            H_sub = ys_sub.max() - ys_sub.min()

            dx = xs_sub[:, None] - xs_sub[None, :]
            dy = ys_sub[:, None] - ys_sub[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            min_d = d.min()

            r_rect = (2.0 - W_sub - H_sub) / 2.0
            r_pair = min_d / 2.0
            r = min(r_rect, r_pair)

            if r <= 0:
                continue

            w = W_sub + 2.0 * r
            h = H_sub + 2.0 * r

            x = (xs_sub - xs_sub.min()) + (w - W_sub) / 2.0
            y = (ys_sub - ys_sub.min()) + (h - H_sub) / 2.0

            consider(x, y, r, w, h)

            if time.time() < DEADLINE - 25.0:
                result = _optimize_aspect(x, y, r)
                if result is not None:
                    x2, y2, r2, w2, h2 = result
                    consider(x2, y2, r2, w2, h2)
        except Exception:
            continue

    # ---- Phase 2: local optimization + randomized restarts ----
    if best_result is not None and time.time() < DEADLINE - 15.0:
        bx = best_result[:, 0].copy()
        by = best_result[:, 1].copy()
        br = best_result[0, 2]
        Wb = bx.max() - bx.min()
        Hb = by.max() - by.min()
        bw = Wb + 2.0 * br
        bh = Hb + 2.0 * br
        if bw + bh > 0:
            bw = 2.0 * bw / (bw + bh)
            bh = 2.0 * bh / (bw + bh)

        # First, try aspect optimization on the best layout
        if time.time() < DEADLINE - 20.0:
            result = _optimize_aspect(bx, by, br)
            if result is not None:
                x2, y2, r2, w2, h2 = result
                consider(x2, y2, r2, w2, h2)
                if r2 > br + 1e-9:
                    bx, by, br = x2, y2, r2
                    bw, bh = w2, h2

        # Local optimization of the best layout using smooth penalty
        if time.time() < DEADLINE - 15.0:
            # Try SLSQP with smooth penalty
            x_s, y_s, r_s = _optimize_positions(bx.copy(), by.copy(), br, bw, bh, 
                                                 maxiter=200, method='SLSQP')
            consider(x_s, y_s, r_s, bw, bh)
            
            # Also try Nelder-Mead
            x_nm, y_nm, r_nm = _optimize_positions(bx.copy(), by.copy(), br, bw, bh, 
                                                    maxiter=400, method='Nelder-Mead')
            consider(x_nm, y_nm, r_nm, bw, bh)
            
            # Use whichever is better for further optimization
            if r_s > r_nm:
                bx, by, br = x_s, y_s, r_s
            else:
                bx, by, br = x_nm, y_nm, r_nm
            
            # Try aspect optimization on the optimized positions
            if time.time() < DEADLINE - 12.0:
                result = _optimize_aspect(bx, by, br)
                if result is not None:
                    x2, y2, r2, w2, h2 = result
                    consider(x2, y2, r2, w2, h2)
                    if r2 > br + 1e-9:
                        bx, by, br = x2, y2, r2
                        bw, bh = w2, h2

        # Randomized restarts: diverse initializations
        rng = np.random.default_rng(42)
        n_trials = 0
        while time.time() < DEADLINE - 8.0 and n_trials < 500:
            n_trials += 1
            
            # Multiple initialization strategies
            strategy = n_trials % 4
            
            if strategy == 0:
                # Small perturbation of best
                x0 = bx.copy()
                y0 = by.copy()
                amp = 0.1 * br
                x0 = x0 + rng.uniform(-amp, amp, N)
                y0 = y0 + rng.uniform(-amp, amp, N)
            elif strategy == 1:
                # Medium perturbation
                x0 = bx.copy()
                y0 = by.copy()
                amp = 0.25 * br
                x0 = x0 + rng.uniform(-amp, amp, N)
                y0 = y0 + rng.uniform(-amp, amp, N)
            elif strategy == 2:
                # Random positions within box
                x0 = rng.uniform(br, bw - br, N)
                y0 = rng.uniform(br, bh - br, N)
            else:
                # Grid-like random with jitter
                side = int(np.ceil(np.sqrt(N)))
                x0 = np.array([(i + 0.5) / side * bw + rng.uniform(-0.05, 0.05) * bw 
                              for i in range(N)])
                y0 = np.array([(j + 0.5) / side * bh + rng.uniform(-0.05, 0.05) * bh 
                              for j in range(N)])
            
            x0 = np.clip(x0, br, bw - br)
            y0 = np.clip(y0, br, bh - br)
            
            # Use smooth penalty optimizer
            x2, y2, r2 = _optimize_positions(x0, y0, br, bw, bh, maxiter=150, method='SLSQP')
            consider(x2, y2, r2, bw, bh)
            
            # Also try Nelder-Mead if time permits
            if time.time() < DEADLINE - 8.0:
                x3, y3, r3 = _optimize_positions(x2, y2, r2, bw, bh, maxiter=100, method='Nelder-Mead')
                consider(x3, y3, r3, bw, bh)
            
            # Try aspect optimization
            if time.time() < DEADLINE - 8.0:
                result = _optimize_aspect(x2, y2, r2)
                if result is not None:
                    x4, y4, r4, w4, h4 = result
                    consider(x4, y4, r4, w4, h4)

    if best_result is not None:
        return best_result

    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
