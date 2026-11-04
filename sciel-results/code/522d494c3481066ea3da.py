"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Dual-Voronoi gap-flow formulation.
- Each circle i has a Voronoi cell (implicitly).
- For every pair (i,j) and every wall, compute the exact largest circle that
  fits in the gap (tangent to both circles / circle+wall), giving a "flow"
  capacity c_ij that depends differentiably on the current configuration.
- Objective = sum of radii + lambda * (smoothed sum of gap capacities),
  which encourages the packing to be "tight" (like a max-flow dual).
- Optimize with multi-start L-BFGS-B gradient ascent on a smooth, differentiable
  objective (no non-smooth penalties; walls handled by smooth log-barrier +
  soft-relu for overlaps).
- Final tight local search for micro-improvements.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26


def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _wall_clear(x, y):
    """Distance from center (x,y) to the nearest wall."""
    return np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))


def _gap_capacity(x, y, r, beta=30.0):
    """
    Dual-Voronoi gap capacity: for each pair (i,j), compute the largest radius
    of a circle tangent to both circle i and circle j (the 'kissing' circle in
    the gap). Also for each circle i, the largest radius tangent to circle i
    and the nearest wall. Sum these as a smooth 'flow' signal.
    
    For two circles with centers d apart and radii r_i, r_j:
    The kissing circle in the gap (on the side away from both) has radius
    r_k = r_i * r_j / (d - r_i - r_j + r_i + r_j) ... actually use Descartes-like:
    If we place a circle tangent externally to both, its center is on the line
    between them extended. The radius r_k satisfies:
        dist(i, k) = r_i + r_k and dist(j, k) = r_j + r_k
    If k is between i and j: d = (r_i + r_k) + (r_j + r_k) => r_k = (d - r_i - r_j)/2
    If k is outside (beyond i): dist(i,k)=r_i+r_k, dist(j,k)=r_j+r_k, d + (r_i+r_k) = r_j+r_k => d = r_j - r_i (only if r_j > r_i + d)
    
    The meaningful gap capacity is the 'between' case: r_k = (d - r_i - r_j)/2 when d > r_i + r_j.
    We use a soft version: soft_relu(d - r_i - r_j)/2, which is 0 when overlapping and
    positive when there's a gap. This is differentiable.
    
    For wall: r_w = wall_clear_i - r_i (soft_relu).
    """
    n = len(x)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = d - r[i] - r[j]
            # soft relu: log1p(exp(beta*gap))/beta, but we want it to be ~0 for gap<0
            # and ~gap for gap>0. Use: (gap + log1p(exp(-beta*gap))/beta) is not right.
            # Simpler: use max(gap, 0) approximated by: gap/2 + (gap + log1p(exp(-beta*gap)))/beta/2 ... 
            # Actually just use: softplus(gap)/beta * beta = softplus(gap). We want the gap itself.
            # Use: gap * sigmoid(beta * gap) / (1 + exp(-beta*gap)) ... 
            # Simplest differentiable: (gap + np.sqrt(gap*gap + eps))/2 approx max(gap,0)
            # But let's just use: np.maximum(gap, 0) is not differentiable. 
            # Use: gap/2 + np.sqrt(gap*gap + 1e-12)/2  (smooth max)
            total += (gap + np.sqrt(gap * gap + 1e-12)) / 2.0
        # Wall gap
        wc = _wall_clear(x[i], y[i])
        wg = wc - r[i]
        total += (wg + np.sqrt(wg * wg + 1e-12)) / 2.0
    return total


def _gap_grad(x, y, r, beta=30.0):
    """Gradient of _gap_capacity w.r.t. x, y, r."""
    n = len(x)
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                d = 1e-12
            gap = d - r[i] - r[j]
            # d/dx_i of (gap + sqrt(gap^2+eps))/2 = 0.5*(1 + gap/sqrt(gap^2+eps)) * d(gap)/dx_i
            s = np.sqrt(gap * gap + 1e-12)
            factor = 0.5 * (1.0 + gap / s)
            # d(gap)/d(x_i) = dx/d, d(gap)/d(x_j) = -dx/d
            gx[i] += factor * dx / d
            gy[i] += factor * dy / d
            gx[j] -= factor * dx / d
            gy[j] -= factor * dy / d
            gr[i] -= factor
            gr[j] -= factor
        # Wall
        wc = _wall_clear(x[i], y[i])
        wg = wc - r[i]
        s = np.sqrt(wg * wg + 1e-12)
        factor = 0.5 * (1.0 + wg / s)
        # d(wc)/dx: 1 if x < 0.5 else -1 (approx, use sign)
        if x[i] < 0.5:
            gx[i] += factor
        else:
            gx[i] -= factor
        if y[i] < 0.5:
            gy[i] += factor
        else:
            gy[i] -= factor
        gr[i] -= factor
    return gx, gy, gr


def _smooth_objective(v, n, gap_weight=0.5, beta=20.0):
    """
    Objective to MINIMIZE: -sum(r) + penalties.
    Penalties: wall violation (soft), overlap (soft).
    Plus a NEGATIVE gap capacity term (we want to MAXIMIZE gap capacity,
    so we subtract it from the objective... wait, we want to minimize -sum(r) - gap_weight*gap).
    Actually: minimize  -sum(r) - gap_weight * gap_capacity + penalty.
    This encourages larger radii AND tighter packing (more gap capacity means
    the circles are well-separated in a structured way... actually gap capacity
    is large when circles are FAR apart. We want the OPPOSITE: we want them
    CLOSE (small gap) to maximize radii.
    
    Rethink: The 'flow' analogy: in a tight packing, the gaps are ZERO (circles
    are tangent). The gap capacity being ZERO is the goal. So we want to MINIMIZE
    the gap capacity (make it as small as possible = tightest packing).
    
    So objective: -sum(r) + gap_weight * gap_capacity + penalties.
    This encourages: large radii, small gaps (tight), no violations.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall penalty (soft)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.log1p(np.exp(beta * wall_viol)) / beta)

    # Overlap penalty (soft)
    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.log1p(np.exp(beta * viol)) / beta

    # Gap capacity (we want to minimize this = tight packing)
    gap = _gap_capacity(x, y, r)

    return -np.sum(r) + gap_weight * gap + 50.0 * (wall_pen + total_viol)


def _smooth_grad(v, n, gap_weight=0.5, beta=20.0):
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)

    # Wall penalty gradient
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sigmoid_w = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        sw = sigmoid_w[i]
        if sw > 1e-10:
            dists = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_idx = int(np.argmin(dists))
            grad[2*n + i] += 50.0 * sw
            if min_idx == 0:
                grad[i] -= 50.0 * sw
            elif min_idx == 1:
                grad[i] += 50.0 * sw
            elif min_idx == 2:
                grad[n + i] -= 50.0 * sw
            else:
                grad[n + i] += 50.0 * sw

    # Overlap penalty gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s > 1e-10:
                factor = 50.0 * s
                if dist > 1e-12:
                    grad[i] += factor * dx / dist
                    grad[n + i] += factor * dy / dist
                    grad[j] -= factor * dx / dist
                    grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor

    # Gap capacity gradient (we want to MINIMIZE gap, so add its gradient)
    gx, gy, gr = _gap_grad(x, y, r)
    grad[0:n] += gap_weight * gx
    grad[n:2*n] += gap_weight * gy
    grad[2*n:3*n] += gap_weight * gr

    # Objective: -sum(r) => gradient of -r is -1
    grad[2*n:3*n] -= 1.0

    return grad


def _gradient_optimize(v0, n, deadline, gap_weight=0.5, max_iter=2000):
    """Run L-BFGS-B to minimize the smooth objective."""
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.499))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, gap_weight=gap_weight)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, gap_weight=gap_weight)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
    except Exception:
        return v0

    return result.x


def _grid_init(side):
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = rng.uniform(0.02, 0.06, size=N)
    return np.column_stack([x, y]), r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Multi-start gradient optimization with different gap_weights and inits
    gap_weights = [0.1, 0.5, 1.0, 2.0]
    seeds = [1, 7, 42, 123, 999, 2024, 31415, 27182]

    for gi, gw in enumerate(gap_weights):
        for si, seed in enumerate(seeds):
            if time.time() > deadline - 20:
                break
            if gi == 0:
                # First: grid init
                xy = _grid_init(5)
                r = xy[:, 2].copy()
                xy = xy[:, :2].copy()
            elif gi == 1:
                xy = _grid_init(6)
                r = xy[:, 2].copy()
                xy = xy[:, :2].copy()
            else:
                xy, r = _random_init(seed + gi * 100)

            v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
            v_opt = _gradient_optimize(v0, N, deadline - 15, gap_weight=gw, max_iter=3000)

            new_xy = np.column_stack([v_opt[0:N], v_opt[N:2*N]])
            new_r = v_opt[2*N:3*N]

            # Clip to valid range
            for i in range(N):
                new_r[i] = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0],
                               new_xy[i, 1], 1.0 - new_xy[i, 1])
                new_r[i] = max(new_r[i], 0.001)
                new_xy[i, 0] = np.clip(new_xy[i, 0], new_r[i], 1.0 - new_r[i])
                new_xy[i, 1] = np.clip(new_xy[i, 1], new_r[i], 1.0 - new_r[i])

            total, valid = _eval_packing(new_xy, new_r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = new_xy.copy()
                best_r = new_r.copy()

    # Local search polish on best
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        final_xy, final_r, final_total, final_valid = _best_fit_local_search(
            best_xy, best_r, deadline - 3, grid_size=100)
        if final_valid and final_total > best_total:
            best_total = final_total
            best_xy = final_xy
            best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
