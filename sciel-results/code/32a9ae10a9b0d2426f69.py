"""Circle packing in perimeter-4 rectangle: DE global search + gradient local refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE

def _max_radii(centers, w, h):
    """Compute the maximum feasible radius for each circle given centers and box.
    Returns array of radii (non-negative)."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        r = min(xi, w - xi, yi, h - yi)
        if r <= 0:
            radii[i] = 0.0
            continue
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            r = min(r, d / 2.0)
            if r <= 0:
                break
        radii[i] = max(0.0, r)
    return radii

def _objective_penalty(x, W, H, mu=1e3):
    """Return (value, grad) where value = -sum(r) + mu * penalty.
    x is shape (N,3): cx, cy, r.
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    # Clamp to a reasonable range for numerical stability
    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)

    # Objective: maximize sum(r) -> minimize -sum(r)
    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary: r <= min(cx, cy, W-cx, H-cy), r > 0
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)

    # grad for boundary violation
    for i in range(n):
        if viol_b[i] <= 0:
            continue
        d_left = cx[i]
        d_bottom = cy[i]
        d_right = W - cx[i]
        d_top = H - cy[i]
        min_d = lo[i]
        g_cx = 0.0
        g_cy = 0.0
        eps = 1e-9
        if d_left <= min_d + eps:
            g_cx += 1.0
        if d_right <= min_d + eps:
            g_cx += -1.0
        if d_bottom <= min_d + eps:
            g_cy += 1.0
        if d_top <= min_d + eps:
            g_cy += -1.0
        cnt = (d_left <= min_d + eps) + (d_right <= min_d + eps) + \
              (d_bottom <= min_d + eps) + (d_top <= min_d + eps)
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0

    # r > 0: violation if r < 0
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            dv = 2.0 * mu * viol_r[i]
            grad[i, 2] += dv * (-1.0)

    # Centers in box (soft)
    mu_c = mu * 0.1
    viol_cx_lo = np.maximum(-cx, 0.0)
    viol_cx_hi = np.maximum(cx - W, 0.0)
    viol_cy_lo = np.maximum(-cy, 0.0)
    viol_cy_hi = np.maximum(cy - H, 0.0)
    val += mu_c * (np.sum(viol_cx_lo**2) + np.sum(viol_cx_hi**2) +
                   np.sum(viol_cy_lo**2) + np.sum(viol_cy_hi**2))
    for i in range(n):
        if viol_cx_lo[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_lo[i] * (-1.0)
        if viol_cx_hi[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_hi[i] * 1.0
        if viol_cy_lo[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_lo[i] * (-1.0)
        if viol_cy_hi[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_hi[i] * 1.0

    # Non-overlap: for each pair, max(r_i + r_j - dist, 0)^2
    mu_o = mu
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu_o * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu_o * viol * 1.0
                grad[j, 0] += 2.0 * mu_o * viol * (-1.0)
                grad[i, 2] += 2.0 * mu_o * viol * 1.0
                grad[j, 2] += 2.0 * mu_o * viol * 1.0
                continue
            dv = 2.0 * mu_o * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _grid_layout(kx, ky, W, H, shrink=0.999):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _random_layout(W, H):
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _optimize_from(x0, W, H, max_iter=200):
    """Run SLSQP from x0. Returns (best_x, best_sum)."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    def fun(x):
        v, _ = _objective_penalty(x, W, H)
        return v

    def grad(x):
        _, g = _objective_penalty(x, W, H)
        return g

    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10,
                                'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    # Second pass from best_x with tighter tolerance
    if _time_ok():
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 150, 'ftol': 1e-12,
                                     'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass

    return best_x, best_sum

def _differential_evolution(objective, bounds, pop_size=35, max_iter=300,
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=100.0):
    """Simple DE. Returns (best_x, best_f)."""
    n_dims = len(bounds)
    start = time.time()

    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])

    fitness = np.array([objective(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    for gen in range(max_iter):
        if time.time() - start > time_limit:
            break

        new_pop = []
        new_fitness = []
        for i in range(pop_size):
            r1, r2, r3 = random.sample(range(pop_size), 3)
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < crossover_prob:
                    trial[j] = mutant[j]
            f_trial = objective(trial)
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])

        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        gi = np.argmin(fitness)
        if fitness[gi] < best_f:
            best_f = fitness[gi]
            best_x = pop[gi].copy()

    return best_x, best_f

def _de_objective(x, w_init, h_init):
    """DE objective: negative sum of max feasible radii.
    x = [aspect_ratio, x1, y1, ..., xN, yN]
    aspect_ratio = w, h = 2 - w.
    """
    w = x[0]
    h = 2.0 - w
    if w < 0.2 or h < 0.2:
        return 1e6
    centers = x[1:].reshape(N, 2)
    centers = np.clip(centers, 0.0, np.array([w, h]))
    radii = _max_radii(centers, w, h)
    return -np.sum(radii)

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    random.seed(42)
    np.random.seed(42)

    # Phase 1: Generate diverse starting candidates
    candidates = []

    # Aspect ratios to try
    Ws = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5)]

    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.99)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    # Add random starts
    for W in Ws:
        H = 2.0 - W
        for _ in range(3):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    # Sort by initial score, best first
    candidates.sort(key=lambda t: -t[0])

    # Phase 2: Differential Evolution for global exploration
    # Use DE to find good starting points, then refine with gradient method
    time_left = DEADLINE - time.time() - 20.0  # leave 20s for refinement
    if time_left > 15:
        # DE over aspect ratio + centers
        # We'll run DE for a few different aspect ratio ranges
        de_results = []
        
        # DE configuration
        pop_size = 30 if time_left > 60 else 20
        max_iter = 150 if time_left > 60 else 80
        time_per_de = min(time_left / 3.0, 40.0)
        
        for w_center in [0.9, 1.1, 1.3]:
            if not _time_ok():
                break
            h_center = 2.0 - w_center
            if h_center < 0.3:
                continue
            
            # Bounds: aspect ratio in [w_center-0.2, w_center+0.2], centers in [0.01, 1.99]
            w_lo = max(0.3, w_center - 0.2)
            w_hi = min(1.7, w_center + 0.2)
            bounds = [(w_lo, w_hi)]
            for _ in range(N * 2):
                bounds.append((0.01, 1.99))
            
            # Initial population centered around w_center
            try:
                best_x, best_f = _differential_evolution(
                    lambda x: _de_objective(x, w_center, h_center),
                    bounds,
                    pop_size=pop_size,
                    max_iter=max_iter,
                    mutation=(0.5, 1.0),
                    crossover_prob=0.7,
                    time_limit=time_per_de
                )
                
                w = best_x[0]
                h = 2.0 - w
                centers = best_x[1:].reshape(N, 2)
                centers = np.clip(centers, 0.0, np.array([w, h]))
                radii = _max_radii(centers, w, h)
                s = np.sum(radii)
                
                if s > 0:
                    # Convert to (cx, cy, r) format
                    arr = np.column_stack([centers, radii])
                    de_results.append((s, w, h, arr))
            except Exception:
                pass
        
        # Add DE results to candidates
        for s, w, h, arr in de_results:
            candidates.append((s, w, h, arr))
        
        # Re-sort
        candidates.sort(key=lambda t: -t[0])

    # Phase 3: Gradient-based local refinement on top candidates
    top_k = min(10, len(candidates))
    for idx in range(top_k):
        if not _time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        remaining = DEADLINE - time.time()
        max_iter = 150 if remaining > 60 else 80
        x_opt, s_opt = _optimize_from(g0, W, H, max_iter=max_iter)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H

    # Phase 4: Additional random restarts from best found
    if _time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(6, int(remaining // 10))
        for _ in range(n_extra):
            if not _time_ok():
                break
            # Perturb best
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0 = np.clip(x0, 0.0, None)
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_opt, s_opt = _optimize_from(x0, best_W, best_H, max_iter=60)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt

    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.99)
        best_arr = g
        best_W, best_H = W, H

    # Final clamping to ensure validity
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])

    # Final validation: if invalid, fall back to a known-good grid
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        for s0, W, H, g0 in candidates[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = np.maximum(g0[:, 2], 1e-6)
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.minimum(r2, lo2)
            r2 = np.maximum(r2, 1e-6)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        # Ultimate fallback: 5x5 grid in unit square
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
