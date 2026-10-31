"""Circle packing in perimeter-4 rectangle: SLSQP constraint refinement + adaptive expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21


def _is_valid(x, y, r, w, h, tol=1e-7):
    """Check if packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(x < r + tol) or np.any(x > w - r - tol):
        return False
    if np.any(y < r + tol) or np.any(y > h - r - tol):
        return False
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d2 = dx * dx + dy * dy
    r_sum = r[iu[0]] + r[iu[1]]
    if np.any(d2 < (r_sum + tol) ** 2):
        return False
    return True


def _slsqp_refine(x, y, r, w, h, time_budget=8.0):
    """SLSQP-based refinement: maximize sum(r) with explicit non-overlap and wall constraints."""
    start = time.time()

    # Variables: x[0..N-1], y[N..2N-1], r[2N..3N-1]
    def objective(v):
        return -np.sum(v[2*N:3*N])

    # Inequality constraints: for each pair i<j, (xi-xj)^2+(yi-yj)^2 >= (ri+rj)^2
    # SLSQP needs g(v) >= 0 form. Use: d^2 - (ri+rj)^2 >= 0
    # Also wall constraints: xi - ri >= 0, w - xi - ri >= 0, etc.

    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])

    def constraints(v):
        """Return array of constraint values, all must be >= 0."""
        xi = v[:N]
        yi = v[N:2*N]
        ri = v[2*N:3*N]
        vals = []

        # Wall constraints
        vals.extend(xi - ri)           # xi >= ri
        vals.extend(w - xi - ri)       # w - xi >= ri
        vals.extend(yi - ri)           # yi >= ri
        vals.extend(h - yi - ri)       # h - yi >= ri

        # Pair non-overlap: d^2 - (ri+rj)^2 >= 0
        dx = xi[iu[0]] - xi[iu[1]]
        dy = yi[iu[0]] - yi[iu[1]]
        d2 = dx * dx + dy * dy
        rs = ri[iu[0]] + ri[iu[1]]
        vals.extend(d2 - rs * rs)

        return np.array(vals)

    def jac_objective(v):
        j = np.zeros(3 * N)
        j[2*N:3*N] = -1.0
        return j

    def jac_constraints(v):
        """Jacobian of constraints w.r.t. v. Shape: (n_constraints, 3N)."""
        xi = v[:N]
        yi = v[N:2*N]
        ri = v[2*N:3*N]
        n_c = 4 * N + n_pairs
        J = np.zeros((n_c, 3 * N))

        # Wall: xi - ri >= 0  -> d/dxi = 1, d/dri = -1
        J[:N, :N] = np.eye(N)
        J[:N, 2*N:3*N] = -np.eye(N)
        # w - xi - ri >= 0 -> d/dxi = -1, d/dri = -1
        J[N:2*N, :N] = -np.eye(N)
        J[N:2*N, 2*N:3*N] = -np.eye(N)
        # yi - ri >= 0
        J[2*N:3*N, N:2*N] = np.eye(N)
        J[2*N:3*N, 2*N:3*N] = -np.eye(N)
        # h - yi - ri >= 0
        J[3*N:4*N, N:2*N] = -np.eye(N)
        J[3*N:4*N, 2*N:3*N] = -np.eye(N)

        # Pair constraints: d2 - rs^2 >= 0
        # d/dxi_k: 2*(xi_k - xi_j) for pair (i,j) where k=i; -2*(xi_k-xi_j) where k=j
        dx = xi[iu[0]] - xi[iu[1]]
        dy = yi[iu[0]] - yi[iu[1]]
        rs = ri[iu[0]] + ri[iu[1]]

        base = 4 * N
        for k in range(n_pairs):
            i = iu[0][k]
            j = iu[1][k]
            row = base + k
            J[row, i] = 2.0 * dx[k]
            J[row, j] = -2.0 * dx[k]
            J[row, N + i] = 2.0 * dy[k]
            J[row, N + j] = -2.0 * dy[k]
            J[row, 2*N + i] = -2.0 * rs[k]
            J[row, 2*N + j] = -2.0 * rs[k]

        return J

    v0 = np.concatenate([x, y, r])
    bounds = []
    for i in range(N):
        bounds.append((r[i] * 0.5, 1.0))  # x
    for i in range(N):
        bounds.append((r[i] * 0.5, 1.0))  # y
    for i in range(N):
        bounds.append((r[i] * 0.9, 1.0))  # r

    maxiter = int(min(300, time_budget * 30))
    try:
        res = minimize(objective, v0, method='SLSQP', jac=jac_objective,
                       constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jac_constraints}],
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.success or res.fun < -np.sum(r) + 1e-8:
            vo = res.x
            xo, yo, ro = vo[:N], vo[N:2*N], vo[2*N:3*N]
            if _is_valid(xo, yo, ro, w, h, tol=1e-6):
                return xo, yo, ro, w
    except Exception:
        pass

    return None


def _max_radius_for_circle(i, x, y, r, w, h):
    """Compute maximum radius for circle i given current positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _try_expand_circle(i, x, y, r, w, h):
    """Try to expand circle i by moving it and increasing radius. Returns (new_x, new_y, new_r, success)."""
    r_max = _max_radius_for_circle(i, x, y, r, w, h)
    if r_max <= r[i] + 1e-9:
        return None

    # Binary search for max radius at current position
    lo, hi = r[i], r_max
    for _ in range(20):
        mid = (lo + hi) / 2.0
        r_test = r.copy()
        r_test[i] = mid
        if _is_valid(x, y, r_test, w, h, tol=1e-8):
            lo = mid
        else:
            hi = mid
    if lo > r[i] + 1e-9:
        return (x[i], y[i], lo, True)
    return None


def _try_move_expand(i, x, y, r, w, h):
    """Try moving circle i to a new position that allows larger radius."""
    best = None
    best_r = r[i]

    # 16 directions with multiple step sizes
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    base_steps = [0.02, 0.04, 0.06, 0.08, 0.1]

    for angle in angles:
        for step in base_steps:
            nx = x[i] + step * np.cos(angle)
            ny = y[i] + step * np.sin(angle)

            # Check walls
            if nx < r[i] - 1e-9 or nx > w - r[i] + 1e-9:
                continue
            if ny < r[i] - 1e-9 or ny > h - r[i] + 1e-9:
                continue

            # Check overlaps
            valid = True
            for j in range(N):
                if i == j:
                    continue
                dx = nx - x[j]
                dy = ny - y[j]
                d2 = dx * dx + dy * dy
                rs = r[i] + r[j]
                if d2 < rs * rs - 1e-9:
                    valid = False
                    break
            if not valid:
                continue

            # Compute max radius at new position
            r_wall = min(nx, w - nx, ny, h - ny)
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = nx - x[j]
                dy = ny - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > best_r + 1e-7:
                # Binary search to find actual max valid radius
                lo, hi2 = r[i], r_max
                for _ in range(15):
                    mid = (lo + hi2) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    x_test = x.copy()
                    x_test[i] = nx
                    y_test = y.copy()
                    y_test[i] = ny
                    if _is_valid(x_test, y_test, r_test, w, h, tol=1e-8):
                        lo = mid
                    else:
                        hi2 = mid
                if lo > best_r:
                    best_r = lo
                    best = (nx, ny, lo)

    if best is not None:
        return best
    return None


def _expansion_polish(x, y, r, w, h, time_budget=5.0):
    """Aggressive iterative expansion: in-place radius growth + position re-optimization."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    start = time.time()

    for outer in range(80):
        if time.time() - start > time_budget:
            break

        improved = False
        # Process smallest circles first (more room to grow)
        order = np.argsort(r)

        for i in order:
            if time.time() - start > time_budget:
                break

            # Phase 1: Try in-place radius expansion
            res = _try_expand_circle(i, x, y, r, w, h)
            if res is not None:
                x[i], y[i], r[i], _ = res
                improved = True
                continue

            # Phase 2: Try move + expand
            res2 = _try_move_expand(i, x, y, r, w, h)
            if res2 is not None:
                x[i], y[i], r[i] = res2
                improved = True

        if not improved:
            break

    return x, y, r


def _lbfgs_optimize(x, y, r, w, h, time_budget=10.0):
    """Optimize using L-BFGS-B with soft penalties, then SLSQP refine."""
    start = time.time()

    def to_vec(x, y, r, w):
        v = np.zeros(3 * N + 1)
        v[0::3][:N] = x
        v[1::3][:N] = y
        v[2::3][:N] = r
        v[-1] = w
        return v

    def from_vec(v):
        x = v[0::3][:N].copy()
        y = v[1::3][:N].copy()
        r = v[2::3][:N].copy()
        w = v[-1]
        return x, y, r, w

    def objective(v):
        x, y, r, w = from_vec(v)
        h = 2.0 - w
        penalty = 0.0
        penalty += 100.0 * np.sum(np.maximum(r - x, 0.0)**2)
        penalty += 100.0 * np.sum(np.maximum(x + r - w, 0.0)**2)
        penalty += 100.0 * np.sum(np.maximum(r - y, 0.0)**2)
        penalty += 100.0 * np.sum(np.maximum(y + r - h, 0.0)**2)
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(r_sum - d, 0.0)
        penalty += 100.0 * np.sum(overlap**2)
        return -np.sum(r) + penalty

    v0 = to_vec(x, y, r, w)
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 1.0))
        bounds.append((1e-4, 1.0))
        bounds.append((1e-4, 1.0))
    bounds.append((0.3, 1.7))

    maxiter = int(min(500, time_budget * 50))
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        v_opt = res.x
        xo, yo, ro, wo = from_vec(v_opt)
        ho = 2.0 - wo
        if _is_valid(xo, yo, ro, wo, ho):
            # Now refine with SLSQP
            remaining = time_budget - (time.time() - start)
            if remaining > 2.0:
                refined = _slsqp_refine(xo, yo, ro, wo, ho, time_budget=remaining)
                if refined is not None:
                    return refined
            return xo, yo, ro, wo
        return None
    except Exception:
        return None


def _force_relax(x, y, r, w, h, max_iter=2000):
    """Force-directed relaxation to resolve overlaps."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    iu = np.triu_indices(N, k=1)

    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall repulsion
        fx -= np.maximum(r - x, 0.0) * 50
        fx += np.maximum(x + r - w, 0.0) * 50
        fy -= np.maximum(r - y, 0.0) * 50
        fy += np.maximum(y + r - h, 0.0) * 50

        # Pairwise repulsion
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        d = np.sqrt(np.maximum(d2, 1e-12))
        overlap = np.maximum(r_sum - d, 0.0)

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 10.0
        ux = dx / d
        uy = dy / d

        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            fx[i] += ux[k] * f_pairs[k]
            fy[i] += uy[k] * f_pairs[k]
            fx[j] -= ux[k] * f_pairs[k]
            fy[j] -= uy[k] * f_pairs[k]

        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        dt = 0.005 if max_force > 0.1 else 0.002

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y, r


def _generate_initial_guesses():
    """Generate diverse initial packings for multi-start."""
    guesses = []
    side = 5

    # Grid-based with various aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.3, 0.75]
    for w0 in aspect_ratios:
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
            r[i] = min(w0 / side, h0 / side) * 0.38
        guesses.append((x, y, r, w0))

    # Random starts with fixed seeds
    for seed_val in range(8):
        rng = np.random.default_rng(seed_val * 7919 + 42)
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


def _get_fallback():
    """Generate a valid fallback packing."""
    w = 1.0
    h = 1.0
    side = 5
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side
        y[i] = (row + 0.5) / side
        r[i] = 0.09
    if not _is_valid(x, y, r, w, h, tol=1e-6):
        r = r * 0.9
    return np.column_stack((x, y, r))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    guesses = _generate_initial_guesses()

    # Phase 1: Multi-start optimization
    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        remaining = deadline - time.time()
        if remaining < 12.0:
            break

        h0 = 2.0 - w0

        # Step 1: Force relax
        x, y, r = _force_relax(x0, y0, r0, w0, h0, max_iter=600)

        # Step 2: L-BFGS-B + SLSQP refinement
        opt_budget = min(10.0, remaining - 10.0)
        if opt_budget > 3.0:
            result = _lbfgs_optimize(x, y, r, w0, h0, time_budget=opt_budget)
            if result is not None:
                x, y, r, w = result
                h = 2.0 - w
            else:
                w = w0
                h = h0
        else:
            w = w0
            h = h0

        # Step 3: Expansion polish
        polish_budget = min(8.0, deadline - time.time() - 8.0)
        if polish_budget > 1.0:
            x, y, r = _expansion_polish(x, y, r, w, h, time_budget=polish_budget)

        if _is_valid(x, y, r, w, h, tol=1e-7):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

    # Phase 2: Deep polish of best result with SLSQP
    if best_result is not None and time.time() < deadline - 15:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # SLSQP refinement
        slsqp_budget = min(12.0, deadline - time.time() - 12.0)
        if slsqp_budget > 3.0:
            refined = _slsqp_refine(x, y, r, w, h, time_budget=slsqp_budget)
            if refined is not None:
                x, y, r, w = refined
                h = 2.0 - w

        # Expansion polish
        polish_budget = min(10.0, deadline - time.time() - 8.0)
        if polish_budget > 1.0:
            x, y, r = _expansion_polish(x, y, r, w, h, time_budget=polish_budget)

        if _is_valid(x, y, r, w, h, tol=1e-7):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

    # Phase 3: Aspect ratio perturbation
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w_orig = best_result[-1]
        h_orig = 2.0 - w_orig

        for w_try in [w_orig * 0.92, w_orig * 1.08, w_orig * 0.88, w_orig * 1.12, w_orig * 0.95, w_orig * 1.05]:
            if time.time() > deadline - 6:
                break
            if w_try < 0.35 or w_try > 1.65:
                continue
            h_try = 2.0 - w_try

            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy)

            xs, ys, rs = _force_relax(xs, ys, rs, w_try, h_try, max_iter=400)
            pb = min(4.0, deadline - time.time() - 4.0)
            if pb > 1.0:
                xs, ys, rs = _expansion_polish(xs, ys, rs, w_try, h_try, time_budget=pb)

            if _is_valid(xs, ys, rs, w_try, h_try, tol=1e-7):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = xs
                    z[1::3][:N] = ys
                    z[2::3][:N] = rs
                    z[-1] = w_try
                    best_result = z.copy()

    if best_result is None:
        return _get_fallback()

    x = best_result[0::3][:N].copy()
    y = best_result[1::3][:N].copy()
    r = best_result[2::3][:N].copy()
    w_final = best_result[-1]
    h_final = 2.0 - w_final

    # Final safety: shrink radii slightly to ensure strict validity
    lo, hi = 0.85, 1.0
    for _ in range(25):
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w_final, h_final, tol=1e-6):
            lo = mid
        else:
            hi = mid

    r_final = r * lo
    if not _is_valid(x, y, r_final, w_final, h_final, tol=1e-6):
        return _get_fallback()

    return np.column_stack((x, y, r_final))
