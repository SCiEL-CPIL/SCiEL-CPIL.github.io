"""Circle packing in unit square: 26 circles, maximize sum of radii.

Structural change from the previous SLSQP/Delaunay approach:
1. Use a soft-penalty (softplus) objective with analytic gradients and L-BFGS-B
   to handle all constraints simultaneously (walls, non-overlap, r>=0).
2. Add a Hertzian-contact / symplectic (velocity-Verlet) physics phase that
   grows radii and resolves overlaps through non-linear repulsive forces.
3. Keep the multi-restart + polish structure from the current program.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9

_start_time = time.time()
_TIME_LIMIT = 142.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    """Smooth derivative of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))


def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps
    for _ in range(300):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and analytic gradient for soft-plus penalty method.
    Returns (value, gradient).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Left x boundary
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    """
    High-precision polish using analytic gradient L-BFGS-B.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break

        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v

        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g

        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14,
                                      'gtol': 1e-12, 'maxfun': 10000})
            obj_val, _ = _objective_grad(result.x, penalty_weight)
            if obj_val < best_val:
                best_params = result.x
                best_val = obj_val
        except Exception:
            pass

    return best_params.reshape(N, 3)


def _physics_hertzian(xyz, time_limit=5.0, grow_rate=0.0008):
    """
    Hertzian-contact symplectic (velocity-Verlet) simulation.
    Circles interact via non-linear repulsive forces proportional to
    penetration_depth^(3/2). Radii grow slowly; overlaps are resolved
    through force integration with damping.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    vx = np.zeros(N)
    vy = np.zeros(N)

    best_score = np.sum(r)
    best_xyz = xyz.copy()

    dt = 0.003
    damping = 0.92
    k_hertz = 50.0  # Hertzian stiffness

    def _compute_forces(x, y, r):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise Hertzian repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(max(dist_sq, 1e-12))
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Hertzian force: F = k * overlap^(3/2)
                    F = k_hertz * (overlap ** 1.5)
                    inv_d = 1.0 / dist
                    fx[i] += F * dx * inv_d
                    fy[i] += F * dy * inv_d
                    fx[j] -= F * dx * inv_d
                    fy[j] -= F * dy * inv_d

        # Wall repulsion (soft)
        for i in range(N):
            # Left wall
            pen = r[i] - x[i]
            if pen > 0:
                fx[i] += k_hertz * (pen ** 1.5)
            # Right wall
            pen = x[i] + r[i] - 1.0
            if pen > 0:
                fx[i] -= k_hertz * (pen ** 1.5)
            # Bottom wall
            pen = r[i] - y[i]
            if pen > 0:
                fy[i] += k_hertz * (pen ** 1.5)
            # Top wall
            pen = y[i] + r[i] - 1.0
            if pen > 0:
                fy[i] -= k_hertz * (pen ** 1.5)

        return fx, fy

    # Initial forces
    fx, fy = _compute_forces(x, y, r)

    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break

        # Grow radii
        r *= (1.0 + grow_rate)
        # Clip to walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))

        # Velocity Verlet step
        x += vx * dt + 0.5 * fx * dt * dt
        y += vy * dt + 0.5 * fy * dt * dt

        fx_new, fy_new = _compute_forces(x, y, r)

        vx = (vx + 0.5 * (fx + fx_new) * dt) * damping
        vy = (vy + 0.5 * (fy + fy_new) * dt) * damping

        fx, fy = fx_new, fy_new

        # Hard clip positions to [0,1]
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        # Enforce wall constraints on radii
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(wall_max, 1e-6))

        # Check score
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xyz = np.column_stack([x, y, r]).copy()

        # Convergence check: if forces are tiny and no growth helps
        if iteration > 1000 and np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break

    return best_xyz


def _slsqp_step(pos, r, edges, time_budget):
    """One SLSQP step to optimize positions and radii (kept as fallback)."""
    n = len(r)
    x0 = np.concatenate([pos.flatten(), r])

    def objective(p):
        return -np.sum(p[2*n:])

    cons = []
    for i in range(n):
        xi = 2*i
        yi = 2*i+1
        ri = 2*n+i
        def c1(p, xi=xi, ri=ri): return p[xi] - p[ri]
        def j1(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi] = 1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c1, 'jac': j1})
        def c2(p, xi=xi, ri=ri): return 1.0 - p[xi] - p[ri]
        def j2(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi] = -1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c2, 'jac': j2})
        def c3(p, yi=yi, ri=ri): return p[yi] - p[ri]
        def j3(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi] = 1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c3, 'jac': j3})
        def c4(p, yi=yi, ri=ri): return 1.0 - p[yi] - p[ri]
        def j4(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi] = -1.0; v[ri] = -1.0; return v
        cons.append({'type': 'ineq', 'fun': c4, 'jac': j4})

    for (i, j) in edges:
        xi, yi = 2*i, 2*i+1
        xj, yj = 2*j, 2*j+1
        ri, rj = 2*n+i, 2*n+j
        def c_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]; dy = p[yi] - p[yj]
            return np.hypot(dx, dy) - p[ri] - p[rj]
        def j_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]; dy = p[yi] - p[yj]
            d = max(np.hypot(dx, dy), 1e-12)
            v = np.zeros(3*n)
            v[xi] = dx/d; v[yi] = dy/d
            v[xj] = -dx/d; v[yj] = -dy/d
            v[ri] = -1.0; v[rj] = -1.0
            return v
        cons.append({'type': 'ineq', 'fun': c_ov, 'jac': j_ov})

    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 50, 'ftol': 1e-8})
        if res.success or res.fun < objective(x0):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_r = res.x[2*n:]
            if _validate(np.column_stack([new_pos, new_r])):
                return new_pos, new_r
    except Exception:
        pass
    return pos, r


def _delaunay_edges(pos):
    """Build Delaunay triangulation edges."""
    from scipy.spatial import Delaunay
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _grow_and_relax(pos, r, edges, t_start, deadline):
    """Alternating growth and relaxation."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)

    for iteration in range(20):
        if time.time() > t_start + deadline:
            break

        # Grow radii
        for i in range(n):
            if time.time() > t_start + deadline:
                break
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)

        # SLSQP relaxation
        new_pos, new_r = _slsqp_step(pos, r, edges, 3.0)
        pos = new_pos
        r = new_r

        edges = _delaunay_edges(pos)

        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()

        if np.sum(r) - best_score < 1e-10:
            break

    return best_pos, best_r


def _polish(pos, r, t_start, deadline):
    """Final polish with multiple SLSQP iterations."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)

    for _ in range(4):
        if time.time() > t_start + deadline:
            break
        edges = _delaunay_edges(pos)
        new_pos, new_r = _slsqp_step(pos, r, edges, 5.0)
        if np.sum(new_r) > best_score:
            best_score = np.sum(new_r)
            best_pos = new_pos
            best_r = new_r
        pos = best_pos
        r = best_r

    return best_pos, best_r


def construct():
    global _start_time
    _start_time = time.time()
    DEADLINE = _TIME_LIMIT - 8.0  # safety margin

    best_score = -1.0
    best_result = None

    num_restarts = 10

    for restart in range(num_restarts):
        if _elapsed() > DEADLINE - 15.0:
            break

        rng = np.random.default_rng(2000 + restart)

        # Initial configurations
        if restart == 0:
            # Pure grid 5x5 + 1 extra
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side
                pos[i, 1] = (row + 0.5) / side
                r[i] = 0.4 / side
        elif restart == 1:
            # Grid with jitter
            side = 5
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side + rng.normal(0, 0.02)
                pos[i, 1] = (row + 0.5) / side + rng.normal(0, 0.02)
                r[i] = 0.35 / side
            pos = np.clip(pos, 0.05, 0.95)
        elif restart == 2:
            # Hexagonal-ish packing
            pos = np.zeros((N, 2))
            r = np.zeros(N)
            spacing = 0.19
            idx = 0
            for row in range(5):
                for col in range(5):
                    if idx >= N:
                        break
                    x = 0.1 + col * spacing + (0.5 * spacing if row % 2 else 0)
                    y = 0.1 + row * spacing * 0.866
                    if x < 0.95 and y < 0.95:
                        pos[idx, 0] = x
                        pos[idx, 1] = y
                        r[idx] = 0.08
                        idx += 1
            # Fill remaining
            for i in range(idx, N):
                pos[i, 0] = rng.uniform(0.1, 0.9)
                pos[i, 1] = rng.uniform(0.1, 0.9)
                r[i] = 0.04
        else:
            # Random
            pos = rng.random((N, 2)) * 0.8 + 0.1
            r = np.full(N, 0.02 + 0.01 * rng.random(N))

        # Ensure initial validity
        xyz = np.column_stack([pos, r])
        if not _validate(xyz):
            xyz = _enforce_validity(xyz.flatten())
            pos = xyz[:, :2]
            r = xyz[:, 2]

        # Phase 1: Hertzian physics relaxation with radius growth
        xyz = _physics_hertzian(xyz, time_limit=min(6.0, _remaining() - 20.0))
        pos = xyz[:, :2]
        r = xyz[:, 2]

        # Phase 2: Grow and relax (SLSQP)
        edges = _delaunay_edges(pos)
        pos, r = _grow_and_relax(pos, r, edges, _start_time, DEADLINE - 10.0)

        # Phase 3: Polish with SLSQP
        pos, r = _polish(pos, r, _start_time, DEADLINE - 5.0)

        # Phase 4: Analytic gradient polish (L-BFGS-B with softplus)
        xyz = np.column_stack([pos, r])
        xyz = _polish_analytic(xyz, time_budget=min(5.0, _remaining() - 3.0))
        pos = xyz[:, :2]
        r = xyz[:, 2]

        # Final validity enforcement
        xyz = _enforce_validity(xyz.flatten())
        score = np.sum(xyz[:, 2])

        if score > best_score:
            best_score = score
            best_result = xyz

        # Early exit if very good
        if best_score > 3.0:
            break

    if best_result is None:
        # Fallback grid
        side = 5
        pos = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            r[i] = 0.4 / side
        best_result = np.column_stack([pos, r])

    return best_result


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
