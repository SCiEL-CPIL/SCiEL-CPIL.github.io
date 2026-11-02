"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square and ensure non-overlap by shrinking if necessary."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
        
    for _iter in range(15):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-5
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break
            
    return circles


def _objective_penalty(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective with soft penalty for constraint violations.
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    # Full pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Gradient of the penalty objective.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    n = len(r)
    
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Objective gradient: -1 for each radius
    gr[:] = -1.0
    
    # Boundary penalty gradients
    def _sp_grad(z, eps):
        """Gradient of softplus(z, eps) = eps * logaddexp(0, z/eps)"""
        # d/dz [eps * log(1 + exp(z/eps))] = 1 / (1 + exp(-z/eps)) = sigmoid(z/eps)
        return 1.0 / (1.0 + np.exp(-z / eps))
    
    # Left boundary: lx = r - x
    lx = r - x
    sp_grad_lx = _sp_grad(lx, smooth_eps)
    # d/dx: -sp_grad, d/dr: +sp_grad
    gx -= 2 * sp_grad_lx * penalty_weight
    gr += 2 * sp_grad_lx * penalty_weight
    
    # Right boundary: ux = x + r - 1
    ux = x + r - 1.0
    sp_grad_ux = _sp_grad(ux, smooth_eps)
    gx += 2 * sp_grad_ux * penalty_weight
    gr += 2 * sp_grad_ux * penalty_weight
    
    # Bottom boundary: ly = r - y
    ly = r - y
    sp_grad_ly = _sp_grad(ly, smooth_eps)
    gy -= 2 * sp_grad_ly * penalty_weight
    gr += 2 * sp_grad_ly * penalty_weight
    
    # Top boundary: uy = y + r - 1
    uy = y + r - 1.0
    sp_grad_uy = _sp_grad(uy, smooth_eps)
    gy += 2 * sp_grad_uy * penalty_weight
    gr += 2 * sp_grad_uy * penalty_weight
    
    # Negative radius penalty
    neg_r = -r
    sp_grad_neg = _sp_grad(neg_r, smooth_eps)
    gr -= 2 * sp_grad_neg * penalty_weight
    
    # Pairwise overlap penalties
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    
    sp_grad_ov = _sp_grad(overlap, smooth_eps)
    # d/d(overlap) = sp_grad
    # overlap = r_i + r_j - dist
    # d(overlap)/d(x_i) = -dx/dist, d(overlap)/d(x_j) = +dx/dist
    # d(overlap)/d(y_i) = -dy/dist, d(overlap)/d(y_j) = +dy/dist
    # d(overlap)/d(r_i) = 1, d(overlap)/d(r_j) = 1
    
    coeff = 2 * sp_grad_ov * penalty_weight
    dist_safe = np.maximum(dist, 1e-12)
    
    gx[idx_i] -= coeff * dx / dist_safe
    gx[idx_j] += coeff * dx / dist_safe
    gy[idx_i] -= coeff * dy / dist_safe
    gy[idx_j] += coeff * dy / dist_safe
    gr[idx_i] += coeff
    gr[idx_j] += coeff
    
    # Combine into flat gradient
    grad = np.zeros(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    
    return grad


def _get_contact_edges(circles):
    """Get contact edges from Delaunay triangulation of centers."""
    centers = circles[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return None


def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _force_relax(circles, n_steps=200, dt=0.001):
    """
    Physics-based force-directed relaxation.
    Repulsive forces between overlapping circles, attractive to expand radii.
    """
    circles = circles.copy()
    n = len(circles)
    
    for step in range(n_steps):
        if _remaining() < 5:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                    # Push radii apart
                    fr[i] -= 0.1 * overlap
                    fr[j] -= 0.1 * overlap
        
        # Boundary forces
        for i in range(n):
            r = circles[i, 2]
            x = circles[i, 0]
            y = circles[i, 1]
            
            # Left boundary
            if x - r < 0:
                fx[i] += 10.0 * (r - x)
            # Right boundary
            if x + r > 1:
                fx[i] -= 10.0 * (x + r - 1)
            # Bottom boundary
            if y - r < 0:
                fy[i] += 10.0 * (r - y)
            # Top boundary
            if y + r > 1:
                fy[i] -= 10.0 * (y + r - 1)
            
            # Encourage larger radii (small attractive force)
            fr[i] += 0.01
        
        # Update positions
        for i in range(n):
            circles[i, 0] += dt * fx[i]
            circles[i, 1] += dt * fy[i]
            circles[i, 2] += dt * fr[i]
            
            # Clip to valid range
            circles[i, 2] = np.clip(circles[i, 2], 0.01, 0.5)
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
    
    return circles


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty and gradient-based method."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 0: Force-directed relaxation to get a good starting point
    remaining = deadline - time.time()
    if remaining > 10:
        circles = np.column_stack([params[0::3], params[1::3], params[2::3]])
        circles = _force_relax(circles, n_steps=100, dt=0.0005)
        if _validate(circles):
            params = circles.flatten()
            best_params = params.copy()
            best_score = np.sum(params[2::3])

    # Phase 1: Low penalty with gradient
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 5.0
        try:
            res = minimize(_objective_penalty, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty with gradient
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 50.0
        try:
            res = minimize(_objective_penalty, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty with gradient
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 500.0
        try:
            res = minimize(_objective_penalty, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Final polish with very high penalty and tight tolerance
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 2000.0
        try:
            res = minimize(_objective_penalty, best_params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_penalty(best_params, penalty, 1e-5):
                best_params = res.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass

    return best_params, best_score


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses multi-start optimization with force-directed relaxation and gradient-based refinement.
    """
    start_time = time.time()
    deadline = _start_time + _TIME_LIMIT

    # Generate multiple initial configurations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_corner_weighted(seed=999),
    ]

    best_overall = None
    best_score_overall = -np.inf

    for i, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Allocate time proportionally
        alloc = min(remaining - 5, 25)
        local_deadline = time.time() + alloc

        params, score = _optimize_single(xyz0, local_deadline)

        # Validate
        circles = np.column_stack([params[0::3], params[1::3], params[2::3]])
        if _validate(circles):
            if score > best_score_overall:
                best_score_overall = score
                best_overall = circles.copy()
        else:
            # Try to fix
            circles_fixed = _fix_boundaries(circles)
            if _validate(circles_fixed):
                score_fixed = np.sum(circles_fixed[:, 2])
                if score_fixed > best_score_overall:
                    best_score_overall = score_fixed
                    best_overall = circles_fixed.copy()

    # Always ensure the result is valid by fixing boundaries and validating
    if best_overall is not None:
        # Try to repair the best result
        repaired = _fix_boundaries(best_overall)
        if _validate(repaired):
            best_overall = repaired
        else:
            # If repair failed, discard and fall back
            best_overall = None

    if best_overall is None:
        # Fallback: grid of 26 circles with small radii
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result

    return best_overall

# EVOLVE-BLOCK-END
