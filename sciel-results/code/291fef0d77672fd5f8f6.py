"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # seconds, safety margin before 150s limit


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return DEADLINE - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
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
    """Project circles inside unit square."""
    circles = circles.copy()
    for i in range(N):
        r = max(0, circles[i, 2])
        if r < 0.5:
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        else:
            circles[i, 0] = 0.5
            circles[i, 1] = 0.5
        circles[i, 2] = r
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


def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective using Delaunay-based contact graph for overlap constraints.
    Faster than full pairwise when contact graph is sparse.
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

    # Overlaps on contact edges only
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps) ** 2
    else:
        # Fallback to full pairwise
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


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


def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 1: Low penalty, focus on layout
    penalty = 5.0
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 50.0
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty, tighten constraints
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 500.0
        try:
            res = minimize(_objective_penalty, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_penalty(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 4: Delaunay-based refinement
    remaining = deadline - time.time()
    if remaining > 3:
        circles = params.reshape(N, 3)
        circles = _fix_boundaries(circles)
        edges = _get_contact_edges(circles)
        if edges is not None:
            try:
                res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                              bounds=bounds, args=(100.0, 1e-4, edges),
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < _objective_delaunay(params, 100.0, 1e-4, edges):
                    params = res.x
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
            except Exception:
                pass

    return best_params.reshape(N, 3)


def _monte_carlo_refine(circles, start_time, deadline, n_steps=5000):
    """Monte Carlo refinement with annealing."""
    n = len(circles)
    current = circles.copy()
    best = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    temp = 0.05
    step_size = 0.015
    
    for step in range(n_steps):
        if time.time() - start_time > deadline - 5:
            break
        
        # Pick a random circle to move
        idx = np.random.randint(n)
        move_type = np.random.choice(3)
        
        new_circles = current.copy()
        if move_type == 0:
            new_circles[idx, 0] += np.random.normal(0, step_size)
            new_circles[idx, 1] += np.random.normal(0, step_size)
        elif move_type == 1:
            new_circles[idx, 2] *= np.exp(np.random.normal(0, step_size * 0.5))
            new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, 0.45)
        else:
            new_circles[idx, 0] += np.random.normal(0, step_size)
            new_circles[idx, 1] += np.random.normal(0, step_size)
            new_circles[idx, 2] *= np.exp(np.random.normal(0, step_size * 0.3))
            new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.01, 0.45)
        
        # Ensure positions in [0,1]
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, 1)
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, 1)
        
        # Compute energy
        old_energy = _objective_penalty(current.flatten(), 100, 1e-4)
        new_energy = _objective_penalty(new_circles.flatten(), 100, 1e-4)
        delta_e = new_energy - old_energy
        
        # Metropolis acceptance
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
            current = new_circles
        
        # Track best
        current_sum = np.sum(current[:, 2])
        if _validate(current) and current_sum > best_sum:
            best_sum = current_sum
            best = current.copy()
        
        # Anneal
        if step % 100 == 0:
            temp *= 0.95
            step_size *= 0.97
    
    return best


def construct():
    global _start_time
    _start_time = time.time()
    
    # Multiple initialization strategies
    initializations = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_corner_weighted(seed=999)
    ]
    
    best_circles = None
    best_sum = 0.0
    
    for init in initializations:
        if _remaining() < 15:
            break
        
        # Optimize
        optimized = _optimize_single(init, DEADLINE)
        optimized = _fix_boundaries(optimized)
        
        # Check validity and sum
        if _validate(optimized):
            s = np.sum(optimized[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = optimized.copy()
    
    # Monte Carlo refinement
    if best_circles is not None and _remaining() > 10:
        refined = _monte_carlo_refine(best_circles, _start_time, DEADLINE)
        if _validate(refined):
            s = np.sum(refined[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = refined.copy()
    
    # Final polish with high-precision optimization
    if best_circles is not None and _remaining() > 5:
        polished = _optimize_single(best_circles, DEADLINE)
        polished = _fix_boundaries(polished)
        if _validate(polished):
            s = np.sum(polished[:, 2])
            if s > best_sum:
                best_circles = polished.copy()
                best_sum = s
    
    if best_circles is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
