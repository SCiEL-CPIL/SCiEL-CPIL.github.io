"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 148  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


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
        circles[i, 0] = np.clip(circles[i, 0], r, 1 - r) if r < 0.5 else 0.5
        circles[i, 1] = np.clip(circles[i, 1], r, 1 - r) if r < 0.5 else 0.5
        circles[i, 2] = r
    return circles


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


def _interior_point_objective(params, mu, contact_edges=None):
    """
    Interior-point objective with log-barrier for boundary constraints
    and slack-based formulation for overlap constraints.
    
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    mu: barrier parameter (decreased over iterations)
    contact_edges: list of (i,j) pairs for overlap constraints
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    obj = -np.sum(r)
    
    # Log-barrier for boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r, r > 0
    # Use softplus-based barrier for numerical stability
    eps = 1e-6
    lx = r - x  # should be <= 0
    ux = x + r - 1.0  # should be <= 0
    ly = r - y
    uy = y + r - 1.0
    
    # Log-barrier: -mu * log(max(eps, constraint))
    barrier = 0.0
    barrier += -mu * np.sum(np.log(np.maximum(eps, 1.0 + lx)))  # 1+lx = 1 + r - x
    barrier += -mu * np.sum(np.log(np.maximum(eps, 1.0 - ux)))  # 1-ux = 1 - x - r + 1 = 2 - x - r... 
    # Actually let's use proper barriers:
    # For x >= r: barrier = -mu * log(x - r + eps)
    # For x <= 1-r: barrier = -mu * log(1 - x - r + eps)
    
    # Redo with correct barriers
    barrier = 0.0
    barrier += -mu * np.sum(np.log(np.maximum(eps, x - r)))
    barrier += -mu * np.sum(np.log(np.maximum(eps, 1.0 - x - r)))
    barrier += -mu * np.sum(np.log(np.maximum(eps, y - r)))
    barrier += -mu * np.sum(np.log(np.maximum(eps, 1.0 - y - r)))
    barrier += -mu * np.sum(np.log(np.maximum(eps, r)))
    
    obj += barrier
    
    # Overlap constraints with slack variables (implicit via penalty)
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            overlap = r[i] + r[j] - dist
            # Soft penalty for overlap
            obj += mu * 100.0 * _softplus(overlap, 1e-4) ** 2
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        obj += mu * 100.0 * np.sum(_softplus(overlap, 1e-4) ** 2)
    
    return obj


def _kkt_optimize(params0, deadline, mu_start=0.1, mu_end=1e-5, max_iters=200):
    """
    Primal-dual interior-point style optimization.
    Decreases barrier parameter mu geometrically, solving subproblems at each level.
    """
    params = params0.copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 1 - 1e-6))
        bounds.append((1e-6, 1 - 1e-6))
        bounds.append((1e-6, 0.5))
    
    n_mu_steps = 30
    mu_ratio = (mu_end / mu_start) ** (1.0 / n_mu_steps)
    
    for step in range(n_mu_steps):
        if time.time() > deadline - 2:
            break
        
        mu = mu_start * (mu_ratio ** step)
        
        # Get contact edges for current solution
        circles = params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        
        try:
            res = minimize(_interior_point_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(mu, contact_edges),
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < _interior_point_objective(params, mu, contact_edges):
                params = res.x
                score = np.sum(params[2::3])
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception:
            pass
    
    return best_params


def _physics_relax(circles, deadline, n_steps=200):
    """
    Simple physics-based relaxation: circles repel each other and are attracted to boundaries.
    Helps escape local minima before gradient-based optimization.
    """
    circles = circles.copy()
    dt = 0.01
    damping = 0.95
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        forces = np.zeros_like(circles)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    f = (min_dist - dist) / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                forces[i, 0] += (r[i] - x[i]) * 10
            if x[i] > 1 - r[i]:
                forces[i, 0] -= (x[i] - (1 - r[i])) * 10
            if y[i] < r[i]:
                forces[i, 1] += (r[i] - y[i]) * 10
            if y[i] > 1 - r[i]:
                forces[i, 1] -= (y[i] - (1 - r[i])) * 10
        
        # Update positions with velocity damping
        circles[:, :2] += forces[:, :2] * dt
        circles = _fix_boundaries(circles)
    
    return circles


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
    """Optimize from a single initial config using interior-point method."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    # Phase 0: Physics relaxation to escape bad local minima
    remaining = deadline - time.time()
    if remaining > 15:
        relaxed = _physics_relax(xyz0, deadline)
        score = np.sum(relaxed[:, 2])
        if score > best_score:
            best_score = score
            best_params = relaxed.flatten().copy()
            params = relaxed.flatten().copy()
    
    # Phase 1: Interior-point optimization with decreasing barrier
    remaining = deadline - time.time()
    if remaining > 10:
        ip_result = _kkt_optimize(params, deadline, mu_start=0.1, mu_end=1e-5, max_iters=200)
        score = np.sum(ip_result[2::3])
        if score > best_score:
            best_score = score
            best_params = ip_result.copy()
    
    # Phase 2: Fine-tuning with high-precision L-BFGS on contact graph
    remaining = deadline - time.time()
    if remaining > 5:
        circles = best_params.reshape(N, 3)
        contact_edges = _get_contact_edges(circles)
        
        def _final_objective(p):
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            obj = -np.sum(r)
            
            # Boundary penalties
            eps = 1e-6
            obj += 500.0 * np.sum(_softplus(r - x, eps) ** 2)
            obj += 500.0 * np.sum(_softplus(x + r - 1.0, eps) ** 2)
            obj += 500.0 * np.sum(_softplus(r - y, eps) ** 2)
            obj += 500.0 * np.sum(_softplus(y + r - 1.0, eps) ** 2)
            obj += 500.0 * np.sum(_softplus(-r, eps) ** 2)
            
            # Overlap penalties on contact edges
            if contact_edges is not None:
                for (i, j) in contact_edges:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                    overlap = r[i] + r[j] - dist
                    obj += 500.0 * _softplus(overlap, eps) ** 2
            
            return obj
        
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 1 - 1e-6))
            bounds.append((1e-6, 1 - 1e-6))
            bounds.append((1e-6, 0.5))
        
        try:
            res = minimize(_final_objective, best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _final_objective(best_params):
                score = np.sum(res.x[2::3])
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Validate and fix
    best_circles = best_params.reshape(N, 3)
    best_circles = _fix_boundaries(best_circles)
    if not _validate(best_circles):
        best_circles[:, 2] *= 0.98
        best_circles = _fix_boundaries(best_circles)
        if not _validate(best_circles):
            best_circles = xyz0.copy()
    
    return best_circles


def construct():
    global _start_time
    _start_time = time.time()

    deadline = _start_time + _TIME_LIMIT - 3  # 3s safety margin

    # Try multiple initializations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
    ]

    best_circles = None
    best_val = -np.inf

    for idx, init in enumerate(inits):
        if time.time() > deadline - 8:
            break

        # Allocate time: more time for first init, less for later ones
        remaining = deadline - time.time()
        if idx == 0:
            max_time = min(remaining - 3, 55)
        elif idx == 1:
            max_time = min(remaining - 3, 40)
        else:
            max_time = min(remaining - 3, 30)

        if max_time < 8:
            break

        try:
            result = _optimize_single(init, deadline)
            val = np.sum(result[:, 2])
            if val > best_val:
                best_val = val
                best_circles = result
        except Exception:
            continue

    if best_circles is None or not _validate(best_circles):
        # Final fallback
        best_circles = _initial_grid()
        best_circles = _fix_boundaries(best_circles)

    return best_circles

# EVOLVE-BLOCK-END
