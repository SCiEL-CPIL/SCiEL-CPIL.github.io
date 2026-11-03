"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _softplus_grad(z, eps=1e-4):
    """Gradient of softplus."""
    return 1.0 / (1.0 + np.exp(-z / eps))

def _remaining_time():
    if _GLOBAL_DEADLINE is None:
        return 145.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _validate(xyz, tol=1e-7):
    """Check validity of configuration."""
    n = xyz.shape[0]
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < -1e-9):
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

def _enforce_validity(xyz, max_iters=100):
    """Iteratively enforce validity by shrinking and projecting."""
    xyz = xyz.copy()
    for _ in range(max_iters):
        if _validate(xyz):
            break
        # Shrink radii slightly
        xyz[:, 2] *= 0.995
        # Project into square
        for i in range(N):
            r = max(0.0, xyz[i, 2])
            xyz[i, 0] = np.clip(xyz[i, 0], r, 1 - r) if r < 0.5 else 0.5
            xyz[i, 1] = np.clip(xyz[i, 1], r, 1 - r) if r < 0.5 else 0.5
    return xyz

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
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
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Full pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Gradient of the objective for N=26 circles.
    """
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    # softplus'(z) = sigmoid(z/eps)
    slx = _softplus_grad(lx, smooth_eps)
    sux = _softplus_grad(ux, smooth_eps)
    sly = _softplus_grad(ly, smooth_eps)
    suy = _softplus_grad(uy, smooth_eps)
    
    # d/dx of (softplus(r-x))^2 = 2*softplus(r-x)*sigmoid((r-x)/eps)*(-1)
    gx += -2 * _softplus(lx, smooth_eps) * slx
    gx += 2 * _softplus(ux, smooth_eps) * sux
    gy += -2 * _softplus(ly, smooth_eps) * sly
    gy += 2 * _softplus(uy, smooth_eps) * suy
    gr += 2 * _softplus(lx, smooth_eps) * slx
    gr += 2 * _softplus(ux, smooth_eps) * sux
    gr += 2 * _softplus(ly, smooth_eps) * sly
    gr += 2 * _softplus(uy, smooth_eps) * suy
    
    # Negative radius
    snr = _softplus_grad(-r, smooth_eps)
    gr += -2 * _softplus(-r, smooth_eps) * snr
    
    # Pairwise overlaps
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    so = _softplus_grad(overlap, smooth_eps)
    
    # d/dx_i of (softplus(r_i+r_j-dist))^2 = 2*softplus*so*d(dist)/dx_i
    # d(dist)/dx_i = dx_i/dist, d(dist)/dx_j = -dx_j/dist = dx_i/dist (since dx = x_i - x_j)
    # Actually d(dist)/dx_i = (x_i - x_j)/dist = dx/dist
    # d(dist)/dx_j = (x_j - x_i)/dist = -dx/dist
    # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
    
    coeff = 2 * _softplus(overlap, smooth_eps) * so
    gx[idx_i] += coeff * (-dx / dist)
    gx[idx_j] += coeff * (dx / dist)
    gy[idx_i] += coeff * (-dy / dist)
    gy[idx_j] += coeff * (dy / dist)
    gr[idx_i] += coeff
    gr[idx_j] += coeff
    
    # Objective gradient: -sum(r) + penalty_weight * penalty
    # d(-sum(r))/dr_i = -1
    gr -= 1.0
    
    gx *= penalty_weight
    gy *= penalty_weight
    gr *= penalty_weight
    
    grad = np.empty(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    return grad

def _polish_analytic(xyz, max_iter=200, tol=1e-12):
    """
    Analytic polish: iteratively grow each circle to its maximum possible radius
    given current positions, then re-project.
    """
    xyz = xyz.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            x_i, y_i, r_i = xyz[i, 0], xyz[i, 1], xyz[i, 2]
            # Max radius from boundaries
            max_r = min(x_i, 1 - x_i, y_i, 1 - y_i)
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x_i - xyz[j, 0]
                dy = y_i - xyz[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-12:
                    max_r = min(max_r, dist - xyz[j, 2])
            max_r = max(0.0, max_r)
            if max_r > r_i + tol:
                xyz[i, 2] = max_r
                improved = True
        if not improved:
            break
    return xyz

def _physics_relax(xyz, steps=500, dt=0.001):
    """
    Physics-based relaxation: treat overlaps as repulsive forces.
    """
    xyz = xyz.copy()
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for _ in range(steps):
        if _remaining_time() < 3.0:
            break
        x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 10.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 10.0
        
        # Overlap forces
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        mask = overlap > 0
        if np.any(mask):
            force = overlap[mask] * 5.0
            ux = dx[mask] / dist[mask]
            uy = dy[mask] / dist[mask]
            fx[idx_i[mask]] += ux * force
            fx[idx_j[mask]] -= ux * force
            fy[idx_i[mask]] += uy * force
            fy[idx_j[mask]] -= uy * force
        
        # Update positions
        xyz[:, 0] += fx * dt
        xyz[:, 1] += fy * dt
        
        # Project
        for i in range(N):
            r = max(0.0, xyz[i, 2])
            xyz[i, 0] = np.clip(xyz[i, 0], r, 1 - r) if r < 0.5 else 0.5
            xyz[i, 1] = np.clip(xyz[i, 1], r, 1 - r) if r < 0.5 else 0.5
    
    return xyz

def _delaunay_edges(xyz):
    """Get Delaunay edges for contact pairs."""
    points = xyz[:, :2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs within some distance
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xyz[i,0]-xyz[j,0])**2 + (xyz[i,1]-xyz[j,1])**2)
                if d < 0.3:
                    edges.append((i, j))
        return edges

def _delaunay_objective(params, penalty_weight, smooth_eps=1e-4, edges=None):
    """
    Objective using only Delaunay contact edges for efficiency.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Pairwise overlaps only for contact edges
    if edges is None:
        idx_i, idx_j = np.triu_indices(N, k=1)
    else:
        idx_i = np.array([e[0] for e in edges])
        idx_j = np.array([e[1] for e in edges])
    
    if len(idx_i) > 0:
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _delaunay_polish(xyz, max_iter=100):
    """Polish using Delaunay-restricted objective."""
    params = xyz.flatten().copy()
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    for pen in [100.0, 1000.0, 10000.0]:
        edges = _delaunay_edges(xyz)
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(pen, 1e-5, edges),
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _delaunay_objective(params, pen, 1e-5, edges):
                params = res.x
                xyz = params.reshape(N, 3)
        except Exception:
            pass
    
    return xyz

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

def _initial_corner_cluster():
    """
    Specific initialization placing larger circles in corners and edges.
    """
    positions = []
    # 4 corners
    positions.append([0.1, 0.1, 0.12])
    positions.append([0.9, 0.1, 0.12])
    positions.append([0.1, 0.9, 0.12])
    positions.append([0.9, 0.9, 0.12])
    
    # 4 edge midpoints
    positions.append([0.5, 0.08, 0.10])
    positions.append([0.5, 0.92, 0.10])
    positions.append([0.08, 0.5, 0.10])
    positions.append([0.92, 0.5, 0.10])
    
    # Center
    positions.append([0.5, 0.5, 0.12])
    
    # Fill remaining
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.08)
        positions.append([x, y, r])
    
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

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2 * np.pi * k / n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist * np.cos(angle)
            y = 0.5 + dist * np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.06])
    return np.array(positions[:N])

def _optimize_single(xyz0, use_grad=True):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    phases = [
        (5.0, 1e-3, 300),
        (50.0, 1e-4, 400),
        (500.0, 1e-5, 400),
        (5000.0, 1e-6, 300),
        (50000.0, 1e-7, 200),
    ]
    
    for penalty, eps, maxiter in phases:
        if _remaining_time() < 5.0:
            break
        try:
            if use_grad:
                res = minimize(_objective, params, jac=_objective_grad,
                              method='L-BFGS-B', bounds=bounds,
                              args=(penalty, eps),
                              options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
            else:
                res = minimize(_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, eps),
                              options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def _grow_and_relax(xyz, max_rounds=20):
    """
    Grow circles to max possible, then relax to resolve overlaps.
    """
    for _ in range(max_rounds):
        if _remaining_time() < 5.0:
            break
        # Grow each circle to max
        for i in range(N):
            x_i, y_i, r_i = xyz[i, 0], xyz[i, 1], xyz[i, 2]
            max_r = min(x_i, 1 - x_i, y_i, 1 - y_i)
            for j in range(N):
                if i == j:
                    continue
                dx = x_i - xyz[j, 0]
                dy = y_i - xyz[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-12:
                    max_r = min(max_r, dist - xyz[j, 2])
            max_r = max(0.0, max_r)
            xyz[i, 2] = max_r
        
        # Physics relax to resolve overlaps
        xyz = _physics_relax(xyz, steps=100, dt=0.001)
    
    return xyz

def _simulated_annealing(xyz, max_steps=50000, t_start=1.0, t_end=0.01):
    """
    Simulated annealing with structure-aware moves.
    """
    best_xyz = xyz.copy()
    best_score = np.sum(xyz[:, 2])
    current_xyz = xyz.copy()
    current_energy = -np.sum(xyz[:, 2])
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        T = t_start * (t_end / t_start) ** (step / max_steps)
        
        new_xyz = current_xyz.copy()
        move_type = rng.integers(0, 4)
        
        if move_type == 0:
            # Position jitter
            scale = 0.005 * T
            new_xyz[:, 0] += rng.normal(0, scale, N)
            new_xyz[:, 1] += rng.normal(0, scale, N)
        elif move_type == 1:
            # Radius change
            idx = rng.integers(0, N)
            delta_r = rng.normal(0, 0.003 * T)
            new_xyz[idx, 2] = max(0.01, new_xyz[idx, 2] + delta_r)
        elif move_type == 2:
            # Push apart overlapping
            x, y, r = new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2]
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist = np.sqrt(np.maximum(dx*dx + dy*dy, 1e-12))
            overlaps = r[idx_i] + r[idx_j] - dist
            mask = overlaps > 0
            if np.any(mask):
                pick = np.where(mask)[0][rng.integers(0, np.sum(mask))]
                ii, jj = idx_i[pick], idx_j[pick]
                d = dist[pick]
                if d > 1e-9:
                    push = overlaps[pick] * 0.5 * rng.uniform(0.5, 1.5)
                    ux, uy = dx[pick] / d, dy[pick] / d
                    new_xyz[ii, 0] += ux * push
                    new_xyz[ii, 1] += uy * push
                    new_xyz[jj, 0] -= ux * push
                    new_xyz[jj, 1] -= uy * push
        else:
            # Grow a circle
            idx = rng.integers(0, N)
            x_i, y_i, r_i = new_xyz[idx, 0], new_xyz[idx, 1], new_xyz[idx, 2]
            max_r = min(x_i, 1 - x_i, y_i, 1 - y_i)
            for j in range(N):
                if j == idx:
                    continue
                dx = x_i - new_xyz[j, 0]
                dy = y_i - new_xyz[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-12:
                    max_r = min(max_r, dist - new_xyz[j, 2])
            max_r = max(0.01, max_r)
            if r_i < max_r:
                new_xyz[idx, 2] = min(max_r, r_i + rng.uniform(0.001, 0.01) * T)
        
        # Clip
        new_xyz[:, 0] = np.clip(new_xyz[:, 0], -0.05, 1.05)
        new_xyz[:, 1] = np.clip(new_xyz[:, 1], -0.05, 1.05)
        new_xyz[:, 2] = np.maximum(new_xyz[:, 2], 0.005)
        
        # Energy
        new_energy = -np.sum(new_xyz[:, 2])
        
        # Metropolis
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current_xyz = new_xyz
            current_energy = new_energy
            if current_energy < -best_score:
                best_score = -current_energy
                best_xyz = current_xyz.copy()
    
    return best_xyz

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]
    
    # Optimize from each with gradient
    for xyz0 in inits:
        if _remaining_time() < 20.0:
            break
        xyz_opt, score = _optimize_single(xyz0, use_grad=True)
        
        if not _validate(xyz_opt):
            xyz_opt = _enforce_validity(xyz_opt)
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Grow-and-relax from best
    if _remaining_time() > 15.0:
        grown = _grow_and_relax(best_xyz.copy(), max_rounds=10)
        if _validate(grown):
            grown_score = np.sum(grown[:, 2])
            if grown_score > best_score:
                best_score = grown_score
                best_xyz = grown
    
    # Local refinement from best solution
    if _remaining_time() > 15.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02]:
            if _remaining_time() < 8.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _enforce_validity(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed, use_grad=True)
            if not _validate(xyz_opt):
                xyz_opt = _enforce_validity(xyz_opt)
            if not _validate(xyz_opt):
                continue
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
    
    # Simulated annealing refinement
    if _remaining_time() > 20.0:
        sa_result = _simulated_annealing(best_xyz, max_steps=30000)
        if _validate(sa_result):
            sa_score = np.sum(sa_result[:, 2])
            if sa_score > best_score:
                best_score = sa_score
                best_xyz = sa_result
    
    # Delaunay polish
    if _remaining_time() > 10.0:
        del_result = _delaunay_polish(best_xyz)
        if _validate(del_result):
            del_score = np.sum(del_result[:, 2])
            if del_score > best_score:
                best_score = del_score
                best_xyz = del_result
    
    # Final high-precision polish
    if _remaining_time() > 3.0 and _validate(best_xyz):
        params = best_xyz.flatten().copy()
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N
        try:
            res = minimize(_objective, params, jac=_objective_grad,
                          method='L-BFGS-B', bounds=bounds,
                          args=(100000.0, 1e-8),
                          options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 20})
            if res.success and _validate(res.x.reshape(N, 3)):
                polish_score = np.sum(res.x[2::3])
                if polish_score > best_score:
                    best_xyz = res.x.reshape(N, 3)
                    best_score = polish_score
        except Exception:
            pass
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz)
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
