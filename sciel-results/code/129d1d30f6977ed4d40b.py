"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _remaining_time():
    if _GLOBAL_DEADLINE is None:
        return 145.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps

def _objective(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
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
    """Objective with analytic gradient."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    f = 0.0
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # Boundary terms
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    for i in range(N):
        # lx_i = r_i - x_i
        s = _softplus(lx[i], smooth_eps)
        f += penalty_weight * s**2
        ds = np.exp(lx[i]/smooth_eps) / (1 + np.exp(lx[i]/smooth_eps)) if lx[i] > -50*smooth_eps else 0.0
        gx[i] += penalty_weight * 2*s*ds*(-1)
        gr[i] += penalty_weight * 2*s*ds*(1)
        
        # ux_i = x_i + r_i - 1
        s = _softplus(ux[i], smooth_eps)
        f += penalty_weight * s**2
        ds = np.exp(ux[i]/smooth_eps) / (1 + np.exp(ux[i]/smooth_eps)) if ux[i] > -50*smooth_eps else 0.0
        gx[i] += penalty_weight * 2*s*ds*(1)
        gr[i] += penalty_weight * 2*s*ds*(1)
        
        # ly_i = r_i - y_i
        s = _softplus(ly[i], smooth_eps)
        f += penalty_weight * s**2
        ds = np.exp(ly[i]/smooth_eps) / (1 + np.exp(ly[i]/smooth_eps)) if ly[i] > -50*smooth_eps else 0.0
        gy[i] += penalty_weight * 2*s*ds*(-1)
        gr[i] += penalty_weight * 2*s*ds*(1)
        
        # uy_i = y_i + r_i - 1
        s = _softplus(uy[i], smooth_eps)
        f += penalty_weight * s**2
        ds = np.exp(uy[i]/smooth_eps) / (1 + np.exp(uy[i]/smooth_eps)) if uy[i] > -50*smooth_eps else 0.0
        gy[i] += penalty_weight * 2*s*ds*(1)
        gr[i] += penalty_weight * 2*s*ds*(1)
        
        # -r_i term
        f -= r[i]
        gr[i] -= 1.0
        
        # -r_i penalty
        s = _softplus(-r[i], smooth_eps)
        f += penalty_weight * s**2
        ds = np.exp(-r[i]/smooth_eps) / (1 + np.exp(-r[i]/smooth_eps)) if -r[i] > -50*smooth_eps else 0.0
        gr[i] += penalty_weight * 2*s*ds*(-1)
    
    # Pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = max(dx*dx + dy*dy, 1e-12)
        dist = np.sqrt(dist_sq)
        overlap = r[i] + r[j] - dist
        s = _softplus(overlap, smooth_eps)
        f += penalty_weight * s**2
        if overlap > -50*smooth_eps:
            ds = np.exp(overlap/smooth_eps) / (1 + np.exp(overlap/smooth_eps))
            # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
            gx[i] += penalty_weight * 2*s*ds*(-dx/dist)
            gx[j] += penalty_weight * 2*s*ds*(dx/dist)
            gy[i] += penalty_weight * 2*s*ds*(-dy/dist)
            gy[j] += penalty_weight * 2*s*ds*(dy/dist)
            gr[i] += penalty_weight * 2*s*ds*(1)
            gr[j] += penalty_weight * 2*s*ds*(1)
    
    # Assemble gradient
    grad = np.empty(3*N)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    
    return f, grad

def _validate(xyz, tol=1e-7):
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _fix_boundaries(xyz):
    xyz = xyz.copy()
    for i in range(N):
        r = max(0.0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1.0-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1.0-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _compute_energy(xyz, penalty_weight=100.0):
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1)
    bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_pen = np.sum(overlap**2)
    neg_r_pen = np.sum(np.maximum(0, -r)**2)
    penalty = bound_pen + overlap_pen + neg_r_pen
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(xyz, tol=1e-4):
    points = xyz[:, :2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []

def _delaunay_objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    if contact_edges is not None and len(contact_edges) > 0:
        idx_i = np.array([e[0] for e in contact_edges])
        idx_j = np.array([e[1] for e in contact_edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _initial_hex():
    positions = []
    r = 0.09
    cols, rows = 5, 6
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5/cols
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
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
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

def _initial_corner_cluster():
    positions = []
    positions.append([0.11, 0.11, 0.13])
    positions.append([0.89, 0.11, 0.13])
    positions.append([0.11, 0.89, 0.13])
    positions.append([0.89, 0.89, 0.13])
    positions.append([0.5, 0.09, 0.11])
    positions.append([0.5, 0.91, 0.11])
    positions.append([0.09, 0.5, 0.11])
    positions.append([0.91, 0.5, 0.11])
    positions.append([0.5, 0.5, 0.13])
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.09)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _initial_optimized_hex():
    positions = []
    r = 0.095
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(6):
        for col in range(5):
            if row % 2 == 0:
                x = (col + 0.5) * dx + r
            else:
                x = (col + 0.5) * dx + r + dx/2
            y = (row + 0.5) * dy + r
            if x + r <= 1.0 and y + r <= 1.0:
                positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.7])
    return np.array(positions[:N])

def _initial_voronoi(seed=0):
    """
    Voronoi-based initialization: place circles at strategic locations
    and set initial radii based on local Voronoi cell size.
    """
    rng = np.random.default_rng(seed)
    # Start with corner and edge placements
    positions = np.array([
        [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
        [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5],
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.5],
    ])
    # Add random points
    while len(positions) < N:
        positions = np.vstack([positions, rng.uniform(0.05, 0.95, 2)])
    positions = positions[:N]
    
    # Compute Voronoi-based radii
    from scipy.spatial import Voronoi
    try:
        vor = Voronoi(positions)
        radii = np.zeros(N)
        for i in range(N):
            # Find region for point i
            region_idx = vor.point_region[i]
            if region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if len(region) > 0:
                    # Get vertices of the Voronoi cell
                    vertices = vor.vertices[region]
                    # Clip to unit square
                    vertices = vertices[(vertices[:, 0] >= 0) & (vertices[:, 0] <= 1) & 
                                        (vertices[:, 1] >= 0) & (vertices[:, 1] <= 1)]
                    if len(vertices) > 0:
                        # Minimum distance from center to any edge of the cell
                        min_dist = np.inf
                        for v in vertices:
                            d = np.linalg.norm(v - positions[i])
                            min_dist = min(min_dist, d)
                        # Also check boundaries
                        min_dist = min(min_dist, positions[i][0], 1-positions[i][0],
                                       positions[i][1], 1-positions[i][1])
                        radii[i] = 0.45 * min_dist
                    else:
                        radii[i] = 0.06
                else:
                    radii[i] = 0.06
            else:
                radii[i] = 0.06
        # Ensure minimum radius
        radii = np.maximum(radii, 0.04)
    except Exception:
        radii = np.full(N, 0.07)
    
    return np.column_stack([positions, radii])

def _pt_mc_search(initial_states, temps, max_steps, rng):
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s) for s in states]
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        i = rng.integers(0, n_replicas)
        T = temps[i]
        new_state = states[i].copy()
        
        move_type = rng.integers(0, 5)
        
        if move_type == 0:
            scale = 0.008 * T
            new_state[:, 0] += rng.normal(0, scale, N)
            new_state[:, 1] += rng.normal(0, scale, N)
        elif move_type == 1:
            idx = rng.integers(0, N)
            delta_r = rng.normal(0, 0.005 * T)
            new_state[idx, 2] = max(0.01, new_state[idx, 2] + delta_r)
        elif move_type == 2:
            x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlaps = r[idx_i] + r[idx_j] - dist
            overlap_mask = overlaps > 0
            if np.any(overlap_mask):
                overlap_indices = np.where(overlap_mask)[0]
                pick = overlap_indices[rng.integers(0, len(overlap_indices))]
                ii, jj = idx_i[pick], idx_j[pick]
                d = dist[pick]
                if d > 1e-9:
                    push = overlaps[pick] * 0.5 * rng.uniform(0.5, 1.5)
                    ux, uy = dx[pick]/d, dy[pick]/d
                    new_state[ii, 0] += ux * push
                    new_state[ii, 1] += uy * push
                    new_state[jj, 0] -= ux * push
                    new_state[jj, 1] -= uy * push
        elif move_type == 3:
            idx = rng.integers(0, N)
            x_i, y_i, r_i = new_state[idx, 0], new_state[idx, 1], new_state[idx, 2]
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            other_r = np.delete(new_state[:, 2], idx)
            other_x = np.delete(new_state[:, 0], idx)
            other_y = np.delete(new_state[:, 1], idx)
            d2 = (other_x - x_i)**2 + (other_y - y_i)**2
            d = np.sqrt(np.maximum(d2, 1e-12))
            max_r = min(max_r, np.min(d - other_r) if len(other_r) > 0 else max_r)
            max_r = max(0.01, max_r)
            if r_i < max_r:
                new_r = min(max_r, r_i + rng.uniform(0.001, 0.01) * T)
                new_state[idx, 2] = new_r
        else:
            # Gradient-based move: compute local gradient and step
            params = new_state.flatten()
            try:
                _, grad = _objective_grad(params, 500.0, 1e-4)
                # Take a small step in negative gradient direction
                lr = 0.0005 * T
                params -= lr * np.abs(grad)
                new_state = params.reshape(N, 3)
            except Exception:
                # Fallback to jitter
                new_state[:, 0] += rng.normal(0, 0.003 * T, N)
                new_state[:, 1] += rng.normal(0, 0.003 * T, N)
        
        new_state[:, 0] = np.clip(new_state[:, 0], -0.05, 1.05)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.05, 1.05)
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.005)
        
        new_energy = _compute_energy(new_state)
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            states[i] = new_state
            energies[i] = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()
    
    return best_state

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradients."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty, focus on layout expansion
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 12:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty with analytic gradient
    remaining = _remaining_time()
    if remaining > 10:
        penalty = 50.0
        try:
            res = minimize(_objective_grad, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          jac=True,
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            try:
                res = minimize(_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-4),
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < _objective(params, penalty, 1e-4):
                    params = res.x
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
            except Exception:
                pass
    
    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 8:
        penalty = 500.0
        try:
            res = minimize(_objective_grad, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          jac=True,
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            try:
                res = minimize(_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-5),
                              options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
                if res.fun < _objective(params, penalty, 1e-5):
                    params = res.x
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
            except Exception:
                pass
    
    # Phase 4: Delaunay-based optimization
    remaining = _remaining_time()
    if remaining > 6:
        penalty = 1000.0
        xyz_current = params.reshape(N, 3)
        contact_edges = _get_contact_edges(xyz_current)
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _delaunay_objective(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Very high penalty
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 5000.0
        try:
            res = minimize(_objective_grad, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          jac=True,
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            try:
                res = minimize(_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-6),
                              options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
                if res.fun < _objective(params, penalty, 1e-6):
                    params = res.x
                    best_params = params.copy()
                    best_score = np.sum(params[2::3])
            except Exception:
                pass
    
    # Phase 6: Ultra-high penalty
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def _delaunay_polish(xyz, max_iters=15):
    """Iterative Delaunay-based polish with analytic gradients."""
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    penalties = [2000.0, 5000.0, 10000.0, 20000.0, 50000.0]
    smooth_eps_schedule = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    
    for it in range(max_iters):
        if _remaining_time() < 3.0:
            break
        
        xyz_current = params.reshape(N, 3)
        contact_edges = _get_contact_edges(xyz_current)
        
        p_idx = min(it // 3, len(penalties) - 1)
        penalty = penalties[p_idx]
        smooth_eps = smooth_eps_schedule[p_idx]
        
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps, contact_edges),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30})
            
            if _validate(res.x.reshape(N, 3)):
                new_score = np.sum(res.x[2::3])
                if new_score > best_score:
                    params = res.x
                    best_params = params.copy()
                    best_score = new_score
                else:
                    params = res.x
        except Exception:
            try:
                res = minimize(_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, smooth_eps),
                              options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
                if _validate(res.x.reshape(N, 3)):
                    new_score = np.sum(res.x[2::3])
                    if new_score > best_score:
                        params = res.x
                        best_params = params.copy()
                        best_score = new_score
                    else:
                        params = res.x
            except Exception:
                break
    
    return best_params.reshape(N, 3), best_score

def _gradient_descent_polish(xyz, max_iters=1000, lr=0.001):
    """
    Direct gradient descent polish using analytic gradients.
    This is a structural alternative to L-BFGS-B for final refinement.
    """
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds_lo = np.array([0.0]*N + [0.0]*N + [0.0]*N)
    bounds_hi = np.array([1.0]*N + [1.0]*N + [0.5]*N)
    
    penalty = 10000.0
    smooth_eps = 1e-6
    
    for it in range(max_iters):
        if _remaining_time() < 2.0:
            break
        
        try:
            f, grad = _objective_grad(params, penalty, smooth_eps)
        except Exception:
            break
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Simple gradient descent with momentum-like decay
        step = lr * grad / (1.0 + 0.1 * grad_norm)
        new_params = params - step
        
        # Project to bounds
        new_params = np.clip(new_params, bounds_lo, bounds_hi)
        
        # Check if improvement
        try:
            new_f = _objective(new_params, penalty, smooth_eps)
        except Exception:
            break
        
        if new_f < f:
            params = new_params
            xyz_temp = params.reshape(N, 3)
            if _validate(xyz_temp):
                score = np.sum(xyz_temp[:, 2])
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        else:
            # Reduce learning rate
            lr *= 0.95
            if lr < 1e-8:
                break
    
    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations including Voronoi-based
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_optimized_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_corner_weighted(2),
        _initial_corner_cluster(),
        _initial_voronoi(0),
        _initial_voronoi(1),
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 15.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Local refinement from the best solution found
    remaining = _remaining_time()
    if remaining > 20.0:
        rng = np.random.default_rng(42)
        for scale in [0.003, 0.008, 0.015, 0.025, 0.04]:
            if remaining < 8.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 30.0:
        rng = np.random.default_rng(99)
        replicas = [best_xyz.copy()]
        for scale in [0.003, 0.008, 0.015, 0.025, 0.04]:
            perturbed = best_xyz.copy()
            perturbed += rng.normal(0, scale, perturbed.shape)
            perturbed = _fix_boundaries(perturbed)
            replicas.append(perturbed)
        
        temps = [0.3, 0.7, 1.5, 3.0, 6.0, 12.0]
        n_rep = min(len(replicas), len(temps))
        replicas = replicas[:n_rep]
        temps = temps[:n_rep]
        
        max_steps = int(remaining * 600)
        max_steps = max(15000, min(max_steps, 300000))
        
        pt_result = _pt_mc_search(replicas, temps, max_steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
    
    # Delaunay-based iterative polish
    remaining = _remaining_time()
    if remaining > 15.0 and _validate(best_xyz):
        polished, polished_score = _delaunay_polish(best_xyz, max_iters=15)
        if _validate(polished) and polished_score > best_score:
            best_xyz = polished
            best_score = polished_score
    
    # Gradient descent polish (structural alternative)
    remaining = _remaining_time()
    if remaining > 10.0 and _validate(best_xyz):
        gd_xyz, gd_score = _gradient_descent_polish(best_xyz, max_iters=500, lr=0.002)
        if _validate(gd_xyz) and gd_score > best_score:
            best_xyz = gd_xyz
            best_score = gd_score
    
    # Final high-precision multi-method polish pass
    remaining = _remaining_time()
    if remaining > 5.0 and _validate(best_xyz):
        params = best_xyz.flatten().copy()
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        polish_configs = [
            (100000.0, 1e-8, 500),
            (50000.0, 1e-9, 400),
            (10000.0, 1e-7, 300),
        ]
        
        for pen, eps, maxiter in polish_configs:
            remaining = _remaining_time()
            if remaining < 2.0:
                break
            try:
                res = minimize(_objective_grad, params, method='L-BFGS-B',
                              bounds=bounds, args=(pen, eps),
                              jac=True,
                              options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 30})
                if res.success and _validate(res.x.reshape(N, 3)):
                    polish_score = np.sum(res.x[2::3])
                    if polish_score > best_score:
                        best_xyz = res.x.reshape(N, 3)
                        best_score = polish_score
                        params = res.x.copy()
            except Exception:
                try:
                    res = minimize(_objective, params, method='L-BFGS-B',
                                  bounds=bounds, args=(pen, eps),
                                  options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 30})
                    if res.success and _validate(res.x.reshape(N, 3)):
                        polish_score = np.sum(res.x[2::3])
                        if polish_score > best_score:
                            best_xyz = res.x.reshape(N, 3)
                            best_score = polish_score
                            params = res.x.copy()
                except Exception:
                    pass
    
    return best_xyz

# EVOLVE-BLOCK-END
