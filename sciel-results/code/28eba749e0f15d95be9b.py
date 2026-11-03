import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

def _elapsed():
    if _GLOBAL_START is None:
        return 0.0
    return time.time() - _GLOBAL_START

def _remaining():
    if _GLOBAL_DEADLINE is None:
        return 145.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps

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

def _objective_penalty(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    # Boundary
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
    # Pairwise
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
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

def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
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

def _pt_mc_search(initial_states, temps, max_steps, rng):
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s) for s in states]
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if _remaining() < 25.0:
            break
            
        # Parallel Tempering / Replica Exchange
        # Pick two random replicas
        i, j = rng.choice(n_replicas, 2, replace=False)
        
        # Propose moves for both
        new_states = []
        new_energies = []
        for k in range(n_replicas):
            ns = states[k].copy()
            ci = rng.integers(N)
            ns[ci, 0] += rng.uniform(-0.05, 0.05)
            ns[ci, 1] += rng.uniform(-0.05, 0.05)
            ns[ci, 2] += rng.uniform(-0.02, 0.02)
            ns[ci, 0] = np.clip(ns[ci, 0], 0, 1)
            ns[ci, 1] = np.clip(ns[ci, 1], 0, 1)
            ns[ci, 2] = np.clip(ns[ci, 2], 0.01, 0.5)
            new_states.append(ns)
            new_energies.append(_compute_energy(ns))
            
        # Metropolis acceptance for each replica
        for k in range(n_replicas):
            delta_e = new_energies[k] - energies[k]
            if delta_e < 0 or rng.random() < np.exp(-delta_e / temps[k]):
                states[k] = new_states[k]
                energies[k] = new_energies[k]
                
        # Swap acceptance between i and j
        delta_swap = (energies[i] - energies[j]) * (1.0/temps[i] - 1.0/temps[j])
        if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
            states[i], states[j] = states[j], states[i]
            energies[i], energies[j] = energies[j], energies[i]
            
        # Update best
        for k in range(n_replicas):
            if energies[k] < best_energy:
                best_energy = energies[k]
                best_state = states[k].copy()
                
    return best_state

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # 1. Generate multiple initial configurations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(0),
        _initial_corner_cluster(),
        _initial_optimized_hex()
    ]
    
    # 2. Parallel Tempering / Replica Exchange Search
    temps = [1.0, 0.5, 0.25, 0.125, 0.0625]
    max_steps = 2000
    best_state = _pt_mc_search(inits, temps, max_steps, rng)
    
    # 3. Refinement with SLSQP using Delaunay contact edges
    # Try multiple refinement passes with different penalty weights
    
    best_refined = best_state
    best_refined_energy = _compute_energy(best_state)
    
    for pw in [100.0, 500.0, 2000.0]:
        if _remaining() < 15.0:
            break
            
        # Get contact edges from current best
        edges = _get_contact_edges(best_state)
        
        def obj(v):
            return _objective_delaunay(v, pw, 1e-4, edges)
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0.01, 0.5))
            
        try:
            res = minimize(obj, best_state, method='SLSQP', bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-6})
            if res.success:
                candidate = res.x
                if _validate(candidate):
                    energy = _compute_energy(candidate)
                    if energy < best_refined_energy:
                        best_refined = candidate
                        best_refined_energy = energy
        except Exception:
            pass
            
    # 4. Final validation and cleanup
    result = _fix_boundaries(best_refined)
    
    # Ensure strict feasibility
    for _ in range(10):
        valid = True
        x, y, r = result[:,0], result[:,1], result[:,2]
        for i in range(N):
            min_dist = min(x[i], 1-x[i], y[i], 1-y[i])
            if r[i] > min_dist:
                r[i] = min_dist * 0.99
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    total_r = r[i] + r[j]
                    if total_r > 0:
                        scale = (dist * 0.99) / total_r
                        r[i] *= scale
                        r[j] *= scale
                    valid = False
        result[:,0] = x
        result[:,1] = y
        result[:,2] = r
        if valid:
            break
            
    return result
