"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(0, z)."""
    return np.logaddexp(0, z / eps) * eps

def _validate(xyz, tol=1e-7):
    """Check validity of (x, y, r) array."""
    n = xyz.shape[0]
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _fix_boundaries(xyz):
    """Clip positions to be within [r, 1-r] for each circle."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0.0, xyz[i, 2])
        xyz[i, 0] = np.clip(xyz[i, 0], r, 1.0 - r) if r < 0.5 else 0.5
        xyz[i, 1] = np.clip(xyz[i, 1], r, 1.0 - r) if r < 0.5 else 0.5
        xyz[i, 2] = r
    return xyz

def _objective_penalty(params, penalty_weight=100.0, smooth_eps=1e-4):
    """Penalty-based objective: minimize -sum(r) + penalty for violations."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    # Boundary violations
    penalty += np.sum(_softplus(r - x, smooth_eps) ** 2)
    penalty += np.sum(_softplus(x + r - 1.0, smooth_eps) ** 2)
    penalty += np.sum(_softplus(r - y, smooth_eps) ** 2)
    penalty += np.sum(_softplus(y + r - 1.0, smooth_eps) ** 2)
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)
    # Pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)
    
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(xyz, tol=1e-4):
    """Get Delaunay edges as candidate contact pairs."""
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
        return []

def _objective_delaunay(params, penalty_weight=100.0, smooth_eps=1e-4, contact_edges=None):
    """Penalty objective using only Delaunay edges for overlaps (faster)."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    penalty += np.sum(_softplus(r - x, smooth_eps) ** 2)
    penalty += np.sum(_softplus(x + r - 1.0, smooth_eps) ** 2)
    penalty += np.sum(_softplus(r - y, smooth_eps) ** 2)
    penalty += np.sum(_softplus(y + r - 1.0, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)
    
    if contact_edges is not None and len(contact_edges) > 0:
        idx_i = np.array([e[0] for e in contact_edges])
        idx_j = np.array([e[1] for e in contact_edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)
    
    return -np.sum(r) + penalty_weight * penalty

def _compute_energy(xyz, penalty_weight=100.0):
    """Compute energy for a state (for parallel tempering)."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1)
    bound_pen = np.sum(lx ** 2 + ux ** 2 + ly ** 2 + uy ** 2)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_pen = np.sum(overlap ** 2)
    neg_r_pen = np.sum(np.maximum(0, -r) ** 2)
    penalty = bound_pen + overlap_pen + neg_r_pen
    return -np.sum(r) + penalty_weight * penalty

def _initial_grid():
    """5x5 grid initial configuration."""
    side = 5
    positions = []
    r = 0.36 / side
    for i in range(N):
        row = i // side
        col = i % side
        positions.append([(col + 0.5) / side, (row + 0.5) / side, r])
    return np.array(positions[:N])

def _initial_hex():
    """Hexagonal packing initial configuration."""
    positions = []
    r = 0.09
    cols, rows = 5, 6
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1.0 or y > 1.0:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_corner_cluster():
    """Corner-weighted initial configuration."""
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

def _initial_perturbed_grid(seed):
    """Perturbed grid initial configuration."""
    rng = np.random.default_rng(seed)
    side = 5
    positions = []
    r = 0.33 / side
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side + rng.normal(0, 0.02)
        y = (row + 0.5) / side + rng.normal(0, 0.02)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _initial_random(seed):
    """Random initial configuration."""
    rng = np.random.default_rng(seed)
    positions = []
    for i in range(N):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.02, 0.08)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _slsqp_optimize(xyz, penalty_weight=100.0, maxiter=80, ftol=1e-10):
    """SLSQP optimization with penalty objective."""
    params = xyz.flatten()
    
    def objective(p):
        return _objective_penalty(p, penalty_weight)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    try:
        res = minimize(objective, params, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < objective(params):
            new_xyz = res.x.reshape(N, 3)
            new_xyz = _fix_boundaries(new_xyz)
            if _validate(new_xyz):
                return new_xyz
    except Exception:
        pass
    return xyz

def _slsqp_delaunay(xyz, penalty_weight=100.0, maxiter=60, ftol=1e-10):
    """SLSQP optimization using Delaunay edges for overlap constraints."""
    params = xyz.flatten()
    edges = _get_contact_edges(xyz)
    
    def objective(p):
        return _objective_delaunay(p, penalty_weight, contact_edges=edges)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
    
    try:
        res = minimize(objective, params, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < objective(params):
            new_xyz = res.x.reshape(N, 3)
            new_xyz = _fix_boundaries(new_xyz)
            if _validate(new_xyz):
                return new_xyz
    except Exception:
        pass
    return xyz

def _grow_radii(xyz):
    """Grow each circle's radius to maximum possible given current positions."""
    n = xyz.shape[0]
    new_xyz = xyz.copy()
    for i in range(n):
        dmin = min(xyz[i, 0], 1.0 - xyz[i, 0], xyz[i, 1], 1.0 - xyz[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(xyz[i, 0] - xyz[j, 0], xyz[i, 1] - xyz[j, 1])
                dmin = min(dmin, d - xyz[j, 2])
        new_xyz[i, 2] = max(0.0, dmin)
    return new_xyz

def _parallel_tempering(initial_xyz, t_start, deadline, n_replicas=4, max_steps=2000):
    """Parallel tempering with multiple temperature levels."""
    rng = np.random.default_rng(777)
    temps = [0.03, 0.06, 0.12, 0.25]
    if n_replicas < len(temps):
        temps = temps[:n_replicas]
    
    states = [initial_xyz.copy() for _ in range(n_replicas)]
    energies = [_compute_energy(s) for s in states]
    
    best_xyz = states[0].copy()
    best_energy = energies[0]
    best_score = np.sum(states[0, 2])
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if time.time() > t_start + deadline:
            break
        
        # Propose moves in all replicas
        new_states = []
        new_energies = []
        for rep in range(n_replicas):
            s = states[rep].copy()
            T = temps[rep]
            
            # Random move
            if rng.random() < 0.6:
                i = rng.integers(0, N)
                s[i, 0] += rng.normal(0, T * 0.4)
                s[i, 1] += rng.normal(0, T * 0.4)
                s[i, 0] = np.clip(s[i, 0], 0.0, 1.0)
                s[i, 1] = np.clip(s[i, 1], 0.0, 1.0)
            else:
                i = rng.integers(0, N)
                s[i, 2] = np.clip(s[i, 2] + rng.normal(0, T * 0.25), 0.0, 0.4)
            
            new_states.append(s)
            new_energies.append(_compute_energy(s))
        
        # Accept/reject in each replica
        for rep in range(n_replicas):
            T = temps[rep]
            delta = new_energies[rep] - energies[rep]
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                states[rep] = new_states[rep]
                energies[rep] = new_energies[rep]
            
            # Track best
            if energies[rep] < best_energy:
                best_energy = energies[rep]
                best_xyz = states[rep].copy()
                score = np.sum(states[rep, 2])
                if _validate(states[rep]) and score > best_score:
                    best_score = score
                    best_xyz = states[rep].copy()
        
        # Swap replicas (Metropolis criterion)
        for rep in range(n_replicas - 1):
            if rng.random() < 1.0:
                # Try swapping rep and rep+1
                dE = (energies[rep] / temps[rep + 1] - energies[rep + 1] / temps[rep] -
                      energies[rep + 1] / temps[rep + 1] + energies[rep] / temps[rep])
                # Standard parallel tempering swap criterion
                dE = (energies[rep] - energies[rep + 1]) * (1.0 / temps[rep] - 1.0 / temps[rep + 1])
                if dE < 0 or rng.random() < np.exp(-dE):
                    states[rep], states[rep + 1] = states[rep + 1], states[rep]
                    energies[rep], energies[rep + 1] = energies[rep + 1], energies[rep]
    
    return best_xyz

def _refine_phase(xyz, t_start, deadline):
    """Iterative refinement: grow radii, then optimize positions."""
    best_xyz = xyz.copy()
    best_score = np.sum(xyz[:, 2])
    
    for iteration in range(30):
        if time.time() > t_start + deadline:
            break
        
        # Grow radii
        xyz = _grow_radii(xyz)
        
        # Optimize with SLSQP
        xyz_new = _slsqp_optimize(xyz, penalty_weight=80.0, maxiter=50)
        
        # Also try Delaunay-based
        xyz_del = _slsqp_delaunay(xyz, penalty_weight=80.0, maxiter=40)
        
        # Pick best
        candidates = [xyz, xyz_new, xyz_del]
        best_cand = candidates[0]
        best_cand_score = -np.inf
        for c in candidates:
            if _validate(c):
                s = np.sum(c[:, 2])
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c
        
        if best_cand_score > best_score:
            best_score = best_cand_score
            best_xyz = best_cand.copy()
        
        xyz = best_cand.copy()
    
    return best_xyz

def construct():
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    best_score = -1.0
    best_result = None
    
    # Multiple initial configurations
    initial_configs = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_cluster(),
        _initial_perturbed_grid(100),
        _initial_perturbed_grid(200),
        _initial_random(300),
    ]
    
    for restart, init_xyz in enumerate(initial_configs):
        if time.time() > t_start + DEADLINE - 20.0:
            break
        
        # Ensure valid start
        if not _validate(init_xyz):
            # Scale down radii until valid
            scale = 0.5
            while not _validate(init_xyz * np.array([1, 1, scale])) and scale > 0.01:
                scale *= 0.7
            init_xyz = init_xyz.copy()
            init_xyz[:, 2] *= scale
        
        # Phase 1: Parallel tempering to explore
        if time.time() < t_start + DEADLINE - 15.0:
            pt_xyz = _parallel_tempering(init_xyz, t_start, DEADLINE - 15.0, 
                                          n_replicas=3, max_steps=1500)
        else:
            pt_xyz = init_xyz.copy()
        
        # Phase 2: Refinement
        refined_xyz = _refine_phase(pt_xyz, t_start, DEADLINE - 5.0)
        
        # Phase 3: Final polish
        polished_xyz = _slsqp_optimize(refined_xyz, penalty_weight=50.0, maxiter=60)
        if _validate(polished_xyz) and np.sum(polished_xyz[:, 2]) > np.sum(refined_xyz[:, 2]):
            refined_xyz = polished_xyz
        
        # Track best
        if _validate(refined_xyz):
            score = np.sum(refined_xyz[:, 2])
            if score > best_score:
                best_score = score
                best_result = refined_xyz.copy()
        
        if best_score > 2.95:
            break
    
    # Fallback
    if best_result is None:
        best_result = _initial_grid()
        # Make sure it's valid
        if not _validate(best_result):
            scale = 0.5
            while not _validate(best_result * np.array([1, 1, scale])) and scale > 0.01:
                scale *= 0.7
            best_result = best_result.copy()
            best_result[:, 2] *= scale
    
    return best_result

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
