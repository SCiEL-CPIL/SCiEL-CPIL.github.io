"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145.0  # seconds, safety margin before 150s kill

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
    if np.any(r < -tol):
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
        
    for _iter in range(10):
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

def _objective_full(params, penalty_weight, smooth_eps=1e-4):
    """
    Full pairwise objective for final polish (most accurate but slower).
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

def _objective_knn(params, penalty_weight, smooth_eps=1e-4, k_neighbors=10):
    """
    Vectorized objective using k-NN for overlap checking (faster than full pairwise).
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
    
    # k-NN overlaps
    centers = np.column_stack([x, y])
    try:
        tree = cKDTree(centers)
        # Query k nearest neighbors for each point
        dists, indices = tree.query(centers, k=min(k_neighbors+1, N))
        # dists[i][0] is 0 (self), so skip index 0
        for i in range(N):
            for j_idx in range(1, min(k_neighbors+1, N)):
                j = indices[i, j_idx]
                if j <= i:
                    continue  # avoid double-counting
                overlap = r[i] + r[j] - dists[i, j_idx]
                penalty += _softplus(overlap, smooth_eps)**2
    except Exception:
        # Fallback to full pairwise
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
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

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
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
    """Grid initial config."""
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
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _compute_overlap_penalty(xyz):
    """Compute hard overlap penalty for a configuration."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    penalty = 0.0
    for i in range(N):
        penalty += max(0, r[i] - x[i])**2
        penalty += max(0, x[i] + r[i] - 1.0)**2
        penalty += max(0, r[i] - y[i])**2
        penalty += max(0, y[i] + r[i] - 1.0)**2
        penalty += max(0, -r[i])**2
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap**2
    return penalty

def _parallel_tempering(xyz0, deadline, n_replicas=6, seed=42):
    """
    Parallel tempering (replica exchange) Monte Carlo for circle packing.
    Simultaneously optimizes positions and radii.
    """
    rng = np.random.default_rng(seed)
    temps = np.array([50.0, 20.0, 8.0, 3.0, 1.0, 0.3])[:n_replicas]
    
    replicas = []
    for rep in range(n_replicas):
        if rep == 0:
            replicas.append(xyz0.copy())
        else:
            xyz_rep = xyz0.copy()
            xyz_rep[:, 0] += rng.normal(0, 0.03, N)
            xyz_rep[:, 1] += rng.normal(0, 0.03, N)
            xyz_rep[:, 2] *= rng.uniform(0.8, 1.2, N)
            xyz_rep = _fix_boundaries(xyz_rep)
            replicas.append(xyz_rep)
    
    best_xyz = replicas[0].copy()
    best_score = np.sum(replicas[0][:, 2])
    
    n_steps = 5000
    swap_interval = 20
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        cooling_factor = 1.0 - 0.3 * (step / n_steps)
        current_temps = temps * cooling_factor
        
        for rep in range(n_replicas):
            xyz = replicas[rep]
            T = current_temps[rep]
            
            move_type = rng.random()
            
            if move_type < 0.4:
                i = rng.integers(N)
                dx = rng.normal(0, 0.02)
                dy = rng.normal(0, 0.02)
                new_x = np.clip(xyz[i, 0] + dx, 0, 1)
                new_y = np.clip(xyz[i, 1] + dy, 0, 1)
                
                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                xyz_new = xyz.copy()
                xyz_new[i, 0] = new_x
                xyz_new[i, 1] = new_y
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)
                
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new
            
            elif move_type < 0.7:
                i = rng.integers(N)
                dr = rng.normal(0, 0.01)
                new_r = np.clip(xyz[i, 2] + dr, 0.01, 0.45)
                
                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                xyz_new = xyz.copy()
                xyz_new[i, 2] = new_r
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)
                
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new
            
            else:
                i, j = rng.choice(N, size=2, replace=False)
                xyz_new = xyz.copy()
                xyz_new[[i, j], 0] = xyz[[j, i], 0] + rng.normal(0, 0.01, 2)
                xyz_new[[i, j], 1] = xyz[[j, i], 1] + rng.normal(0, 0.01, 2)
                xyz_new[:, 0] = np.clip(xyz_new[:, 0], 0, 1)
                xyz_new[:, 1] = np.clip(xyz_new[:, 1], 0, 1)
                
                old_energy = -np.sum(xyz[:, 2]) + 100.0 * _compute_overlap_penalty(xyz)
                new_energy = -np.sum(xyz_new[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_new)
                
                delta_e = new_energy - old_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    replicas[rep] = xyz_new
        
        if step % swap_interval == 0:
            rep_a = rng.integers(n_replicas - 1)
            rep_b = rep_a + 1
            
            xyz_a = replicas[rep_a]
            xyz_b = replicas[rep_b]
            T_a = current_temps[rep_a]
            T_b = current_temps[rep_b]
            
            E_a = -np.sum(xyz_a[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_a)
            E_b = -np.sum(xyz_b[:, 2]) + 100.0 * _compute_overlap_penalty(xyz_b)
            
            delta = (1.0/T_b - 1.0/T_a) * (E_b - E_a)
            if delta < 0 or rng.random() < np.exp(-delta):
                replicas[rep_a], replicas[rep_b] = replicas[rep_b], replicas[rep_a]
        
        cold_score = np.sum(replicas[0][:, 2])
        if cold_score > best_score and _validate(replicas[0]):
            best_score = cold_score
            best_xyz = replicas[0].copy()
        
        for rep in range(n_replicas):
            rep_score = np.sum(replicas[rep][:, 2])
            if rep_score > best_score and _validate(replicas[rep]):
                best_score = rep_score
                best_xyz = replicas[rep].copy()
    
    return best_xyz, best_score

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config: PT-MC then local polish."""
    best_params = xyz0.copy()
    best_score = np.sum(xyz0[:, 2])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 0: Parallel tempering MC (global search)
    remaining = deadline - time.time()
    if remaining > 20:
        pt_budget = deadline - 8.0
        try:
            xyz_pt, score_pt = _parallel_tempering(xyz0, pt_budget, n_replicas=5, seed=int(hash(tuple(xyz0[:3].round(4)))) % 10000)
            if score_pt > best_score:
                best_params = xyz_pt.copy()
                best_score = score_pt
        except Exception:
            pass
    
    # Phase 1: k-NN based local optimization
    params = best_params.flatten().copy()
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 10.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Delaunay-based optimization with higher penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 50.0
        contact_edges = _get_contact_edges(best_params)
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_delaunay(params, penalty, 1e-4, contact_edges):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Full pairwise with high penalty
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 200.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective_full(params, penalty, 1e-5):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Even higher penalty
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 1000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective_full(params, penalty, 1e-6):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 5000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 20})
            if res.fun < _objective_full(params, penalty, 1e-7):
                params = res.x
                if np.sum(params[2::3]) > best_score:
                    best_params = params.reshape(N, 3).copy()
                    best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    global_start = time.time()
    deadline = global_start + 143.0  # 7s safety margin before 150s kill
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_ring(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_ring(seed=999)
    ]
    
    # Optimize each initial config with a portion of the time budget
    best_xyz = None
    best_score = -np.inf
    
    n_configs = len(initial_configs)
    time_per_config = (deadline - global_start) / (n_configs + 1)  # +1 for final polish
    
    for i, xyz0 in enumerate(initial_configs):
        if time.time() > deadline - 15.0:
            break
        
        # Allocate time for this config
        config_deadline = min(deadline - 10.0, global_start + (i + 1) * time_per_config)
        
        try:
            xyz_opt, score = _optimize_single(xyz0, config_deadline)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(100):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.95
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if _validate(xyz_opt):
                score = np.sum(xyz_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz_opt.copy()
        except Exception:
            pass
    
    # Final polish on the best result
    if best_xyz is not None and _remaining() > 5.0:
        try:
            # Do one more round of optimization on the best result
            final_deadline = deadline - 2.0
            xyz_final, score_final = _optimize_single(best_xyz, final_deadline)
            
            if not _validate(xyz_final):
                xyz_fix = xyz_final.copy()
                for _ in range(100):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.95
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_final = xyz_fix
            
            if _validate(xyz_final):
                score_final = np.sum(xyz_final[:, 2])
                if score_final > best_score:
                    best_score = score_final
                    best_xyz = xyz_final.copy()
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_xyz is None or not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
