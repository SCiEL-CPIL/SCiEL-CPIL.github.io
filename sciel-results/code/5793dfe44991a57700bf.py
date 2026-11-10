import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so

def _compute_score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _pair_indices(n):
    """Precompute pair indices for vectorized distance computation."""
    i, j = np.triu_indices(n, k=1)
    return i, j

def _sqp_epigraph(points, max_iter=200, tol=1e-10):
    """
    Epigraph formulation: maximize r subject to:
      ||p_i - p_j||^2 <= R^2 for all pairs (R fixed to 1 after normalization)
      ||p_i - p_j||^2 >= r^2 for all pairs
    Variables: [x_1, ..., x_n, y_1, ..., y_n, r]
    Objective: -r
    """
    n = len(points)
    i_idx, j_idx = _pair_indices(n)
    npairs = len(i_idx)
    
    # Normalize so max distance ~ 1
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        pts = points / d_max
    else:
        pts = points.copy()
    
    r0 = pdist(pts).min() * 0.99
    x0 = np.concatenate([pts.flatten(), [r0]])
    
    def objective(v):
        return -v[2*n]
    
    def jac(v):
        g = np.zeros(2*n + 1)
        g[2*n] = -1.0
        return g
    
    # Constraint: all pairwise distances^2 <= 1 (since normalized)
    def max_dist_fun(v):
        pts_v = v[:2*n].reshape(n, 2)
        diffs = pts_v[i_idx] - pts_v[j_idx]
        d2 = np.sum(diffs**2, axis=1)
        return 1.0 - d2  # >= 0 means d2 <= 1
    
    # Constraint: all pairwise distances^2 >= r^2
    def min_dist_fun(v):
        pts_v = v[:2*n].reshape(n, 2)
        r = v[2*n]
        diffs = pts_v[i_idx] - pts_v[j_idx]
        d2 = np.sum(diffs**2, axis=1)
        return d2 - r**2  # >= 0 means d2 >= r^2
    
    constraints = [
        {'type': 'ineq', 'fun': max_dist_fun},
        {'type': 'ineq', 'fun': min_dist_fun}
    ]
    
    bounds = [(-5, 5)] * (2*n) + [(1e-8, 1.0)]
    
    try:
        res = so.minimize(objective, x0, method='SLSQP', jac=jac, bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success or res.fun < 0:
            return res.x[:2*n].reshape(n, 2), res.x[2*n]
        else:
            return pts, r0
    except Exception:
        return pts, r0

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)
    
    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)

    # 8. 3 concentric rings: 1 + 5 + 10
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(5) / 5
    ring1 = 0.5 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(10) / 10
    ring2 = 1.0 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 9. Square lattice with perturbation
    base = np.array([[0,0],[1,0],[2,0],[3,0],
                     [0,1],[1,1],[2,1],[3,1],
                     [0,2],[1,2],[2,2],[3,2],
                     [0,3],[1,3],[2,3],[3,3]], dtype=float)
    base = (base - 1.5) / 1.5  # center and scale
    candidates.append(base + rng.randn(16, 2) * 0.02)

    return candidates

def _fast_local_search(points, time_budget=50.0, seed=42):
    """
    Fast local search using coordinate-wise perturbations.
    Much faster than full SA since we only recompute affected distances.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        current_points = points / d_max
    else:
        current_points = points.copy()
    
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    si = 0
    
    while time.time() - start_time < time_budget:
        if si >= len(step_sizes):
            break
        
        step_size = step_sizes[si]
        improved = False
        
        for _ in range(500):
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(0, n)
            coord = rng.randint(0, 2)
            delta = rng.choice([-1, 1]) * step_size
            
            new_points = current_points.copy()
            new_points[idx, coord] += delta
            
            new_score = _compute_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        
        if not improved:
            si += 1
    
    return best_points, best_score

def _polish(points, max_iter=300, tol=1e-12):
    """High-precision polish pass with very tight tolerance."""
    try:
        result_pts, r = _sqp_epigraph(points, max_iter=max_iter, tol=tol)
        score = _compute_score(result_pts)
        orig_score = _compute_score(points)
        if score > orig_score:
            return result_pts, score
        return points, orig_score
    except Exception:
        return points, _compute_score(points)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 615.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)
    
    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)
    
    # Phase 1: SQP epigraph optimization from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 120.0:
            break
        
        try:
            result_pts, r = _sqp_epigraph(init_pts, max_iter=150, tol=1e-8)
            result_score = _compute_score(result_pts)
            if result_score > best_score:
                best_points = result_pts
                best_score = result_score
        except Exception:
            pass

    # Phase 2: Fast local search from the best point found
    remaining = deadline - time.time() - 60.0
    if remaining > 30.0:
        try:
            ls_points, ls_score = _fast_local_search(best_points, time_budget=min(remaining - 30.0, 60.0), seed=123)
            if ls_score > best_score:
                best_points = ls_points
                best_score = ls_score
        except Exception:
            pass

    # Phase 3: SQP refinement on the locally-searched result
    remaining = deadline - time.time() - 30.0
    if remaining > 15.0:
        try:
            result_pts, r = _sqp_epigraph(best_points, max_iter=200, tol=1e-10)
            result_score = _compute_score(result_pts)
            if result_score > best_score:
                best_points = result_pts
                best_score = result_score
        except Exception:
            pass

    # Phase 4: High-precision polish
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            polished_pts, polished_score = _polish(best_points, max_iter=300, tol=1e-12)
            if polished_score > best_score:
                best_points = polished_pts
                best_score = polished_score
        except Exception:
            pass

    return best_points
