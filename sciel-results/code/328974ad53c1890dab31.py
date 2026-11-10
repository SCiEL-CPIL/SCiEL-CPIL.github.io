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
    """Return (i, j) pairs for all i < j."""
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _sqp_epigraph(points, max_iter=200, tol=1e-10):
    """
    Epigraph formulation: maximize t subject to:
    - ||p_i - p_j|| <= 1 for all pairs (normalize max dist to 1)
    - ||p_i - p_j|| >= t for all pairs
    Variables: x = [p_0, p_1, ..., p_{n-1}, t]
    """
    n = len(points)
    m = n * (n - 1) // 2
    pairs = _pair_indices(n)
    
    def objective(v):
        return -v[2*n]  # minimize -t = maximize t
    
    def jac_objective(v):
        grad = np.zeros(2*n + 1)
        grad[2*n] = -1.0
        return grad
    
    constraints = []
    
    # Max distance constraints: ||p_i - p_j||^2 <= 1
    for idx, (i, j) in enumerate(pairs):
        def cons(v, i=i, j=j):
            pi = v[2*i:2*i+2]
            pj = v[2*j:2*j+2]
            return 1.0 - np.sum((pi - pj)**2)
        constraints.append({'type': 'ineq', 'fun': cons})
    
    # Min distance constraints: ||p_i - p_j||^2 >= t^2
    for idx, (i, j) in enumerate(pairs):
        def cons(v, i=i, j=j):
            pi = v[2*i:2*i+2]
            pj = v[2*j:2*j+2]
            t = v[2*n]
            return np.sum((pi - pj)**2) - t**2
        constraints.append({'type': 'ineq', 'fun': cons})
    
    bounds = [(-5, 5)] * (2*n) + [(1e-6, 1.0)]
    v0 = np.append(points.flatten(), 0.1)
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac_objective,
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _fast_local_search(points, time_budget=30.0, seed=42):
    """
    Fast local search using coordinate descent and Nelder-Mead style moves.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Normalize
    d = pdist(best_points)
    d_max = d.max()
    if d_max > 1e-9:
        best_points = best_points / d_max
    
    best_score = _compute_score(best_points)
    start_time = time.time()
    
    step_size = 0.02
    improvement_count = 0
    
    while time.time() - start_time < time_budget:
        if step_size < 1e-8:
            break
        
        improved = False
        for _ in range(200):  # Try 200 random moves per round
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(0, n)
            delta = rng.randn(2) * step_size
            new_points = best_points.copy()
            new_points[idx] += delta
            
            new_score = _compute_score(new_points)
            if new_score > best_score + 1e-12:
                best_points = new_points
                best_score = new_score
                improved = True
                improvement_count += 1
        
        if not improved:
            step_size *= 0.7
        else:
            improvement_count = 0
    
    return best_points, best_score

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

    # 8. Three concentric rings: 1 center + 5 ring + 10 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(5) / 5
    ring1 = 0.7 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(10) / 10 + np.pi/10
    ring2 = 1.4 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 9. 2x8 with different aspect ratio
    xs = np.tile(np.array([-0.7, 0.7]), 8)
    ys = np.linspace(-1, 1, 16)
    rect2 = np.column_stack([xs, ys])
    candidates.append(rect2 + rng.randn(16, 2) * 0.03)

    return candidates

def _polish(points, time_budget=60.0):
    """High-precision polish pass."""
    n = len(points)
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Normalize
    d = pdist(best_points)
    d_max = d.max()
    if d_max > 1e-9:
        best_points = best_points / d_max
    best_score = _compute_score(best_points)
    
    # Round 1: SLSQP with moderate tolerance
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        result_x = _sqp_epigraph(best_points, max_iter=300, tol=1e-9)
        if result_x is not None and len(result_x) == 2*n + 1:
            new_points = result_x[:2*n].reshape(n, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    
    # Round 2: Fast local search
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        ls_points, ls_score = _fast_local_search(best_points, time_budget=min(remaining - 3.0, 20.0), seed=123)
        if ls_score > best_score:
            best_points = ls_points
            best_score = ls_score
    
    # Round 3: Final SLSQP polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 3.0:
        result_x = _sqp_epigraph(best_points, max_iter=500, tol=1e-12)
        if result_x is not None and len(result_x) == 2*n + 1:
            new_points = result_x[:2*n].reshape(n, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 600.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)
    
    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)
    
    # Phase 1: SLSQP optimization from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 90.0:
            break
        
        time_budget = min(25.0, deadline - time.time() - 60.0)
        if time_budget < 3.0:
            break
        
        try:
            # Normalize first
            d = pdist(init_pts)
            d_max = d.max()
            if d_max > 1e-9:
                normalized = init_pts / d_max
            else:
                normalized = init_pts.copy()
            
            result_x = _sqp_epigraph(normalized, max_iter=150, tol=1e-8)
            if result_x is not None and len(result_x) == 2*n + 1:
                new_points = result_x[:2*n].reshape(n, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
        except Exception as e:
            pass

    # Phase 2: Fast local search from best
    remaining = deadline - time.time() - 30.0
    if remaining > 10.0:
        try:
            ls_points, ls_score = _fast_local_search(best_points, time_budget=min(remaining - 5.0, 40.0), seed=99)
            if ls_score > best_score:
                best_points = ls_points
                best_score = ls_score
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            polished_points, polished_score = _polish(best_points, time_budget=min(remaining - 3.0, 80.0))
            if polished_score > best_score:
                best_points = polished_points
                best_score = polished_score
        except Exception:
            pass

    return best_points
