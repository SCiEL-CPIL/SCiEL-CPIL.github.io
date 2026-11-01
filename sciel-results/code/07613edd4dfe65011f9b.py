import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _triangle_areas(points):
    """Compute all triangle areas (positive) for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute minimum triangle area for given points (assumes valid shape)."""
    return _triangle_areas(points).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _soft_min(x, beta=100.0):
    """Soft-min approximation using log-sum-exp."""
    x_min = x.min()
    shifted = x - x_min
    shifted = np.clip(shifted, 0, 50.0 / beta)
    return x_min - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _neg_smooth_score(points_flat, beta=100.0):
    """Negative smooth score for minimization."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 1e6
        areas = _triangle_areas(points)
        soft_min_area = _soft_min(areas, beta)
        return -soft_min_area / ha
    except Exception:
        return 1e6

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _coordinate_descent_polish(points, max_iter=3000, tol=1e-10, deadline=None):
    """
    Coordinate descent focusing on the bottleneck triangle.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        areas = _triangle_areas(best_pts)
        min_idx = np.argmin(areas)
        
        i, j, k = _TRI_INDICES[min_idx]
        bottleneck_vertices = [i, j, k]
        
        improved = False
        for vi in bottleneck_vertices:
            base = best_pts[vi].copy()
            best_local = base.copy()
            best_local_score = best_score
            
            # Adaptive step sizes based on current scale
            scale = np.linalg.norm(base) + 0.1
            for step in [0.1 * scale, 0.05 * scale, 0.02 * scale, 0.01 * scale, 0.005 * scale]:
                # Try 8 directions + 4 cardinal
                angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                for a in angles:
                    dx = step * np.cos(a)
                    dy = step * np.sin(a)
                    trial = best_pts.copy()
                    trial[vi] = base + np.array([dx, dy])
                    trial_score = _score(trial)
                    if trial_score > best_local_score + 1e-12:
                        best_local_score = trial_score
                        best_local = trial[vi].copy()
            
            if best_local_score > best_score + 1e-12:
                best_pts[vi] = best_local
                best_score = best_local_score
                improved = True
        
        if not improved:
            # Try larger steps
            for vi in bottleneck_vertices:
                base = best_pts[vi].copy()
                best_local = base.copy()
                best_local_score = best_score
                
                scale = np.linalg.norm(base) + 0.1
                for step in [0.2 * scale, 0.3 * scale, 0.5 * scale]:
                    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                    for a in angles:
                        dx = step * np.cos(a)
                        dy = step * np.sin(a)
                        trial = best_pts.copy()
                        trial[vi] = base + np.array([dx, dy])
                        trial_score = _score(trial)
                        if trial_score > best_local_score + 1e-12:
                            best_local_score = trial_score
                            best_local = trial[vi].copy()
                
                if best_local_score > best_score + 1e-12:
                    best_pts[vi] = best_local
                    best_score = best_local_score
                    improved = True
            
            if not improved:
                break
    
    return best_pts, best_score

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_config(n_outer=7, n_inner=7, ratio=0.5, outer_rot=0.0, inner_rot=0.0, seed=None):
    """
    Two-ring configuration: n_outer points on outer circle, n_inner on inner circle.
    This is the key insight - reducing boundary vertices from 14 to 7+inner improves
    consecutive triangle areas significantly.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        outer_rot += rng.uniform(0, 2 * np.pi)
        inner_rot += rng.uniform(0, 2 * np.pi)
    
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + outer_rot
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + inner_rot
    
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([ratio * np.cos(inner_angles), ratio * np.sin(inner_angles)])
    
    return np.vstack([outer, inner])

def _concentric_circles(n, seed=42):
    """Points on two concentric circles (7+7 split)."""
    return _two_ring_config(7, 7, 0.5, seed=seed)

def _random_in_disk(n, seed=None):
    """Random points in unit disk."""
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _perturbed_polygon(n, base_radius=1.0, noise_scale=0.05, seed=None, rotation=0.0):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, base_radius, rotation)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def _perturbed_two_ring(n_outer=7, n_inner=7, ratio=0.5, noise_scale=0.05, seed=None):
    """Perturbed two-ring configuration."""
    rng = np.random.RandomState(seed)
    base = _two_ring_config(n_outer, n_inner, ratio, seed=seed)
    noise = rng.uniform(-noise_scale, noise_scale, (N, 2))
    return base + noise

def _simulated_annealing(points, start_temp=0.1, end_temp=1e-4, max_iter=20000, deadline=None, rng=None):
    """
    Simulated Annealing to maximize the score.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    current_pts = points.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Precompute for speed
    n_pts = len(current_pts)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Cooling schedule
        t = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Generate move: perturb a random point
        idx = rng.randint(n_pts)
        # Step size proportional to temperature and point scale
        scale = np.linalg.norm(current_pts[idx]) + 0.1
        step_size = t * 0.5 * scale
        delta = rng.normal(0, step_size, 2)
        
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_pts = new_pts.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            if t > 0:
                delta_e = current_score - new_score
                # If delta_e is large, exp(-delta_e/t) is small
                # We accept if rng.random() < exp(-delta_e/t)
                # To avoid underflow/overflow issues with score differences
                # Note: scores are positive.
                if delta_e < t * 10: # Only compute exp if likely to accept
                    prob = np.exp(-delta_e / t)
                    if rng.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                        if current_score > best_score:
                            best_pts = new_pts.copy()
                            best_score = current_score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Two-ring configurations
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for outer_rot in [0.0, np.pi / 7, np.pi / 14]:
            for inner_rot in [0.0, np.pi / 7, np.pi / 14]:
                inits.append(_two_ring_config(7, 7, ratio, outer_rot, inner_rot))
    
    # 2. Perturbed two-ring
    for seed in range(10):
        for ratio in [0.4, 0.5, 0.6]:
            inits.append(_perturbed_two_ring(7, 7, ratio, 0.05, seed=seed + 200))
    
    # 3. Regular polygon
    inits.append(_regular_polygon(N, 1.0, 0.0))
    
    # 4. Random
    for seed in range(5):
        inits.append(_random_in_disk(N, seed=seed))
        
    # Sort inits by score
    scored_inits = []
    for init in inits:
        try:
            init_norm = _normalize(init)
            sc = _score(init_norm)
            scored_inits.append((sc, init_norm))
        except Exception:
            continue
            
    scored_inits.sort(key=lambda x: -x[0])
    
    if scored_inits:
        best_init_score, best_init_pts = scored_inits[0]
        if best_init_score > best_score:
            best_score = best_init_score
            best_points = best_init_pts.copy()
            
    # Main Search Loop: Simulated Annealing + Polish
    # We will run SA from the top few initializations
    
    top_k = 5
    for i in range(min(top_k, len(scored_inits))):
        if time.time() > deadline - 10:
            break
            
        init_pts = scored_inits[i][1]
        
        # Run Simulated Annealing
        try:
            sa_pts, sa_score = _simulated_annealing(
                init_pts, 
                start_temp=0.2, 
                end_temp=1e-5, 
                max_iter=50000, 
                deadline=deadline - 5,
                rng=np.random.RandomState(i*100)
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()
        except Exception:
            pass
        
        # Polish the best found so far (either init or SA result)
        if best_points is not None:
            try:
                polished, pol_score = _coordinate_descent_polish(
                    best_points, 
                    max_iter=1000, 
                    deadline=deadline - 2
                )
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished.copy()
            except Exception:
                pass
    
    # If we have time, try a few more random restarts with SA
    rng_global = np.random.RandomState(999)
    while time.time() < deadline - 15:
        # Generate a random two-ring-like init
        ratio = rng_global.uniform(0.3, 0.7)
        seed = rng_global.randint(0, 10000)
        init = _perturbed_two_ring(7, 7, ratio, 0.1, seed=seed)
        init = _normalize(init)
        
        sa_pts, sa_score = _simulated_annealing(
            init, 
            start_temp=0.1, 
            end_temp=1e-5, 
            max_iter=20000, 
            deadline=deadline - 5,
            rng=rng_global
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
        # Quick polish
        if best_points is not None:
            polished, pol_score = _coordinate_descent_polish(
                best_points, 
                max_iter=500, 
                deadline=deadline - 2
            )
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
                
    # Final Polish
    if best_points is not None:
        polished, pol_score = _coordinate_descent_polish(
            best_points, 
            max_iter=2000, 
            deadline=deadline - 1
        )
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished.copy()
            
    if best_points is None:
        best_points = _normalize(_two_ring_config(7, 7, 0.5))
    else:
        best_points = _normalize(best_points)
        
    return best_points
