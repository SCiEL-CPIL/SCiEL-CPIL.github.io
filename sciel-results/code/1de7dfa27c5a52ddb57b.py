import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
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

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
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

def neg_score(points_flat):
    """
    Objective function for minimizers.
    """
    points = points_flat.reshape(-1, 2)
    return -compute_score(points)

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_config(n1, n2, r1, r2, rot1, rot2):
    """
    Generate points on two concentric circles.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _generate_3ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    """
    Generate points on three concentric circles.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        sc = compute_score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
        return sc

    def _time_left():
        return deadline - time.time()

    # Phase 1: Quick local optimization of ring configurations
    # This provides a strong initial guess and baseline score.
    
    # Try 2-ring splits
    splits_2 = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5)]
    
    for n1, n2 in splits_2:
        if _time_left() < 10:
            break
        # Optimize ring parameters: r2/r1, rot1, rot2
        # Fix r1=1, optimize r2, rot1, rot2
        def _neg_ring_score_2(params):
            r2, rot1, rot2 = params
            if r2 < 0.1 or r2 > 1.0:
                return 1.0
            pts = _generate_ring_config(n1, n2, 1.0, r2, rot1, rot2)
            return -compute_score(pts)
        
        # Try a few initial guesses
        for r_init in [0.5, 0.7, 0.9]:
            for rot_init in [0.0, np.pi/n1]:
                if _time_left() < 5:
                    break
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _neg_ring_score_2,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    r2, rot1, rot2 = res.x
                    pts = _generate_ring_config(n1, n2, 1.0, r2, rot1, rot2)
                    _update_best(pts)
                except Exception:
                    pass

    # Try 3-ring splits (e.g., 5-4-5, 6-4-4)
    splits_3 = [(5, 4, 5), (6, 4, 4), (4, 5, 5)]
    for n1, n2, n3 in splits_3:
        if _time_left() < 10:
            break
        def _neg_ring_score_3(params):
            r2, r3, rot1, rot2, rot3 = params
            if r2 < 0.1 or r2 > 1.0 or r3 < 0.1 or r3 > 1.0:
                return 1.0
            pts = _generate_3ring_config(n1, n2, n3, 1.0, r2, r3, rot1, rot2, rot3)
            return -compute_score(pts)
        
        for r2_init, r3_init in [(0.5, 0.8), (0.6, 0.9)]:
            if _time_left() < 5:
                break
            x0 = np.array([r2_init, r3_init, 0.0, 0.0, 0.0])
            try:
                res = scipy.optimize.minimize(
                    _neg_ring_score_3,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                r2, r3, rot1, rot2, rot3 = res.x
                pts = _generate_3ring_config(n1, n2, n3, 1.0, r2, r3, rot1, rot2, rot3)
                _update_best(pts)
            except Exception:
                pass

    # Phase 2: Global Search with Differential Evolution
    # We search in the full 28-dimensional space.
    # Bounds: [-1, 1] for x and y. The hull area will normalize the score.
    bounds = [(-1, 1)] * (2 * N)
    
    # Create initial population from best ring configs + random
    # DE can use 'initial' as a sequence of points
    init_population = []
    
    # Add best ring config if found
    if best_points is not None:
        # Normalize to fit in [-1, 1] roughly
        max_coord = np.max(np.abs(best_points))
        if max_coord > 0:
            init_population.append(best_points.flatten() / max_coord)
    
    # Add a few random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        init_population.append(rng.uniform(-1, 1, 2 * N))
    
    # Ensure population size is at least popsize * len(bounds)
    # popsize=15, so need 15 * 28 = 420 points? No, popsize is number of individuals.
    # len(bounds) = 28. popsize=15 means 15 individuals.
    # We have 6 individuals. DE will fill the rest with random.
    
    if _time_left() > 60:
        try:
            # DE is slow, so limit generations.
            # maxiter=50, popsize=15
            res = scipy.optimize.differential_evolution(
                neg_score,
                bounds,
                initial=init_population,
                popsize=15,
                maxiter=50,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=False, # We will polish manually to save time
                callback=None
            )
            if res.success or res.x is not None:
                candidate = res.x.reshape(-1, 2)
                _update_best(candidate)
        except Exception:
            pass

    # Phase 3: Local Polish with Nelder-Mead
    # Refine the best point found so far.
    if best_points is not None and _time_left() > 20:
        # Try a few different NM runs with slight perturbations to escape local minima
        for i in range(5):
            if _time_left() < 5:
                break
            if i == 0:
                start_pt = best_points.flatten()
            else:
                noise = rng.uniform(-0.05, 0.05, 2 * N)
                start_pt = best_points.flatten() + noise
            
            try:
                res = scipy.optimize.minimize(
                    neg_score,
                    start_pt,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                candidate = res.x.reshape(-1, 2)
                _update_best(candidate)
            except Exception:
                pass

    # Final check
    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon(N, 1.0)
        best_score = compute_score(best_points)
    
    return best_points
