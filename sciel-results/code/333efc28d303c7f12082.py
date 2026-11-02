import time
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for triangles to speed up min area calculation
# There are C(14, 3) = 364 triangles
_TRI_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRI_IDX.append((i, j, k))
_TRI_IDX = np.array(_TRI_IDX)

def _min_triangle_area(points):
    """
    Compute the minimum area over all triples of points.
    points: (N, 2) array
    """
    pi = points[_TRI_IDX[:, 0]]
    pj = points[_TRI_IDX[:, 1]]
    pk = points[_TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        if len(points) < 3:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the ratio: min triangle area / convex hull area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _objective(x):
    """
    Objective function for scipy optimizers.
    x: 1D array of length 2*N
    Returns negative score (to be minimized).
    """
    points = x.reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    return -s

def _generate_two_ring_config(r_out, r_in, rot_offset=0.0):
    """
    Generate a configuration with 7 points on an outer heptagon and 7 on an inner heptagon.
    """
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rot_offset
    angles_in = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7.0 + rot_offset
    
    x_out = r_out * np.cos(angles_out)
    y_out = r_out * np.sin(angles_out)
    x_in = r_in * np.cos(angles_in)
    y_in = r_in * np.sin(angles_in)
    
    # Interleave or just stack? The order doesn't matter for the set of points, 
    # but for the convex hull, it just needs the points.
    # The score function doesn't care about index order, only the set of points.
    points = np.vstack([
        np.column_stack([x_out, y_out]),
        np.column_stack([x_in, y_in])
    ])
    return points.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # List of initial guesses
    initial_guesses = []
    
    # 1. Regular 14-gon
    angles_14 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg14 = np.column_stack([np.cos(angles_14), np.sin(angles_14)])
    initial_guesses.append(p_reg14.flatten())
    
    # 2. Two-ring configurations (Heptagons)
    # Based on inspiration, r_in/r_out ratio around 0.4-0.6 is good
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    for ratio in ratios:
        p_ring = _generate_two_ring_config(1.0, ratio)
        initial_guesses.append(p_ring)
        
    # 3. Slightly perturbed two-ring
    for ratio in [0.4, 0.5, 0.6]:
        p_ring = _generate_two_ring_config(1.0, ratio)
        # Add small noise to break symmetry if needed, though regular is likely local min for this structure
        # Actually, regular is the symmetric optimum for this sub-manifold. 
        # We want to allow the optimizer to move away from this if it's better.
        initial_guesses.append(p_ring)

    # Evaluate initial guesses to find the best start
    best_start_x = None
    best_start_score = -1e10
    for guess in initial_guesses:
        s = _score(guess.reshape(-1, 2))
        if s > best_start_score:
            best_start_score = s
            best_start_x = guess

    best_x = best_start_x.copy()
    best_score = best_start_score

    # Helper to check time
    def time_check():
        return time.time() > deadline - 15.0  # Leave 15s for final polish

    # --- Phase 1: Basinhopping from the best initial guess ---
    # We use basinhopping to escape local minima in the 28D space.
    # The two-ring structure is a strong attractor, so starting there is key.
    try:
        bh_res = basinhopping(
            _objective,
            best_x,
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7}},
            niter=30,  # Reduced niter to ensure time safety
            stepsize=0.2,
            callback=lambda *args, **kwargs: time_check(),
            seed=42
        )
        if bh_res.fun < -best_score:
            best_x = bh_res.x
            best_score = _score(best_x.reshape(-1, 2))
    except Exception:
        pass

    # --- Phase 2: Multi-start Local Refinement ---
    # Try refining a few of the initial guesses directly with Powell or L-BFGS-B
    # This is useful if basinhopping got stuck in a bad basin.
    if time.time() < deadline - 20.0:
        # We will try the top 3 initial guesses
        scored_guesses = []
        for guess in initial_guesses:
            s = _score(guess.reshape(-1, 2))
            scored_guesses.append((s, guess))
        
        scored_guesses.sort(key=lambda x: x[0], reverse=True)
        
        for s, guess in scored_guesses[:3]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    _objective,
                    guess,
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = _score(best_x.reshape(-1, 2))
            except Exception:
                pass

    # --- Phase 3: Final High-Precision Polish ---
    if time.time() < deadline - 5.0:
        try:
            # Use L-BFGS-B for smooth convergence if possible, otherwise NM
            res = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < -best_score:
                best_x = res.x
                best_score = _score(best_x.reshape(-1, 2))
        except Exception:
            pass
        
        # One more NM pass for robustness
        if time.time() < deadline - 2.0:
            try:
                res = minimize(
                    _objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = _score(best_x.reshape(-1, 2))
            except Exception:
                pass

    return best_x.reshape(-1, 2)
