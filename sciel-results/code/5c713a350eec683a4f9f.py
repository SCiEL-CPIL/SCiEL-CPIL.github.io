# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _make_icosahedron_plus_2():
    """Icosahedron (12 verts) + 2 points at poles of a face."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1, s2 * phi])
            verts.append([s1, s2 * phi, 0])
            verts.append([s2 * phi, 0, s1])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_cuboctahedron_plus_2():
    """Cuboctahedron (12 verts) + 2 points."""
    verts = []
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_random_unit(n=14, d=3, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    points = rng.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def _coordinate_descent_polish(points, deadline, max_rounds=500):
    """
    Coordinate descent with adaptive step sizes on the Cartesian coordinates.
    This is the structural idea from inspiration 1: a robust local refiner that
    doesn't rely on smooth gradients and can handle the non-smooth min/max objective.
    """
    n, d = points.shape
    points = points.copy()
    best_score = _score(points)
    step_sizes = np.full((n, d), 0.1)
    
    for _round in range(max_rounds):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(d):
                if time.time() > deadline:
                    break
                    
                cur_score = _score(points)
                
                # Try positive step
                pts_plus = points.copy()
                pts_plus[i, j] += step_sizes[i, j]
                score_plus = _score(pts_plus)
                
                # Try negative step
                pts_minus = points.copy()
                pts_minus[i, j] -= step_sizes[i, j]
                score_minus = _score(pts_minus)
                
                if score_plus > cur_score:
                    points = pts_plus
                    improved = True
                    step_sizes[i, j] *= 1.1  # grow step slightly
                elif score_minus > cur_score:
                    points = pts_minus
                    improved = True
                    step_sizes[i, j] *= 1.1
                else:
                    step_sizes[i, j] *= 0.7  # shrink step
                    
                    # If step is too small, try a random kick
                    if step_sizes[i, j] < 1e-5:
                        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
                        points[i, j] += rng.randn() * 0.05
                        step_sizes[i, j] = 0.05
                        
        if not improved:
            # Try a global random perturbation to escape local minima
            if _round > 20:
                rng = np.random.RandomState(int(time.time() * 1000) % 10000)
                perturbed = points + rng.randn(n, d) * 0.1
                if _score(perturbed) > best_score:
                    points = perturbed
                else:
                    # No improvement after many rounds, break
                    if _round > 50:
                        break
                    
    # Update best if we found something better
    final_score = _score(points)
    if final_score > best_score:
        best_score = final_score
        
    return points

def _polish_slsqp_sphere(points, deadline):
    """
    SLSQP polish with sphere constraint for each point.
    This is the refinement machinery from the knowledge: SLSQP on sphere constraints.
    """
    n, d = points.shape
    x0 = points.flatten()
    
    # Objective to minimize: -score
    def obj(x):
        pts = x.reshape(n, d)
        return -_score(pts)
    
    # Constraints: each point must be on the unit sphere (or close to it)
    # Actually, we don't strictly need sphere constraint for the final answer,
    # but it helps with scaling. Let's just use bounds or no constraints for SLSQP
    # to allow free movement in 3D, which is more general.
    
    try:
        res = minimize(obj, x0, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-12})
        if res.success or (not res.success and _score(res.x.reshape(n, d)) > _score(x0.reshape(n, d))):
            return res.x.reshape(n, d)
    except Exception:
        pass
    
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a combination of:
    1. Structured initial guesses (icosahedron, cuboctahedron)
    2. Differential Evolution for global search
    3. Coordinate Descent for robust local refinement (from inspiration 1)
    4. SLSQP for smooth local refinement (from knowledge)
    """
    n = 14
    start_time = time.time()
    deadline = start_time + 580  # 630s limit, keep 50s margin
    
    best_score = -1.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        if points is not None and np.all(np.isfinite(points)):
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    # Generate diverse initial guesses
    guesses = []
    guesses.append(_make_icosahedron_plus_2())
    guesses.append(_make_cuboctahedron_plus_2())
    for i in range(10):
        guesses.append(_make_random_unit(n, 3, seed=1000 + i * 137))
    
    # Evaluate initial scores
    for gc in guesses:
        _update_best(gc)
    
    # Phase 1: Differential Evolution (global search)
    if time.time() < deadline - 180:
        try:
            # Define objective for DE (minimize -score)
            def de_obj(x):
                return -_score(x.reshape(n, 3))
            
            bounds = [(-2, 2)] * (n * 3)
            
            de_result = differential_evolution(
                de_obj, bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            if not np.any(np.isnan(de_result.x)):
                de_points = de_result.x.reshape(n, 3)
                _update_best(de_points)
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starts
    # For each of the top initial guesses and the DE result, apply local refinement
    if best_points is not None:
        _update_best(best_points)  # Ensure best is in the list
    
    # Create a list of starting points for local refinement
    start_points = list(guesses[:5])  # Top 5 structured/random starts
    if best_points is not None:
        start_points.append(best_points)
    
    for sp in start_points:
        if time.time() > deadline - 60:
            break
        
        # Refinement 1: Coordinate Descent (robust, non-smooth)
        if time.time() < deadline - 30:
            try:
                cd_points = _coordinate_descent_polish(sp, deadline - 10)
                _update_best(cd_points)
            except Exception:
                pass
        
        # Refinement 2: SLSQP (smooth, gradient-based)
        if time.time() < deadline - 10:
            try:
                slsqp_points = _polish_slsqp_sphere(sp, deadline - 5)
                _update_best(slsqp_points)
            except Exception:
                pass
        
        # Refinement 3: Coordinate Descent on SLSQP result
        if time.time() < deadline - 5:
            try:
                cd_points2 = _coordinate_descent_polish(slsqp_points, deadline - 2)
                _update_best(cd_points2)
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish on the best solution
    if best_points is not None and time.time() < deadline - 2:
        try:
            final_points = _coordinate_descent_polish(best_points, deadline - 1)
            _update_best(final_points)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _make_icosahedron_plus_2()
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = _make_icosahedron_plus_2()
    
    return best_points

# EVOLVE-BLOCK-END
