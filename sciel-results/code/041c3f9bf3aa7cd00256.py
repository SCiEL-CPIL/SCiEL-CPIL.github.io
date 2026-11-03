from scipy.optimize import differential_evolution
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution as _de
import time
import itertools
from itertools import combinations
from scipy.optimize import differential_evolution

N = 13

# Precompute all triples for vectorized area computation
_TRIPLES = list(combinations(range(N), 3))
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])

def _all_triangle_areas(pts):
    """Compute all C(n,3) triangle areas efficiently using vectorization."""
    n = len(pts)
    if n < 3:
        return np.array([0.0])
    
    pi = pts[_I_IDX]
    pj = pts[_J_IDX]
    pk = pts[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    areas = _all_triangle_areas(pts)
    return np.min(areas) if len(areas) > 0 else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _soft_min_area(areas, alpha):
    """
    Soft minimum of areas using log-sum-exp.
    Returns: -1/alpha * log(sum(exp(-alpha * A_k)))
    This is a lower bound on min(A_k), tight as alpha -> inf.
    """
    if len(areas) == 0:
        return 0.0
    M = np.min(areas)
    shifted = areas - M
    log_sum = -alpha * M + np.log(np.sum(np.exp(-alpha * shifted)))
    return -log_sum / alpha

def _neg_soft_score(params, alpha):
    pts = params.reshape(-1, 2)
    areas = _all_triangle_areas(pts)
    soft_min = _soft_min_area(areas, alpha)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return -(soft_min / hull_a)

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _concentric_init(n_circles, radii, angles_per_circle, rng=None):
    """
    Generate points on concentric circles.
    n_circles: number of circles
    radii: array of radii (length n_circles)
    angles_per_circle: array of how many points on each circle
    Returns: (N, 2) array of points
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    pts = []
    for c in range(n_circles):
        r = radii[c]
        n_pts = angles_per_circle[c]
        if n_pts == 0:
            continue
        # Evenly spaced angles with possible offset
        base_angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        # Add small random offset for variety
        offset = rng.uniform(0, 2 * np.pi)
        angles = base_angles + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts.extend(zip(x, y))
    
    pts = np.array(pts)
    if len(pts) < N:
        # Pad with center points
        pad = np.zeros((N - len(pts), 2))
        pts = np.vstack([pts, pad])
    elif len(pts) > N:
        pts = pts[:N]
    
    return pts

def _concentric_params_to_points(params, n_circles, counts):
    """
    Convert parameter vector to point positions.
    params: [r1, theta1, r2, theta2, ..., rn, thetan] for each circle
    counts: list of how many points per circle
    """
    pts = []
    idx = 0
    for c in range(n_circles):
        r = params[idx]
        theta0 = params[idx + 1]
        idx += 2
        n_c = counts[c]
        if n_c == 1:
            # Single point: just place at angle theta0
            pts.append([r * np.cos(theta0), r * np.sin(theta0)])
        elif n_c > 1:
            angles = theta0 + np.linspace(0, 2 * np.pi, n_c, endpoint=False)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.extend(zip(x, y))
    pts = np.array(pts)
    return pts


def _generate_concentric_initializations():
    """Generate diverse concentric circle configurations."""
    rng = np.random.RandomState(123)
    inits = []
    
    # Configuration 1: 13 points on one circle (regular polygon)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Configuration 2: 7 outer + 6 inner
    for r2 in [0.3, 0.4, 0.5, 0.6]:
        ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
        ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
    
    # Configuration 3: 12 outer + 1 center
    angles_m = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # Configuration 4: 9 outer + 4 middle
    for r2 in [0.4, 0.5, 0.6]:
        ang1 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
        ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
    
    # Configuration 5: 6 outer + 5 middle + 2 inner
    for r2, r3 in [(0.4, 0.2), (0.5, 0.25), (0.6, 0.3)]:
        ang1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
        ang3 = np.linspace(0, 2 * np.pi, 2, endpoint=False) + np.pi/2
        ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
        inits.append(np.vstack([ring1, ring2, ring3]))
    
    # Configuration 6: Random concentric with 2-3 circles
    for _ in range(15):
        n_circles = rng.choice([2, 3])
        if n_circles == 2:
            n1 = rng.choice([5, 6, 7, 8])
            n2 = N - n1
            r1 = 1.0
            r2 = rng.uniform(0.3, 0.7)
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2*np.pi)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2*np.pi)
            ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            inits.append(np.vstack([ring1, ring2]))
        else:
            n1 = rng.choice([4, 5, 6])
            n2 = rng.choice([3, 4, 5])
            n3 = N - n1 - n2
            if n3 < 1:
                continue
            r1 = 1.0
            r2 = rng.uniform(0.4, 0.7)
            r3 = rng.uniform(0.15, 0.35)
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2*np.pi)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2*np.pi)
            ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + rng.uniform(0, 2*np.pi)
            ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
            inits.append(np.vstack([ring1, ring2, ring3]))
    
    # Configuration 7: Random points in disk
    for _ in range(10):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    return inits


def _symmetric_optimize(n_circles, counts, deadline, seed=42):
    """
    Optimize concentric regular-polygon configuration using differential evolution
    on the reduced parameter space (2 * n_circles parameters).
    """
    n_params = 2 * n_circles
    
    def _neg_score_sym(params):
        pts = _concentric_params_to_points(params, n_circles, counts)
        if len(pts) != N:
            return 0.0
        return -_score(pts)
    
    # Bounds: radius in [0, 1.5], angle in [0, 2*pi]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.5))  # radius
        bounds.append((0.0, 2 * np.pi))  # angle offset
    
    try:
        res = differential_evolution(
            _neg_score_sym,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=seed,
            polish=True,
            init='sobol'
        )
        best_params = res.x
        best_pts = _concentric_params_to_points(best_params, n_circles, counts)
        return best_pts, _score(best_pts)
    except Exception:
        return None, 0.0

def _refine_with_soft_min(pts, deadline):
    """
    Refine points using soft-min annealing.
    Returns improved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Anneal alpha from low to high
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
        
        try:
            def _neg_soft_closure(params, a=alpha):
                return _neg_soft_score(params, a)
            
            res = minimize(
                _neg_soft_closure,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_pts = res.x.reshape(-1, 2)
            soft_score = _score(soft_pts)
            if soft_score > current_score:
                current_pts = soft_pts
                current_score = soft_score
        except Exception:
            pass
    
    return current_pts

def _refine_with_nelder_mead(pts, deadline):
    """
    Refine points using Nelder-Mead on exact score.
    Returns improved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 5.0:
        return current_pts
    
    try:
        res = minimize(
            _neg_score,
            current_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        nm_pts = res.x.reshape(-1, 2)
        nm_score = _score(nm_pts)
        if nm_score > current_score:
            current_pts = nm_pts
            current_score = nm_score
    except Exception:
        pass
    
    return current_pts

def _targeted_worst_triangle_opt(pts, deadline):
    """
    Identify the worst (minimum area) triangles and try to improve them
    by moving the involved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 10.0:
        return current_pts
    
    # Find all minimum-area triangles
    areas = _all_triangle_areas(current_pts)
    min_area = np.min(areas)
    
    # Find triples with area close to minimum
    threshold = min_area * 1.05
    worst_triples = []
    for idx, area in enumerate(areas):
        if area <= threshold:
            # Decode triple index
            # We need to map linear index back to (i,j,k)
            # C(n,3) enumeration: for n=13, indices 0..285
            # Simple approach: just iterate again
            pass
    
    # Instead of complex indexing, just find worst triples directly
    n = len(current_pts)
    worst_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = current_pts[j] - current_pts[i]
                v2 = current_pts[k] - current_pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area <= threshold:
                    worst_triples.append((i, j, k, area))
    
    # For each worst triple, try small perturbations of its points
    rng = np.random.RandomState(42)
    for (i, j, k, area) in worst_triples[:5]:  # Limit to top 5 worst
        if time.time() > deadline - 5.0:
            break
        
        # Try moving each of the 3 points in random directions
        for trial in range(20):
            trial_pts = current_pts.copy()
            # Perturb one of the three points
            idx = [i, j, k][trial % 3]
            dx = rng.randn() * 0.05
            dy = rng.randn() * 0.05
            trial_pts[idx] += np.array([dx, dy])
            
            trial_score = _score(trial_pts)
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Phase 1: Evaluate standard initializations
    inits = _generate_concentric_initializations()
    scored_inits = []
    for init_pts in inits:
        if time.time() > deadline - 100.0:
            break
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[0])
    
    # Phase 2: Symmetry-exploiting optimization on various circle configurations
    # Try different numbers of concentric circles with various point distributions
    circle_configs = [
        (1, [13]),
        (2, [7, 6]),
        (2, [8, 5]),
        (2, [9, 4]),
        (2, [10, 3]),
        (2, [11, 2]),
        (2, [12, 1]),
        (3, [6, 5, 2]),
        (3, [7, 4, 2]),
        (3, [8, 3, 2]),
        (3, [5, 5, 3]),
        (3, [6, 4, 3]),
        (3, [7, 3, 3]),
        (3, [4, 4, 5]),
        (3, [5, 4, 4]),
        (4, [5, 4, 3, 1]),
        (4, [6, 3, 3, 1]),
        (4, [4, 4, 3, 2]),
    ]
    
    for n_circles, counts in circle_configs:
        if time.time() > deadline - 20.0:
            break
        sym_pts, sym_score = _symmetric_optimize(n_circles, counts, deadline, seed=n_circles * 100 + sum(counts))
        if sym_pts is not None and sym_score > best_score:
            best_score = sym_score
            best_pts = sym_pts.copy()
    
    # Phase 3: Refine top candidates with soft-min and Nelder-Mead
    top_k = min(5, len(scored_inits))
    for rank in range(top_k):
        if time.time() > deadline - 15.0:
            break
        
        s, pts = scored_inits[rank]
        
        # Soft-min annealing
        refined = _refine_with_soft_min(pts, deadline)
        refined_score = _score(refined)
        
        # Nelder-Mead polish
        refined = _refine_with_nelder_mead(refined, deadline)
        refined_score = _score(refined)
        
        # Targeted worst-triangle optimization
        refined = _targeted_worst_triangle_opt(refined, deadline)
        refined_score = _score(refined)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    # Phase 4: Full-space differential evolution if time permits
    if time.time() < deadline - 25.0:
        try:
            bounds = [(-1.5, 1.5)] * (2 * N)
            res_de = differential_evolution(
                _neg_score,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=True,
                init='sobol'
            )
            de_pts = res_de.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 3.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    return best_pts


# EVOLVE-BLOCK-END
