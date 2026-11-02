"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 145.0
START_TIME = time.time()
SAFETY_MARGIN = 3.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples using vectorization."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Pre-compute all C(n,3) triples
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    k_arr = np.array(k_idx)
    
    p0 = pts[i_arr]
    p1 = pts[j_arr]
    p2 = pts[k_arr]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    if min_ta <= 0:
        return 0.0
    return min_ta / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _try_optimize(pts, max_iter=200, xatol=1e-9, fatol=1e-11):
    """Try to optimize a point configuration. Returns improved points or None."""
    def _flatten(p):
        return p.flatten()
    
    def _unflatten(x):
        return x.reshape(N, 2)
    
    try:
        result = minimize(
            _neg_score,
            _flatten(pts),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        if result.fun < 0:
            candidate = _unflatten(result.x)
            return candidate
    except Exception:
        pass
    return None


def _try_optimize_l_bfgs(pts, max_iter=100):
    """Try L-BFGS-B optimization."""
    def _flatten(p):
        return p.flatten()
    
    def _unflatten(x):
        return x.reshape(N, 2)
    
    try:
        result = minimize(
            _neg_score,
            _flatten(pts),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        if result.fun < 0:
            candidate = _unflatten(result.x)
            return candidate
    except Exception:
        pass
    return None


def _sobol_sequence(n, dim=2, seed=42):
    """Generate quasi-random points using a simple Sobol-like sequence."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return sampler.random(n)


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    # 1. Start with regular polygon
    pts0 = _regular_polygon(N)
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # 2. Local refinement from regular polygon
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 5:
        candidate = _try_optimize(best_pts, max_iter=400, xatol=1e-10, fatol=1e-12)
        if candidate is not None:
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    
    # 3. Multi-start with diverse initializations
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    np.random.seed(42)
    
    # Use quasi-random (Sobol) sequences for better coverage
    try:
        sobol_pts_list = []
        sampler = __import__('scipy.stats', fromlist=['qmc']).qmc.Sobol(d=2, scramble=True, seed=42)
        for i in range(50):
            sample = sampler.random(N)
            # Map [0,1]^2 to a region around origin
            mapped = (sample - 0.5) * 2.0
            sobol_pts_list.append(mapped)
    except Exception:
        sobol_pts_list = []
    
    start_points = []
    
    # Add regular polygon variants
    for r in [0.5, 0.7, 1.0, 1.3]:
        for theta in [0, np.pi/6, np.pi/3]:
            poly = _regular_polygon(N, radius=r)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            start_points.append(poly @ rot.T)
    
    # Add perturbed best
    for scale in [0.02, 0.05, 0.1, 0.2, 0.5]:
        start_points.append(best_pts + np.random.normal(0, scale, size=(N, 2)))
    
    # Add Sobol points
    for sp in sobol_pts_list[:20]:
        start_points.append(sp)
    
    # Add random points in unit disk
    for _ in range(15):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0.05, 1.2, N))
        pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        start_points.append(pts_init)
    
    # Add some structured patterns
    # Concentric rings
    for r1, r2 in [(0.3, 0.8), (0.5, 1.0), (0.4, 0.9)]:
        angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False)
        inner = r1 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        outer = r2 * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        combined = np.vstack([inner, outer])
        start_points.append(combined)
    
    # Shuffle start points
    np.random.shuffle(start_points)
    
    # Optimize from each starting point
    for pts_init in start_points:
        elapsed = time.time() - START_TIME
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        if remaining < 3:
            break
        
        # Try Nelder-Mead
        candidate = _try_optimize(pts_init, max_iter=150, xatol=1e-9, fatol=1e-11)
        if candidate is not None:
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
        
        # If time allows, also try L-BFGS-B
        elapsed = time.time() - START_TIME
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        if remaining > 10:
            candidate2 = _try_optimize_l_bfgs(best_pts, max_iter=50)
            if candidate2 is not None:
                cand_score2 = _score(candidate2)
                if cand_score2 > best_score:
                    best_pts = candidate2
                    best_score = cand_score2
    
    # 4. Final fine-tuning if time remains
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 8:
        # Tighter optimization
        candidate = _try_optimize(best_pts, max_iter=300, xatol=1e-11, fatol=1e-13)
        if candidate is not None:
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    
    # 5. One more round of small perturbations
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 5:
        for _ in range(5):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            pts_pert = best_pts + np.random.normal(0, 0.01, size=(N, 2))
            candidate = _try_optimize(pts_pert, max_iter=100, xatol=1e-10, fatol=1e-12)
            if candidate is not None:
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
    
    # Final check to ensure valid output
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback to regular polygon if something went wrong
        best_pts = _regular_polygon(N)

    return best_pts
# EVOLVE-BLOCK-END
