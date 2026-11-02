import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
# C(14, 3) = 364
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        # Check if points are collinear or degenerate
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. We should return well before that.
    # Let's aim for 135s deadline to be safe.
    deadline = start_time + 135.0
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8):
        # Normalize initial points to center and unit scale for better conditioning
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        # Scale to max norm 1
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            elif method == 'L-BFGS-B':
                res = minimize(_objective, pts.flatten(), method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # 1. Initializations
    initializations = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(x_reg)
    
    # Grid 4x4 (remove 2)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    x_grid = grid_pts[:14]
    initializations.append(x_grid)
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Random points
    rng = np.random.default_rng(123)
    for _ in range(2):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)

    # 2. Refine initializations
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        # Nelder-Mead
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1500)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        # Powell on best found so far from this init (if time allows)
        if time.time() < deadline - 20.0:
            cand_pts2, cand_score2 = refine(cand_pts, 'Powell', maxiter=500)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 3. Differential Evolution for global exploration
    # This is the key improvement suggested by inspiration 3.
    # We need to define bounds. Since the problem is scale/translation invariant,
    # we can fix the center to (0,0) and constrain points to a box, say [-1, 1].
    # However, DE requires a fixed dimension. 14*2 = 28 variables.
    
    if time.time() < deadline - 30.0:
        # We need to handle the invariance. 
        # One way: fix point 0 to (0,0) and point 1 to (1,0) to break translation/rotation?
        # No, rotation is also an invariance. 
        # Actually, fixing two points breaks translation and rotation.
        # But the optimal configuration might not have points at specific locations.
        # Better: Use a large box and let the optimizer find the scale.
        # The objective is scale invariant, so the landscape is flat in the radial direction.
        # This can hurt gradient-based methods but DE is derivative-free.
        
        # Let's try DE with bounds [-1, 1] for all coordinates.
        # To ensure we don't get stuck in a degenerate solution (all points same),
        # we rely on the hull area penalty.
        
        # We need a fast objective for DE.
        def de_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            # Quick check for degeneracy
            if np.allclose(pts, pts[0]):
                return 1.0 # Bad score (minimize negative score, so return large positive)
            return -_score(pts)
            
        # DE parameters
        # popsize=10 means 10*28 = 280 evaluations per generation? No, popsize is the multiplier for N.
        # So 10 * 28 = 280 individuals.
        # maxiter=100 generations -> 28000 evaluations.
        # Each evaluation is ~0.1ms? 28000 * 0.1ms = 2.8s.
        # Let's be conservative.
        
        try:
            bounds = [(-1, 1)] * (N * 2)
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=8, 
                tol=1e-4, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True, # Use L-BFGS-B to polish the result
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                cand_pts = res_de.x.reshape(N, 2)
                cand_score = -res_de.fun
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # 4. Final polish on the global best
    if best_points is not None and time.time() < deadline - 10.0:
        # Try a few local refinements with different methods
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > deadline - 10.0:
                break
            cand_pts, cand_score = refine(best_points, method, maxiter=1000)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # 5. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 15.0:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            noise = rng2.normal(0, 0.1, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=500)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    if best_points is None:
        # Fallback
        best_points = initializations[0]
        
    return best_points
