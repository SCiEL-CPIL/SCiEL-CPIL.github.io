"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 10.0


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_tri = min_triangle_area(points)
    return min_tri / hull_area


def construct():
    entry_time = time.time()
    deadline = entry_time + (DEADLINE - SAFETY_MARGIN)
    
    rng = np.random.default_rng(42)
    best_score = 0.0
    best_points = None
    
    # Initial candidates
    # 1. Regular 13-gon
    theta13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    poly13 = np.column_stack([np.cos(theta13), np.sin(theta13)])
    s = score(poly13)
    if s > best_score:
        best_score = s
        best_points = poly13.copy()
        
    # 2. Center + 12 points on circle
    center = np.array([[0.0, 0.0]])
    theta12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring12 = np.column_stack([np.cos(theta12), np.sin(theta12)])
    pts2 = np.vstack([center, ring12])
    s = score(pts2)
    if s > best_score:
        best_score = s
        best_points = pts2.copy()
        
    # 3. Random points
    for _ in range(20):
        pts = rng.random((N, 2))
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Normalize
    def normalize(pts):
        pts = pts - pts.mean(axis=0)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        return pts
        
    best_points = normalize(best_points)
    best_score = score(best_points)

    # Strategy: Global Optimization via Differential Evolution
    # We maximize min_triangle_area / hull_area.
    # To make it differentiable for DE (which doesn't need gradients but smooth landscapes help convergence),
    # we can use a soft-min approximation or just use the exact score.
    # DE is derivative-free, so we can use the exact score directly, but it's noisy.
    # However, DE is robust to noise. Let's try DE on the exact score first.
    # The domain is bounded: x, y in [-1, 1] (after normalization, points are within unit circle roughly).
    # Actually, we can let DE search in a box, say [-1.5, 1.5]^26.
    
    def neg_score_de(x_flat):
        pts = x_flat.reshape(13, 2)
        # Optional: normalize inside the objective to keep points in a reasonable range?
        # No, DE explores the box. We should normalize the output of DE?
        # If we normalize inside, the landscape becomes flat in radial directions.
        # Better: Constrain the search to the unit disk?
        # DE doesn't handle constraints well directly, but we can use bounds.
        # Let's use bounds [-1, 1] for each coordinate.
        # And we will normalize the final result.
        # But during optimization, normalization changes the scale.
        # It is better to optimize on normalized coordinates?
        # Let's fix the scale by normalizing inside the objective function?
        # If I normalize inside, the gradient of the normalization is problematic for some optimizers,
        # but DE doesn't use gradients.
        # However, if I normalize, the search space effectively becomes the space of shapes (projective plane).
        # Let's try optimizing without normalization inside, but with tight bounds,
        # and then normalize the result.
        # The score is scale-invariant. So we can just maximize min_triangle_area / hull_area.
        
        # Check for degeneracy
        if np.any(np.linalg.norm(pts, axis=1) == 0):
            return 0.0
        s = score(pts)
        if not np.isfinite(s):
            return 0.0
        return -s

    # Define bounds for DE
    bounds = [(-1.5, 1.5)] * 26
    
    # Run Differential Evolution
    # DE is stochastic. We need to set seed for reproducibility? rng is used inside DE if provided?
    # scipy differential_evolution takes 'seed' param.
    
    start_time_de = time.time()
    remaining_time = deadline - start_time_de
    if remaining_time > 30: # Need some time for DE
        try:
            # DE parameters:
            # popsize: 15 (default)
            # maxiter: 100
            # tol: 1e-4
            # We can monitor time.
            
            # To save time, we might reduce popsize or maxiter if time is tight.
            # Let's estimate time per iteration.
            # DE evaluates popsize * dim function calls per generation? No, popsize total evaluations per generation?
            # DE generates a new population of size popsize.
            # So evaluations per generation = popsize.
            # 15 * 100 = 1500 evaluations.
            # Each eval is ~1-2ms. 1500 * 2ms = 3s.
            # This is very fast. We can afford more iterations.
            
            # Let's increase popsize and maxiter to explore better.
            # popsize=20, maxiter=200 -> 4000 evals -> ~8s.
            # We have ~140s. We can do much more.
            # popsize=30, maxiter=500 -> 15000 evals -> ~30s.
            
            # Let's do a multi-start DE or just a long DE.
            # DE is global. One long run should be good.
            
            res = differential_evolution(
                neg_score_de, 
                bounds, 
                maxiter=1000, 
                popsize=25, 
                tol=1e-6, 
                seed=42,
                polish=True, # Uses L-BFGS-B to polish the best result
                init='latinhypercube'
            )
            
            if res.x is not None:
                pts_de = res.x.reshape(13, 2)
                # Normalize the result
                pts_de_norm = normalize(pts_de)
                s_de = score(pts_de_norm)
                if s_de > best_score:
                    best_score = s_de
                    best_points = pts_de_norm.copy()
                    
        except Exception as e:
            print(f"DE failed: {e}")

    # Final Polish: Local Nelder-Mead or L-BFGS-B from the best point found
    # DE's polish might have already done this, but let's do a dedicated local search.
    # Use L-BFGS-B on a smooth surrogate? Or just NM on exact score?
    # NM on exact score is robust.
    
    if time.time() < deadline - 5:
        def neg_score_nm(x_flat):
            pts = x_flat.reshape(13, 2)
            # Normalize to keep scale consistent? 
            # If we normalize inside, the search space is distorted.
            # Let's just use the raw points and rely on the fact that we started from a normalized point.
            # But NM might drift in scale.
            # Let's normalize inside the objective for NM to ensure scale invariance is handled?
            # No, normalization is non-linear.
            # Let's just use the exact score on the raw points.
            # If the points drift in scale, the score is invariant, so it's fine.
            s = score(pts)
            if not np.isfinite(s):
                return 0.0
            return -s

        x0 = best_points.flatten()
        
        # Try a few restarts of NM
        for _ in range(5):
            if time.time() > deadline - 2: break
            x0_pert = x0 + rng.normal(0, 0.005, (26,))
            try:
                res_nm = minimize(neg_score_nm, x0_pert, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res_nm.success:
                    pts_opt = res_nm.x.reshape(13, 2)
                    # Normalize
                    pts_opt = normalize(pts_opt)
                    s_opt = score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt.copy()
            except Exception:
                pass

    return best_points

# EVOLVE-BLOCK-END
