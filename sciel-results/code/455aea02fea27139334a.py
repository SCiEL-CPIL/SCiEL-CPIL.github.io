import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Precompute triangle index table once at module level for speed
_TRIANGLE_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)], dtype=np.int32)
_TRI_I = _TRIANGLE_INDICES[:, 0]
_TRI_J = _TRIANGLE_INDICES[:, 1]
_TRI_K = _TRIANGLE_INDICES[:, 2]

def _get_all_triangle_areas(points):
    """
    Calculate areas of all C(N,3) triangles formed by the points.
    Returns a numpy array of areas.
    """
    p_i = points[_TRI_I]
    p_j = points[_TRI_J]
    p_k = points[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective_smooth(points_flat, tau=50.0):
    """
    Smooth surrogate for the negative min triangle area.
    We want to maximize the minimum area, so we minimize -min_area.
    Since the hull area must be 1, we normalize inside.
    
    f(x) = min(area_i)
    smooth_f(x) = -1/tau * log(sum(exp(-tau * area_i)))
    
    We minimize -smooth_f(x) = 1/tau * log(sum(exp(-tau * area_i)))
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6  # Penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate areas
    areas = _get_all_triangle_areas(points)
    
    # 4. Smooth min approximation
    # To avoid overflow/underflow in exp:
    # log(sum(exp(-tau * areas))) = -tau * min_area + log(sum(exp(-tau * (areas - min_area))))
    m = np.min(areas)
    shifted = areas - m
    # shifted >= 0, so exp(-tau * shifted) <= 1, safe to sum
    sum_exp = np.sum(np.exp(-tau * shifted))
    
    # smooth_min = m - (1/tau) * log(sum_exp)
    # We want to minimize -smooth_min = (1/tau) * log(sum_exp) - m
    val = (1.0 / tau) * np.log(sum_exp) - m
    
    return val

def _objective_exact(points_flat):
    """
    Exact objective: -min_triangle_area, assuming points are normalized.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate min area
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return -min_area

def _generate_initial_candidates():
    """
    Generate diverse initial configurations for optimization.
    """
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts.flatten())
    
    # 2. Multi-ring configurations
    # Try different ring distributions: e.g., 7 outer, 7 inner; 6-5-3; etc.
    ring_configs = [
        (7, 7, [1.0, 0.6]),
        (7, 7, [1.0, 0.5]),
        (5, 5, 4, [1.0, 0.7, 0.4]),
        (4, 5, 5, [1.0, 0.7, 0.4]),
        (3, 5, 6, [1.0, 0.7, 0.4]),
        (2, 6, 6, [1.0, 0.8, 0.5]),
        (1, 6, 7, [1.0, 0.8, 0.5]),
    ]
    
    for config in ring_configs:
        if len(config) == 3:
            n1, n2, radii = config
            n3 = 0
        else:
            n1, n2, n3, radii = config
            
        pts_list = []
        
        # Ring 1
        ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        r1 = radii[0]
        pts1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
        pts_list.append(pts1)
        
        # Ring 2
        ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + (np.pi / n2 if n1 % 2 == 0 else np.pi / max(1, n1)) # Offset
        r2 = radii[1]
        pts2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        pts_list.append(pts2)
        
        # Ring 3
        if n3 > 0:
            ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + (np.pi / n3 if n2 % 2 == 0 else np.pi / max(1, n2))
            r3 = radii[2]
            pts3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
            pts_list.append(pts3)
        
        pts = np.vstack(pts_list)
        # Ensure exactly N points (pad or truncate if logic mismatch, though sums are 14)
        if len(pts) != N:
             # Fallback to regular polygon if config doesn't sum to 14
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        candidates.append(pts.flatten())
        
    # 3. Random Gaussian
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(N, 2)
        candidates.append(pts.flatten())
        
    return candidates

def _perturb(x, step=0.1):
    """Perturb a point for basin hopping."""
    return x + np.random.randn(len(x)) * step


def _repair_tightest(points, max_iter=80, tol=1e-10):
    """
    Iteratively push apart the three points forming the smallest triangle.
    This is a targeted local repair that can escape the flat plateau of the
    min-area objective where generic gradient-free methods stall.
    """
    pts = points.copy()
    for _ in range(max_iter):
        areas = _get_all_triangle_areas(pts)
        m = np.min(areas)
        if m < tol:
            break
        # Find all triangles within a tiny tolerance of the minimum
        idxs = np.where(areas <= m + 1e-12)[0]
        if len(idxs) == 0:
            break
        # Pick the first tightest triangle
        t = idxs[0]
        i, j, k = int(_TRI_I[t]), int(_TRI_J[t]), int(_TRI_K[t])
        pi, pj, pk = pts[i], pts[j], pts[k]
        # Centroid of the tightest triangle
        c = (pi + pj + pk) / 3.0
        # Push each vertex away from the centroid
        step = 0.02
        for idx, p in ((i, pi), (j, pj), (k, pk)):
            d = p - c
            nrm = np.linalg.norm(d)
            if nrm > 1e-15:
                pts[idx] = p + step * (d / nrm)
        # Also nudge the other points very slightly to avoid locking in
        # (helps break symmetry without destroying the configuration)
        others = [x for x in range(N) if x not in (i, j, k)]
        for o in others:
            d = pts[o] - c
            nrm = np.linalg.norm(d)
            if nrm > 1e-15:
                pts[o] = pts[o] + 0.002 * (d / nrm)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin before 120s hard limit
    
    best_score = -1.0
    best_points = None
    
    # Get initial candidates
    inits = _generate_initial_candidates()
    
    # Optimization Strategy:
    # 1. For each candidate, run a local optimization using Powell on the smooth objective.
    # 2. Then polish with Nelder-Mead on the exact objective.
    # This is faster and more robust than running a full global DE for every restart within the time limit,
    # provided we have good inits.
    
    # Alternatively, we could run one global DE with a budget, then refine.
    # Given N=14, 28 dimensions. DE is expensive but robust.
    # Let's try a hybrid: A few quick Powell runs on inits, then if time permits, a DE run on the best found so far.
    
    refined_candidates = []
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            # Phase 1: Smooth local optimization
            # Powell is good for medium dimensions and doesn't need gradients
            res1 = minimize(_objective_smooth, x0, method='Powell', 
                            options={'maxiter': 200, 'xtol': 1e-5, 'ftol': 1e-5, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p1 = res1.x.reshape(-1, 2)
            refined_candidates.append(p1)
            
            # Phase 2: Exact local optimization (Polish)
            res2 = minimize(_objective_exact, p1.flatten(), method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p2 = res2.x.reshape(-1, 2)
            
            # Normalize final result for consistency
            p2 = p2 - np.mean(p2, axis=0)
            hull_area = _hull_area(p2)
            if hull_area > 1e-10:
                p2 = p2 / np.sqrt(hull_area)
            
            s = _score(p2)
            
            if s > best_score:
                best_score = s
                best_points = p2
            
            # Phase 3: Targeted repair of the tightest triangle
            if time.time() < deadline - 5.0:
                try:
                    p3 = _repair_tightest(p2, max_iter=60)
                    p3 = p3 - np.mean(p3, axis=0)
                    hull_area3 = _hull_area(p3)
                    if hull_area3 > 1e-10:
                        p3 = p3 / np.sqrt(hull_area3)
                    s3 = _score(p3)
                    if s3 > best_score:
                        best_score = s3
                        best_points = p3
                except Exception:
                    pass
                
        except Exception as e:
            pass
    
    # If we have time, try a broader search using the best point found so far
    if time.time() < deadline - 30.0 and best_points is not None:
        try:
            # Run Basin Hopping on the best point found
            # Basin Hopping is good for escaping local minima
            x_best = best_points.flatten()
            
            res_bh = basinhopping(
                _objective_exact, 
                x_best, 
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}},
                niter=10, 
                T=1.0, 
                stepsize=0.1,
                minimizer='L-BFGS-B' # Try L-BFGS-B as inner minimizer? No, Nelder-Mead is better for non-smooth.
                # Actually, basinhopping uses the minimizer function. Let's stick to Nelder-Mead for consistency.
                # But basinhopping might be slow. Let's just do manual basin hopping steps.
            )
            # Since basinhopping can be tricky with non-smooth objectives and time limits, 
            # let's implement a simple manual basin hopping loop instead if the above is too risky.
            # The above call might fail or be slow. Let's replace with a safer manual loop.
            
        except Exception:
            pass
            
        # Manual Basin Hopping / Random Restarts around best
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            noise = np.random.randn(N, 2) * 0.2
            p_perturbed = best_points + noise
            try:
                # Use a slightly more robust local optimizer
                res_refine = minimize(_objective_exact, p_perturbed.flatten(), method='Nelder-Mead', 
                                      options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                p_ref = res_refine.x.reshape(-1, 2)
                p_ref = p_ref - np.mean(p_ref, axis=0)
                hull_area = _hull_area(p_ref)
                if hull_area > 1e-10:
                    p_ref = p_ref / np.sqrt(hull_area)
                s = _score(p_ref)
                if s > best_score:
                    best_score = s
                    best_points = p_ref
                
                # Also apply tightest-triangle repair after each restart
                if time.time() < deadline - 5.0:
                    try:
                        p_rep = _repair_tightest(p_ref, max_iter=40)
                        p_rep = p_rep - np.mean(p_rep, axis=0)
                        hull_area_r = _hull_area(p_rep)
                        if hull_area_r > 1e-10:
                            p_rep = p_rep / np.sqrt(hull_area_r)
                        s_rep = _score(p_rep)
                        if s_rep > best_score:
                            best_score = s_rep
                            best_points = p_rep
                    except Exception:
                        pass
            except Exception:
                pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = best_points - np.mean(best_points, axis=0)
        hull_area = _hull_area(best_points)
        if hull_area > 1e-10:
             best_points = best_points / np.sqrt(hull_area)

    return best_points

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points shape: {pts.shape}")
