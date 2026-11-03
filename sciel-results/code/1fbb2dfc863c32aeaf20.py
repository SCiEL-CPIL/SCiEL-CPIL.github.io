import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
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

def _objective(flat):
    """
    Objective function for SLSQP.
    We want to maximize min_triangle_area / hull_area.
    This is equivalent to minimizing hull_area / min_triangle_area.
    However, SLSQP handles constraints better.
    
    Alternative formulation:
    Maximize t such that all triangle areas >= t * hull_area.
    This is complex because hull_area depends on points.
    
    Let's use a simpler surrogate:
    Minimize -min_triangle_area / hull_area.
    Since this is non-smooth, we can use a soft-min or just rely on SLSQP's robustness
    with a well-initialized start.
    
    Actually, a common trick for max-min is:
    Minimize -t
    Subject to: area_ij_k - t * hull_area >= 0 for all triples.
    But hull_area is non-linear and non-smooth at vertices.
    
    Let's stick to minimizing the ratio: R = hull_area / min_triangle_area.
    R is smooth almost everywhere.
    """
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _compute_min_triangle_area(pts)
    if min_area < 1e-10:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _generate_fekete_points(n):
    """
    Generate Fekete points in the unit disk.
    Fekete points maximize the product of pairwise distances.
    We use a simple iterative relaxation or just a good initial guess
    like a regular polygon perturbed slightly, or use known Fekete configurations.
    For n=14, a regular 14-gon is a good start, but Fekete points are slightly different.
    Since we don't have a direct Fekete solver, we will use a regular polygon
    and let the optimizer move them. Or we can use a few random restarts.
    
    To be "Fekete-like", we can start with a regular polygon and then
    run a few steps of maximizing the product of distances (Vandermonde determinant).
    """
    # Start with regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Relax to maximize product of distances (log product = sum log dist)
    # This is equivalent to maximizing the Vandermonde determinant magnitude.
    def vandermonde_obj(flat):
        p = flat.reshape(n, 2)
        # Calculate sum of log distances
        # To avoid log(0), add epsilon
        total = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(p[i] - p[j])
                total += np.log(d + 1e-8)
        return -total # Minimize negative log product
    
    # Use Nelder-Mead to relax towards Fekete points
    try:
        res = minimize(vandermonde_obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            pts = res.x.reshape(n, 2)
    except Exception:
        pass
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 1: Global Search with Differential Evolution (Full 28D space)
    # DE is robust and doesn't require gradients.
    # We minimize hull_area / min_triangle_area.
    
    if time.time() < deadline - 60.0:
        def de_obj(flat):
            pts = flat.reshape(N, 2)
            # Normalize to unit area to keep values bounded?
            # The problem statement says "normalized to unit area" but the score is min_area / hull_area.
            # Scaling points scales both area and hull area by the same factor, so the ratio is invariant to scaling.
            # Translation is also invariant.
            # So we can just search in a bounded box, e.g., [-1, 1]^2.
            min_a = _compute_min_triangle_area(pts)
            if min_a < 1e-8:
                return 1e10
            hull_a = _compute_hull_area(pts)
            if hull_a < 1e-8:
                return 1e10
            return hull_a / min_a

        bounds = [(-1, 1)] * (2 * N)
        
        # Run DE with a decent population size
        try:
            res_de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-5, mutation=(0.5, 1.0), recombination=0.9,
                polish=False, init='latinhypercube'
            )
            if res_de.fun < 1e9:
                pts_de = res_de.x.reshape(N, 2)
                score_de = _score(pts_de)
                update_best(pts_de, score_de)
        except Exception:
            pass

    # Phase 2: Local Refinement with SLSQP from Fekete/DE starts
    # SLSQP is a gradient-based method (finite differences) that handles constraints well.
    # We will use it to polish the results from DE and also try Fekete starts.
    
    def refine_slsqp(flat_init, maxiter=200):
        try:
            res = minimize(
                _objective, flat_init, method='SLSQP',
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            return pts, score
        except Exception:
            return None, -np.inf

    # Start 1: Fekete Points
    if time.time() < deadline - 40.0:
        pts_fekete = _generate_fekete_points(N)
        cand_pts, cand_score = refine_slsqp(pts_fekete.flatten(), maxiter=300)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # Start 2: Regular Polygon
    if time.time() < deadline - 30.0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        cand_pts, cand_score = refine_slsqp(pts_reg.flatten(), maxiter=300)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # Start 3: Best from DE
    if best_points is not None and time.time() < deadline - 20.0:
        cand_pts, cand_score = refine_slsqp(best_points.flatten(), maxiter=300)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)

    # Phase 3: Multi-start SLSQP with random perturbations of the best point
    if best_points is not None and time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            noise = np.random.normal(0, 0.01, size=best_points.shape)
            perturbed = best_points + noise
            cand_pts, cand_score = refine_slsqp(perturbed.flatten(), maxiter=200)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # Phase 4: Final Polish with Nelder-Mead if SLSQP got stuck
    if best_points is not None and time.time() < deadline - 5.0:
        def nm_obj(flat):
            return -_score(flat.reshape(N, 2))
        
        try:
            res_nm = minimize(nm_obj, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            pts_nm = res_nm.x.reshape(N, 2)
            score_nm = -res_nm.fun
            update_best(pts_nm, score_nm)
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
