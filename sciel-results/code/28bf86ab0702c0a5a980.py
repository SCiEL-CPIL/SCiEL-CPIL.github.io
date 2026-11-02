import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

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

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # We use a centrally symmetric parameterization to reduce dimensionality
    # Points are (x_i, y_i) for i in 0..6, and (-x_i, -y_i) for i in 7..13
    # This exploits the fact that optimal Heilbronn configs for even N are often centrally symmetric.
    
    def _feasibility_objective(flat, t_target, beta=100.0):
        """
        Objective for fixed target area t_target.
        We want to find points such that all triangle areas >= t_target.
        We minimize a penalty for violations + a term to keep the hull area bounded (or normalized).
        
        Actually, the problem is: maximize min_area / hull_area.
        Let's reframe: 
        Let scale s be a variable. 
        If we fix the shape, the ratio is invariant to scaling.
        So we can fix the hull area to be 1 (or close to 1) and maximize min_area.
        
        Better approach: 
        Minimize:  -min_area  subject to hull_area <= 1.
        Or: Minimize:  hull_area - lambda * min_area.
        
        Let's use the "target area" approach with bisection.
        For a given target ratio R, does there exist a config with min_area/hull_area >= R?
        This is equivalent to: exists points such that min_area >= R * hull_area.
        
        Let's use a simpler robust method:
        Optimize: min  -min(areas) + penalty * (hull_area - 1)^2
        But this is non-convex.
        
        Let's stick to the direct optimization of the ratio, but using a smoother objective.
        Objective: - (min_area / hull_area)
        We can approximate min_area with a log-sum-exp of -areas.
        soft_min = -1/beta * log(sum(exp(-beta * areas)))
        Objective: - soft_min / hull_area
        
        This is smooth and differentiable.
        """
        pts_half = flat.reshape(N_HALF, 2)
        pts = np.vstack([pts_half, -pts_half])
        
        if np.any(~np.isfinite(pts)):
            return 1e10
            
        areas = _all_areas(pts)
        if np.min(areas) < 1e-12:
            return 1e10
            
        # Soft min of areas
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min = -log_sum / beta
        
        hull_area = _compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
            
        return -soft_min / hull_area

    def _objective_hard(flat):
        pts_half = flat.reshape(N_HALF, 2)
        pts = np.vstack([pts_half, -pts_half])
        return -_score(pts)

    # Phase 1: Multi-start local optimization with L-BFGS-B on the soft objective
    # L-BFGS-B is much better than Nelder-Mead for high dimensions and smooth functions.
    
    # Generate diverse starting points
    # 1. Regular polygon (symmetrized)
    angles = np.linspace(0, np.pi, N_HALF, endpoint=False)
    radii = np.ones(N_HALF)
    start_1 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).flatten()
    
    # 2. Random symmetric points
    np.random.seed(42)
    starts = [start_1]
    for _ in range(10):
        pts = np.random.uniform(-1, 1, (N_HALF, 2))
        starts.append(pts.flatten())
        
    # 3. Grid-like structures
    # Try to place points on a hexagonal lattice pattern
    # This is often good for packing-like problems
    hex_starts = []
    # 7 points in a hexagon cluster
    # Center + 6 around
    # But we need symmetric pairs.
    # If we have a point at (0,0), its symmetric pair is (0,0). So we can have 1 center point?
    # No, the parameterization is (p1, p2, p3, p4, p5, p6, p7) and (-p1, ..., -p7).
    # If p7 = (0,0), then -p7 = (0,0). So we have a duplicate point.
    # The problem says "Place 14 points". Usually distinct points are implied or allowed?
    # If points coincide, triangle area is 0. So we must avoid p_i = -p_j or p_i = p_j.
    # In the symmetric parameterization, p_i cannot be 0.
    
    # Let's refine the random starts with a bit of structure
    for i in range(5):
        # Spiral
        r = 0.5 + 0.5 * np.random.rand(N_HALF)
        th = np.linspace(0, np.pi, N_HALF) + np.random.uniform(-0.2, 0.2)
        pts = np.column_stack((r * np.cos(th), r * np.sin(th)))
        starts.append(pts.flatten())

    for i, start in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Use L-BFGS-B with the soft objective
            # Beta controls the sharpness of the soft-min. Start with moderate beta.
            beta = 50.0
            
            res = minimize(
                lambda x: _feasibility_objective(x, t_target=0.0, beta=beta),
                start,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            pts_half = res.x.reshape(N_HALF, 2)
            pts = np.vstack([pts_half, -pts_half])
            score = _score(pts)
            update_best(pts, score)
            
            # Increase beta for sharper minimum approximation
            if time.time() < deadline - 10.0:
                for beta in [100.0, 200.0]:
                    if time.time() > deadline - 10.0:
                        break
                    res2 = minimize(
                        lambda x, b=beta: _feasibility_objective(x, t_target=0.0, beta=b),
                        res.x,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    pts_half2 = res2.x.reshape(N_HALF, 2)
                    pts2 = np.vstack([pts_half2, -pts_half2])
                    score2 = _score(pts2)
                    update_best(pts2, score2)
                    
        except Exception:
            pass

    # Phase 2: Global search using Differential Evolution on the symmetric space
    # DE is robust to local minima.
    if time.time() < deadline - 20.0:
        def de_obj(flat):
            return _feasibility_objective(flat, t_target=0.0, beta=30.0)
            
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-1.5, 1.5)] * (N_HALF * 2)
            res_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-5,
                polish=True,
                init='latinhypercube'
            )
            if res_de.fun < 0: # Valid score
                pts_half = res_de.x.reshape(N_HALF, 2)
                pts = np.vstack([pts_half, -pts_half])
                score = _score(pts)
                update_best(pts, score)
                
                # Refine DE result
                if time.time() < deadline - 5.0:
                    res_ref = minimize(
                        lambda x: _feasibility_objective(x, t_target=0.0, beta=100.0),
                        res_de.x,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    pts_half_ref = res_ref.x.reshape(N_HALF, 2)
                    pts_ref = np.vstack([pts_half_ref, -pts_half_ref])
                    score_ref = _score(pts_ref)
                    update_best(pts_ref, score_ref)
                    
        except Exception:
            pass

    # Phase 3: If we have a good candidate, try to break symmetry?
    # The optimal solution for N=14 might not be centrally symmetric.
    # However, breaking symmetry increases dimension to 28.
    # Let's try a short full-space optimization if we have time.
    if best_points is not None and time.time() < deadline - 15.0:
        flat_full = best_points.flatten()
        
        def _feasibility_objective_full(flat, beta=50.0):
            pts = flat.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            areas = _all_areas(pts)
            if np.min(areas) < 1e-12:
                return 1e10
            z = -beta * areas
            max_z = np.max(z)
            if max_z < -500:
                log_sum = max_z
            else:
                log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
            soft_min = -log_sum / beta
            hull_area = _compute_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            return -soft_min / hull_area

        try:
            res_full = minimize(
                lambda x: _feasibility_objective_full(x, beta=50.0),
                flat_full,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            pts_full = res_full.x.reshape(N, 2)
            score_full = _score(pts_full)
            update_best(pts_full, score_full)
            
        except Exception:
            pass

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
