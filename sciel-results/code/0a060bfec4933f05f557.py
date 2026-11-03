import time
import numpy as np
from scipy.optimize import minimize
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

def _compute_hull_area(points):
    try:
        # Check for degenerate cases
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
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _objective_slsqp(x):
    """
    Objective: Minimize -t
    Constraints: 
    1. For all triples (i,j,k): Area(i,j,k) >= t * Area(Hull)
    2. Points must be distinct (handled by bounds or penalty if needed, but SLSQP handles inequality constraints well)
    
    x layout: [x0, y0, ..., x13, y13, t]
    """
    pts = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Penalty for invalid points
    if not np.all(np.isfinite(pts)):
        return 1e10
    if np.any(pts < -1.1) or np.any(pts > 1.1):
        return 1e10
        
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
        
    areas = _all_areas(pts)
    
    # We want to maximize t such that min(areas) >= t * hull_area
    # Equivalent to min(areas) / hull_area >= t
    # Or: areas / hull_area - t >= 0 for all triples
    
    # To make SLSQP efficient, we don't want 364 constraints if possible, 
    # but SLSQP can handle them. However, providing gradients helps.
    # Let's use the constraint form: areas[i] - t * hull_area >= 0
    
    # If t is negative, it's trivially satisfied, but we want to maximize t.
    # We minimize -t.
    return -t

def _constraints_slsqp(x):
    """
    Returns list of constraint dictionaries.
    Each constraint: {'type': 'ineq', 'fun': lambda x: ...}
    fun(x) >= 0
    """
    pts = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # We cannot easily compute hull area derivative analytically in a simple way for all cases,
    # but we can evaluate the constraint numerically.
    # To save time, we might only constrain the "active" set or all of them.
    # Given N=14, 364 constraints is manageable for SLSQP if we don't call it too often.
    
    # However, SLSQP might struggle with 364 non-smooth constraints (abs in area).
    # Area is |cross product|. Cross product is smooth. Abs is not.
    # We can remove abs by noting that for a fixed orientation of the hull, 
    # we can assume points are in CCW order? No, points are not ordered.
    # But we can use the fact that Area = 0.5 * |det|.
    # det is linear in each point.
    
    # A better approach for SLSQP:
    # Maximize t
    # Subject to: 0.5 * |det(pj-pi, pk-pi)| - t * Area(Hull) >= 0
    # This is still non-smooth due to abs and Area(Hull).
    
    # Alternative: Use a smooth approximation or just rely on numerical derivatives?
    # SLSQP uses numerical derivatives by default if jac is not provided.
    # With 364 constraints, numerical derivatives are expensive (364 * 29 evaluations per step).
    # That's ~10k evaluations per step. Too slow.
    
    # We need a different strategy.
    # Let's use a "soft" min but with a very high power, or use a different optimizer.
    # Actually, the inspiration 1 mentioned SLSQP. How?
    # Maybe by fixing the hull?
    # If we fix the hull to be a unit circle or square, we remove the Area(Hull) dependency.
    # But the task says "normalized to unit area" and "score is min area / hull area".
    # If we constrain points to be inside a unit circle, and maximize min area,
    # the hull area will be <= 1.
    # Score = min_area / hull_area.
    # If we maximize min_area, we implicitly want a large hull area too.
    
    # Let's try a different structural change:
    # Use `scipy.optimize.differential_evolution` with a better objective function
    # that is smoother and guides the search more effectively.
    # The current program uses DE but with a simple score.
    # The inspiration 1 used SLSQP. Let's try to implement SLSQP by fixing the hull area to 1.
    # We can constrain the convex hull to have area 1? No, that's hard.
    
    # Let's try a different representation: Polar coordinates around the centroid?
    # Or, let's try **Basin Hopping** with a very good local optimizer (L-BFGS-B on soft-min).
    
    # The directive says: "COMPUTE BUDGET... STAGNATION... STRUCTURAL change".
    # Current: DE -> NM/Powell/L-BFGS.
    # New: **Basin Hopping** with **L-BFGS-B** on a **Soft-Min** objective, 
    # starting from diverse initial points (including regular polygon, random, etc).
    # Basin Hopping is robust for global optimization and can escape local minima 
    # better than DE for this type of landscape.
    
    pass # Placeholder, will implement below

def _objective_soft_min(x, beta=100.0):
    """
    Smooth approximation of -min(areas) / hull_area.
    We want to maximize min(areas)/hull_area.
    Minimize -log(sum(exp(-beta * areas / hull_area))) / beta  (Soft Min)
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    areas = _all_areas(pts)
    # Normalize areas by hull area to make the problem scale invariant
    # This is important because the score is ratio.
    normalized_areas = areas / hull_area
    
    # Soft min of normalized areas
    z = -beta * normalized_areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Generate diverse initial points
    def _generate_initials():
        inits = []
        # 1. Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # 2. Two concentric circles (7 each)
        angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles2 = angles1 + np.pi / 7
        r1, r2 = 0.5, 1.0
        pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        inits.append(np.vstack([pts1, pts2]))
        
        # 3. Hexagonal grid subset
        # Generate a grid and pick 14 points
        grid_x = np.arange(-2, 3)
        grid_y = np.arange(-2, 3)
        gx, gy = np.meshgrid(grid_x, grid_y)
        pts_grid = np.column_stack((gx.ravel(), gy.ravel()))
        # Pick 14 random points from grid
        idx = np.random.choice(len(pts_grid), N, replace=False)
        inits.append(pts_grid[idx])
        
        # 4. Random points in unit circle
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        inits.append(np.column_stack((radii * np.cos(angles), radii * np.sin(angles))))
        
        # 5. Random points in square
        inits.append(np.random.uniform(-1, 1, (N, 2)))
        
        return inits

    # Basin Hopping with L-BFGS-B refinement
    def _basin_hopping_step(x, step_size, beta, maxiter_lbfgs=200):
        """
        Perform a basin hopping step:
        1. Random perturbation
        2. Local refinement with L-BFGS-B
        """
        x_perturbed = x + step_size * np.random.randn(len(x))
        # Keep points within bounds
        x_perturbed = np.clip(x_perturbed, -1.5, 1.5)
        
        try:
            res = minimize(
                lambda x: _objective_soft_min(x, beta=beta),
                x_perturbed,
                method='L-BFGS-B',
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            return pts, score, res.fun
        except Exception:
            return None, -np.inf, np.inf

    # Main Loop: Manual Basin Hopping
    # We will run multiple "chains" with different betas and step sizes
    
    inits = _generate_initials()
    
    # Parameters for Basin Hopping
    betas = [20.0, 50.0, 100.0, 200.0]
    step_sizes = [0.5, 0.3, 0.1]
    
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        update_best(current_pts, current_score)
        
        # Try different betas
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            # Perform several random steps
            for step_size in step_sizes:
                if time.time() > deadline - 2:
                    break
                    
                # Do a few random walks
                for _ in range(3):
                    if time.time() > deadline - 1:
                        break
                        
                    new_pts, new_score, _ = _basin_hopping_step(
                        current_pts.flatten(), step_size, beta, maxiter_lbfgs=100
                    )
                    
                    if new_pts is not None:
                        # Accept if better (greedy) or with probability (Metropolis)
                        # For simplicity and speed, use greedy acceptance for now,
                        # but occasionally accept worse to escape local minima?
                        # Basin Hopping usually uses Metropolis.
                        # Let's use a simple Metropolis criterion.
                        if new_score > current_score:
                            current_pts = new_pts
                            current_score = new_score
                            update_best(current_pts, current_score)
                        else:
                            # Accept worse with probability exp(-(E_new - E_old)/T)
                            # Here E is -score (energy)
                            delta_e = -new_score - (-current_score)
                            T = 0.1 # Temperature
                            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                                current_pts = new_pts
                                current_score = new_score
                                update_best(current_pts, current_score)
                    else:
                        break

    # Final Polish with higher precision
    if best_points is not None and time.time() < deadline - 2.0:
        # Use Powell or NM for final polish on the best point found
        flat = best_points.flatten()
        try:
            res = minimize(
                lambda x: _objective_soft_min(x, beta=500.0),
                flat,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            update_best(pts, score)
        except Exception:
            pass
        
        # One more NM polish
        if time.time() < deadline - 0.5:
            try:
                res = minimize(
                    lambda x: _objective_soft_min(x, beta=1000.0),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts = res.x.reshape(N, 2)
                score = _score(pts)
                update_best(pts, score)
            except Exception:
                pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
