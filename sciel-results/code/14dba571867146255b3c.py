import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_data(pts):
    """
    Compute convex hull and extract facet data needed for optimization.
    Returns:
        faces: array of indices (F, 3)
        A, B, C: arrays of vertex coordinates for each face (F, 3)
        r0: current max circumradius
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute circumradius for each face
        # For a triangle with sides a, b, c and area S, circumradius R = abc / (4S)
        # Or using vectors: R = ||A-B|| * ||B-C|| * ||C-A|| / (4 * ||(A-B) x (B-C)||)
        
        AB = B - A
        BC = C - B
        CA = A - C
        
        cross = np.cross(AB, BC)
        area_2 = np.linalg.norm(cross, axis=1) # 2 * Area
        
        # Filter degenerate faces
        valid = area_2 > _EPS
        if not np.any(valid):
            return None, None, None, None, np.pi
            
        A_v = A[valid]
        B_v = B[valid]
        C_v = C[valid]
        AB_v = AB[valid]
        BC_v = BC[valid]
        CA_v = CA[valid]
        area_2_v = area_2[valid]
        
        side_a = np.linalg.norm(AB_v, axis=1)
        side_b = np.linalg.norm(BC_v, axis=1)
        side_c = np.linalg.norm(CA_v, axis=1)
        
        # Circumradius R = abc / (4 * Area) = abc / (2 * area_2)
        r_vals = (side_a * side_b * side_c) / (2.0 * area_2_v)
        
        # Ensure normals point outward? 
        # The circumradius formula is invariant to orientation, so we don't strictly need outward normals 
        # for the radius calculation itself, but we need consistent vertices for the solver.
        
        return faces[valid], A_v, B_v, C_v, r_vals, np.max(r_vals)
    except Exception:
        return None, None, None, None, np.pi

def _residuals(pts_flat, target_r):
    """
    Residuals for least_squares.
    We want to minimize the variance of the circumradii around the target,
    or more directly, minimize the maximum. 
    Least squares minimizes sum of squares. 
    If we set residuals = r_i - target_r, it pushes all r_i to target_r.
    To handle the "max" nature, we can use a soft penalty or just rely on the 
    fact that minimizing the spread often reduces the max if the initial guess is good.
    
    A better approach for "minimize max" with least_squares:
    Minimize sum (r_i - R_target)^2 where R_target is updated? No, that's hard.
    
    Alternative: Minimize sum (r_i)^2? No, that minimizes average size, not max.
    
    Let's use the "excess" formulation:
    We want r_i <= R. 
    Residuals: max(0, r_i - R). But R is not a variable in standard LS.
    
    Common trick: Minimize sum (r_i - R_k)^2 where R_k is the current max? 
    This is not smooth.
    
    Better: Use the fact that at the optimum, many facets will have r_i = R_max.
    We can simply minimize the sum of squared circumradii, but weighted heavily by the larger ones?
    
    Actually, a very effective strategy for covering problems using LS is to minimize 
    the sum of squared "excess" distances. But we don't have a grid.
    
    Let's stick to the geometric property: The optimal covering tends to have 
    "equi-faceting" where the circumradii of all hull facets are equal (or nearly so).
    So, minimizing the variance of the circumradii is a strong heuristic.
    residuals = r_i - mean(r) ? No, that allows scaling (but sphere is fixed).
    
    Let's minimize sum (r_i - R_ref)^2 where R_ref is the current max.
    This pushes all facets to have the same radius as the largest one.
    If we can reduce the largest one, R_ref drops, and all follow.
    
    However, `least_squares` minimizes sum(residuals^2).
    If we define residuals as r_i, it minimizes sum r_i^2. This tends to make all facets smaller, 
    which is good, but might not balance them.
    
    Let's try: residuals = r_i - R_current_max.
    This forces all facets to match the current maximum. As the optimizer works, 
    if it can reduce the max, the target drops.
    """
    pts = pts_flat.reshape(N, 3)
    # Normalize to ensure they stay on sphere during optimization steps
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + _EPS)
    
    _, A_v, B_v, C_v, r_vals, max_r = _get_hull_data(pts)
    
    if r_vals is None:
        return np.ones(100) * 100.0 # Large penalty
        
    # Target is the current max radius
    # Residuals: r_i - max_r
    # This minimizes the spread. If the spread is 0, all r_i = max_r.
    # This doesn't explicitly push max_r down, but often configurations with equal facets 
    # are more "compact" and allow for global improvements via the solver's steps.
    # To explicitly push down, we could add a term for the mean radius.
    
    # Hybrid: Minimize sum (r_i - R_target)^2 + lambda * (mean(r) - R_target)^2?
    # Simpler: Just minimize sum r_i^2. This minimizes the "energy" of the facets.
    # In sphere covering, smaller facets generally imply smaller covering radius.
    # Let's try minimizing sum r_i^2.
    
    return r_vals

def _refit(pts, deadline):
    """
    Run least_squares to refine the configuration.
    """
    if time.time() > deadline - 5.0:
        return pts
        
    # Initial guess for bounds? We are on a sphere, so no bounds needed if we normalize.
    # But LS might step off the sphere. We normalize in residuals.
    
    # We need to pass the initial point as a flat array.
    x0 = pts.flatten()
    
    # Define a wrapper that normalizes and calls residuals
    def fun(x):
        return _residuals(x, 0.0)
        
    try:
        # Use 'trf' (Trust Region Reflective) which is robust.
        # method='lm' (Levenberg-Marquardt) is faster but doesn't support bounds.
        # Since we normalize, we don't strictly need bounds, but 'trf' is safer.
        res = least_squares(fun, x0, method='trf', max_nfev=50)
        
        if res.success or res.cost < 1e-10:
            new_pts = res.x.reshape(N, 3)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            return new_pts
    except Exception:
        pass
        
    return pts

def _lloyd_relax(pts, n_iters=10, grid_size=5000):
    """
    Fast Lloyd's relaxation to get out of bad local minima or fix bad initializations.
    """
    # Generate grid
    i = np.arange(grid_size) + 0.5
    z = 1.0 - 2.0 * i / grid_size
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    pts = pts.copy()
    for _ in range(n_iters):
        dots = grid @ pts.T
        nearest_idx = np.argmax(dots, axis=1)
        
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        # Efficient accumulation
        # Using bincount with weights for each dimension
        for dim in range(3):
            new_pts[:, dim] = np.bincount(nearest_idx, weights=grid[:, dim], minlength=N)
        counts = np.bincount(nearest_idx, minlength=N)
        
        valid = counts > 0
        new_pts[valid] /= (np.linalg.norm(new_pts[valid], axis=1, keepdims=True) + _EPS)
        new_pts[~valid] = pts[~valid]
        
        pts = new_pts
        
    return pts

def construct():
    deadline = time.time() + 140.0
    
    # Start with Fibonacci
    best_pts = _initial_fib()
    _, _, _, _, best_r = _get_hull_data(best_pts)
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Generate several diverse initial guesses (Fibonacci + Random perturbations).
    # 2. For each, run a short Lloyd's relaxation to settle into a local basin.
    # 3. Then run the geometric Least Squares polish to optimize the facet radii.
    # 4. Keep the best.
    
    n_restarts = 8
    time_per_restart = (deadline - 10.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        elif restart % 2 == 1:
            # Random perturbation of Fibonacci
            pts = _initial_fib()
            noise = rng.standard_normal((N, 3)) * 0.2
            pts += noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Pure random
            pts = rng.standard_normal((N, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Step 1: Quick Lloyd's to settle
        pts = _lloyd_relax(pts, n_iters=20, grid_size=10000)
        
        # Step 2: Geometric LS Polish
        # We might run LS multiple times with slightly different settings or just once.
        # LS can be slow. Let's limit function evaluations.
        pts = _refit(pts, deadline)
        
        # Step 3: Check improvement
        _, _, _, _, curr_r = _get_hull_data(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
        # Optional: Another round of Lloyd's on the LS result to clean up?
        # LS might leave points slightly suboptimal in terms of Voronoi balance.
        if time.time() < deadline - 15.0:
            pts = _lloyd_relax(pts, n_iters=10, grid_size=10000)
            _, _, _, _, curr_r2 = _get_hull_data(pts)
            if curr_r2 < best_r:
                best_r = curr_r2
                best_pts = pts.copy()
                
    # Final Polish: If time remains, run a very fine LS on the best point
    if time.time() < deadline - 5.0:
        best_pts = _refit(best_pts, deadline)
        _, _, _, _, final_r = _get_hull_data(best_pts)
        if final_r < best_r:
            best_r = final_r
            # best_pts is already updated by _refit return value if we assign it
            # Wait, _refit returns pts, so:
            # best_pts = _refit(best_pts, deadline) was done above.
            # Let's just ensure best_pts is the one with final_r
            # Actually _refit modifies local variable, let's re-fetch or ensure assignment.
            # In the loop above: pts = _refit(pts, deadline). 
            # In the final block: 
            polished = _refit(best_pts, deadline)
            _, _, _, _, final_r2 = _get_hull_data(polished)
            if final_r2 < best_r:
                best_r = final_r2
                best_pts = polished

    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
