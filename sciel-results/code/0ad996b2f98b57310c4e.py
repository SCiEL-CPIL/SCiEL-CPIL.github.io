"""Circle packing in unit square using Delaunay-based constrained optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def _delaunay_constraints(points):
    """
    Given an array of points (n, 2), return a list of edge indices (i, j)
    from the Delaunay triangulation. These are the pairs that need to be
    checked for non-overlap (plus boundary).
    """
    if len(points) < 3:
        return []
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = simplex[i]
                b = simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]

def _make_initial_guess(N):
    """
    Create a better initial guess using a hexagonal-like grid or jittered grid.
    """
    # Use a grid that is denser to allow for larger radii initially
    side = int(np.ceil(np.sqrt(N)))
    # If N is not a perfect square, we might have an incomplete last row
    # Let's just use a simple grid for now, but with some random jitter to break symmetry
    pts = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        # Add small jitter to avoid symmetric local minima
        x = (col + 0.5) / side + np.random.uniform(-0.01, 0.01)
        y = (row + 0.5) / side + np.random.uniform(-0.01, 0.01)
        # Clamp to [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        # Initial radius: small enough to fit
        r = 0.5 / side * 0.8
        pts[i] = [x, y, r]
    return pts

def construct():
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    # Initial guess
    pts = _make_initial_guess(N)
    x0 = pts.flatten()
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    best_obj = -np.sum(pts[:, 2]) # Maximize sum of radii -> Minimize negative
    best_x = x0.copy()
    
    # We'll do a few iterations of: solve with current edges, update edges
    max_outer_iters = 10
    
    for outer_iter in range(max_outer_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Re-compute edges based on current best solution
        centers_cur = best_x.reshape(N, 3)[:, :2]
        edges = _delaunay_constraints(centers_cur)
        
        # Define vectorized objective
        def objective(x):
            radii = x[2::3]
            return -np.sum(radii)
        
        # Define vectorized constraints
        # SLSQP expects a list of constraints, but a single constraint with 'fun' returning an array is efficient.
        # However, SLSQP documentation says 'fun' should return an array of constraint values.
        # Type 'ineq' means fun(x) >= 0.
        
        def constraints_func(x):
            pts = x.reshape(N, 3)
            centers = pts[:, :2]
            radii = pts[:, 2]
            
            # Boundary constraints:
            # x_i - r_i >= 0
            # y_i - r_i >= 0
            # 1 - x_i - r_i >= 0
            # 1 - y_i - r_i >= 0
            c_left = centers[:, 0] - radii
            c_bottom = centers[:, 1] - radii
            c_right = 1.0 - centers[:, 0] - radii
            c_top = 1.0 - centers[:, 1] - radii
            
            # Non-overlap constraints for Delaunay edges:
            # dist(i,j) - r_i - r_j >= 0
            c_nonoverlap = []
            for (i, j) in edges:
                dist = np.linalg.norm(centers[i] - centers[j])
                c_nonoverlap.append(dist - radii[i] - radii[j])
            
            # Combine all constraints into a single array
            # If any constraint is negative, it's violated.
            # We return the array of values. SLSQP will check if all are >= 0.
            if c_nonoverlap:
                return np.concatenate([c_left, c_bottom, c_right, c_top, c_nonoverlap])
            else:
                return np.concatenate([c_left, c_bottom, c_right, c_top])
        
        cons = ({'type': 'ineq', 'fun': constraints_func},)
        
        # Run SLSQP
        try:
            result = minimize(
                objective,
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.fun < best_obj:
                # Check if solution is valid (all constraints satisfied)
                x_sol = result.x
                pts_sol = x_sol.reshape(N, 3)
                centers_sol = pts_sol[:, :2]
                radii_sol = pts_sol[:, 2]
                
                valid = True
                
                # Check boundaries
                if np.any(centers_sol - radii_sol < -1e-6) or np.any(1.0 - centers_sol - radii_sol < -1e-6):
                    valid = False
                
                # Check all pairwise non-overlap (not just Delaunay edges) to be safe
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = np.linalg.norm(centers_sol[i] - centers_sol[j])
                            if radii_sol[i] + radii_sol[j] > dist + 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    best_obj = result.fun
                    best_x = x_sol.copy()
                else:
                    # If invalid, we might try to shrink radii slightly to make it valid
                    # But usually SLSQP respects constraints. If it doesn't, we keep old best.
                    pass
                    
        except Exception:
            pass
    
    # Final solution
    pts_final = best_x.reshape(N, 3)
    
    # Ensure validity by clamping radii if needed
    for i in range(N):
        xi, yi, ri = pts_final[i]
        # Clamp radius to fit within boundaries
        ri = min(ri, xi, yi, 1 - xi, 1 - yi)
        ri = max(ri, 1e-6)
        pts_final[i] = [xi, yi, ri]
    
    # Final pairwise check and shrink if overlapping
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = pts_final[i]
            xj, yj, rj = pts_final[j]
            dist = np.linalg.norm([xi - xj, yi - yj])
            if ri + rj > dist - 1e-8:
                # Shrink both radii slightly
                overlap = ri + rj - dist
                shrink = overlap / 2 + 1e-6
                ri = max(ri - shrink, 1e-6)
                rj = max(rj - shrink, 1e-6)
                pts_final[i, 2] = ri
                pts_final[j, 2] = rj
    
    # Final boundary check
    for i in range(N):
        xi, yi, ri = pts_final[i]
        ri = min(ri, xi - 1e-6, yi - 1e-6, 1 - xi - 1e-6, 1 - yi - 1e-6)
        ri = max(ri, 1e-6)
        pts_final[i, 2] = ri
    
    return pts_final
# EVOLVE-BLOCK-END
