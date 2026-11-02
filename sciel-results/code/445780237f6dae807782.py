"""Circle packing in unit square using Delaunay-based constrained optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def _get_delaunay_edges(centers):
    """
    Get Delaunay edges from centers.
    Returns list of (i, j) pairs.
    """
    if len(centers) < 3:
        return []
    try:
        tri = Delaunay(centers)
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
        return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]


def _make_hexagonal_initial(N):
    """
    Create a hexagonal-like initial arrangement.
    """
    pts = np.zeros((N, 3))
    
    # Hexagonal packing: rows offset
    row_spacing = 0.866  # sqrt(3)/2
    col_spacing = 1.0
    
    # Try to fit in unit square
    # We'll use a grid that approximates hexagonal
    side = int(np.ceil(np.sqrt(N * 0.9)))  # slightly less dense
    if side < 2:
        side = 2
    
    idx = 0
    for row in range(5):
        for col in range(5):
            if idx >= N:
                break
            # Hexagonal offset
            x = (col + 0.5) * col_spacing / 5
            y = (row + 0.5) * row_spacing / 5
            if row % 2 == 1:
                x += 0.5 * col_spacing / 5
            
            # Clamp to [0.05, 0.95]
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Initial radius
            r = 0.08
            
            pts[idx, 0] = x
            pts[idx, 1] = y
            pts[idx, 2] = r
            idx += 1
    
    # If we didn't fill all, add remaining in corners
    if idx < N:
        corners = [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
        for i in range(idx, N):
            c = corners[(i - idx) % 4]
            pts[i, 0] = c[0]
            pts[i, 1] = c[1]
            pts[i, 2] = 0.05
    
    return pts


def _make_grid_initial(N):
    """
    Create a grid initial arrangement.
    """
    pts = np.zeros((N, 3))
    side = int(np.ceil(np.sqrt(N)))
    
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = 0.5 / side * 0.85
        
        pts[i, 0] = x
        pts[i, 1] = y
        pts[i, 2] = r
    
    return pts


def _validate_solution(pts):
    """
    Validate that all circles are non-overlapping and inside the unit square.
    Returns (is_valid, max_violation).
    """
    n = len(pts)
    max_violation = 0.0
    
    # Check boundaries
    for i in range(n):
        xi, yi, ri = pts[i]
        if ri - xi > max_violation:
            max_violation = ri - xi
        if ri - yi > max_violation:
            max_violation = ri - yi
        if xi + ri - 1 > max_violation:
            max_violation = xi + ri - 1
        if yi + ri - 1 > max_violation:
            max_violation = yi + ri - 1
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = pts[i]
            xj, yj, rj = pts[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            violation = ri + rj - dist
            if violation > max_violation:
                max_violation = violation
    
    is_valid = max_violation < 1e-6
    return is_valid, max_violation


def _fix_solution(pts):
    """
    Fix a solution by shrinking radii to resolve overlaps.
    """
    n = len(pts)
    pts = pts.copy()
    
    # Iteratively fix overlaps
    for iteration in range(50):
        fixed = True
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = pts[i]
                xj, yj, rj = pts[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                
                if ri + rj > dist - 1e-8:
                    overlap = ri + rj - dist
                    # Shrink the smaller radius more
                    if ri <= rj:
                        shrink_i = overlap * 0.6 + 1e-7
                        shrink_j = overlap * 0.4 + 1e-7
                    else:
                        shrink_i = overlap * 0.4 + 1e-7
                        shrink_j = overlap * 0.6 + 1e-7
                    
                    new_ri = max(ri - shrink_i, 1e-6)
                    new_rj = max(rj - shrink_j, 1e-6)
                    
                    pts[i, 2] = new_ri
                    pts[j, 2] = new_rj
                    fixed = False
        
        # Check boundaries
        for i in range(n):
            xi, yi, ri = pts[i]
            max_r = min(xi, yi, 1 - xi, 1 - yi)
            if ri > max_r:
                pts[i, 2] = max(max_r - 1e-6, 1e-6)
                fixed = False
        
        if fixed:
            break
    
    return pts


def construct():
    start_time = time.time()
    time_limit = 110  # leave some margin
    
    # Try multiple initializations
    best_obj = np.inf
    best_x = None
    
    initializations = [
        _make_hexagonal_initial(N),
        _make_grid_initial(N),
    ]
    
    # Add a random initialization
    np.random.seed(42)
    random_pts = np.zeros((N, 3))
    for i in range(N):
        random_pts[i, 0] = np.random.uniform(0.1, 0.9)
        random_pts[i, 1] = np.random.uniform(0.1, 0.9)
        random_pts[i, 2] = 0.05
    initializations.append(random_pts)
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit:
            break
        
        x0 = init_pts.flatten()
        centers0 = init_pts[:, :2]
        edges = _get_delaunay_edges(centers0)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((1e-6, 0.5))  # r
        
        # Objective function
        def obj_func(x):
            pts = x.reshape(N, 3)
            return -np.sum(pts[:, 2])
        
        # Build constraints
        constraints = []
        
        # Boundary constraints
        for i in range(N):
            i3 = 3 * i
            constraints.append({'type': 'ineq', 'fun': lambda x, i3=i3: x[i3+2] - x[i3]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i3=i3: x[i3+2] - x[i3+1]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i3=i3: x[i3] + x[i3+2] - 1})
            constraints.append({'type': 'ineq', 'fun': lambda x, i3=i3: x[i3+1] + x[i3+2] - 1})
        
        # Non-overlap for Delaunay edges
        for (i, j) in edges:
            i3 = 3 * i
            j3 = 3 * j
            def _nonoverlap(x, i3=i3, j3=j3):
                xi, yi = x[i3], x[i3+1]
                xj, yj = x[j3], x[j3+1]
                ri = x[i3+2]
                rj = x[j3+2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return ri + rj - dist
            constraints.append({'type': 'ineq', 'fun': _nonoverlap})
        
        # Run SLSQP
        try:
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 400, 'ftol': 1e-9, 'disp': False}
            )
            
            if result.fun < best_obj:
                x_sol = result.x
                pts_sol = x_sol.reshape(N, 3)
                
                # Validate
                valid, _ = _validate_solution(pts_sol)
                
                if valid:
                    best_obj = result.fun
                    best_x = x_sol.copy()
                else:
                    # Try to fix
                    pts_fixed = _fix_solution(pts_sol)
                    obj_fixed = -np.sum(pts_fixed[:, 2])
                    valid_fixed, _ = _validate_solution(pts_fixed)
                    if valid_fixed and obj_fixed < best_obj:
                        best_obj = obj_fixed
                        best_x = pts_fixed.flatten().copy()
                        
        except Exception:
            pass
    
    # If we didn't get a good solution, use the best we have
    if best_x is None:
        best_x = initializations[0].flatten()
    
    # Final solution
    pts_final = best_x.reshape(N, 3)
    
    # Ensure validity
    pts_final = _fix_solution(pts_final)
    
    # Final validation
    valid, _ = _validate_solution(pts_final)
    if not valid:
        # Last resort: shrink all radii
        scale = 0.9
        pts_final[:, 2] *= scale
        pts_final = _fix_solution(pts_final)
    
    return pts_final
# EVOLVE-BLOCK-END
