"""
Circle packing in unit square.
Uses trust-constr NLP solver with hard constraints for non-overlap and boundaries.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

# Deadline guard: The evaluator kills after 150s. We set a strict internal deadline.
# We need to finish well before then to allow for Python overhead.
DEADLINE = time.time() + 140.0

def _time_left():
    return DEADLINE - time.time()

def _hexagonal_init(n):
    """
    Generate a hexagonal lattice initialization.
    This is a much denser and more regular starting point than a square grid.
    """
    # Approximate optimal packing density is ~0.9069 for infinite plane.
    # For N circles in a unit square, average radius r ~ sqrt(0.9069 / (pi * N))
    # But we need to fit them in the square.
    
    # A common heuristic for initial radius in hex packing:
    # Side length of hexagon = 2*r.
    # Distance between centers = 2*r.
    # Horizontal spacing = 2*r.
    # Vertical spacing = 2*r * sqrt(3)/2 = r * sqrt(3).
    
    # Let's estimate r such that a grid of this fits in [0,1]^2.
    # If we have k columns and m rows.
    # Width = (k-1)*2r + 2r = 2kr. So k ~ 1/(2r).
    # Height = (m-1)*r*sqrt(3) + 2r.
    
    # Let's try to fit a hex grid of roughly sqrt(N) x sqrt(N) points.
    # For N=26, maybe 5x5 or 6x5.
    
    # We will generate a large hex grid and take the first N points, 
    # then scale/translate to fit in [0,1]^2 with some margin.
    
    # Generate points
    points = []
    r_est = 0.15 # Initial guess for scaling
    
    # Generate a dense hex grid in a larger area
    cols = 10
    rows = 10
    for i in range(rows):
        for j in range(cols):
            x = j * 2.0
            y = i * np.sqrt(3)
            if i % 2 == 1:
                x += 1.0
            points.append((x, y))
            
    points = np.array(points)
    
    # Sort by radius from center to keep the "core" of the hex grid
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    dists = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    # Now scale and translate these points to fit in [0, 0.95]^2 (leaving room for radii)
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()
    
    # Calculate scale factor
    # We want the spread to be less than 1.0
    # Let's target a spread of 0.8 to allow for radii up to ~0.1
    target_spread = 0.8
    scale = target_spread / max(max_x - min_x, max_y - min_y, 1e-6)
    
    scaled_x = (points[:, 0] - min_x) * scale
    scaled_y = (points[:, 1] - min_y) * scale
    
    # Center in the square
    offset_x = (1.0 - (max(scaled_x) - min(scaled_x))) / 2.0
    offset_y = (1.0 - (max(scaled_y) - min(scaled_y))) / 2.0
    
    centers = np.column_stack([scaled_x + offset_x, scaled_y + offset_y])
    
    # Estimate radii based on local spacing
    # For a hex grid, nearest neighbor distance is d. r = d/2.
    # Our scale factor maps original distance 'd_orig' to 'd_scaled = d_orig * scale'.
    # Original nearest neighbor dist was 2.0 (horizontal) or sqrt(3) (vertical).
    # Let's just assign a uniform radius based on the target spread.
    # If we fit ~5x5 points in 0.8 width, spacing is ~0.16. r ~ 0.08.
    r_init = 0.08 * scale * 2.0 # Rough estimate
    
    # Ensure r_init is reasonable
    r_init = min(r_init, 0.15)
    
    xyz = np.zeros((n, 3))
    xyz[:, 0] = centers[:, 0]
    xyz[:, 1] = centers[:, 1]
    xyz[:, 2] = r_init
    
    return xyz

def _objective(xyz):
    """
    Objective function: Minimize negative sum of radii.
    """
    return -np.sum(xyz[:, 2])

def _constraints(xyz):
    """
    Return list of constraint dictionaries for scipy.optimize.minimize (trust-constr).
    Each constraint is a NonlinearConstraint or LinearConstraint.
    """
    n = len(xyz)
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    cons = []
    
    # 1. Boundary Constraints
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # We can define these as linear constraints or nonlinear.
    # Linear is faster.
    # Let's use LinearConstraint for boundaries.
    # A @ v >= b
    # v = [x0, y0, r0, x1, y1, r1, ...]
    
    A_bound = np.zeros((4*n, 3*n))
    b_lower = np.zeros(4*n)
    b_upper = np.inf * np.ones(4*n)
    
    for i in range(n):
        # x_i - r_i >= 0
        A_bound[4*i, i*3] = 1.0
        A_bound[4*i, i*3+2] = -1.0
        
        # 1 - x_i - r_i >= 0  =>  -x_i - r_i >= -1
        A_bound[4*i+1, i*3] = -1.0
        A_bound[4*i+1, i*3+2] = -1.0
        b_lower[4*i+1] = -1.0
        
        # y_i - r_i >= 0
        A_bound[4*i+2, i*3+1] = 1.0
        A_bound[4*i+2, i*3+2] = -1.0
        
        # 1 - y_i - r_i >= 0 => -y_i - r_i >= -1
        A_bound[4*i+3, i*3+1] = -1.0
        A_bound[4*i+3, i*3+2] = -1.0
        b_lower[4*i+3] = -1.0
        
    cons.append((A_bound, b_lower, b_upper))
    
    # 2. Non-overlap Constraints
    # For each pair (i, j) with i < j:
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # This is a nonlinear constraint.
    # We define g(x) = (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # scipy trust-constr expects constraints in the form:
    # slb <= fun(x) <= ub
    # So we set slb = 0, ub = inf
    
    # We need to provide a function that returns the vector of all g(x) values.
    # And optionally Jacobian for speed.
    
    num_pairs = n * (n - 1) // 2
    
    def nonoverlap_fun(x_vec):
        """
        x_vec is flat vector of length 3*n.
        Returns array of length num_pairs.
        """
        xyz_arr = x_vec.reshape(n, 3)
        xi = xyz_arr[:, 0]
        yi = xyz_arr[:, 1]
        ri = xyz_arr[:, 2]
        
        g_vals = np.zeros(num_pairs)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                dist_sq = dx*dx + dy*dy
                sum_r = ri[i] + ri[j]
                g_vals[idx] = dist_sq - sum_r*sum_r
                idx += 1
        return g_vals
        
    # Jacobian of nonoverlap_fun
    # d/dx_i (dx^2 + dy^2 - (ri+rj)^2) = 2*dx
    # d/dy_i ... = 2*dy
    # d/dri ... = -2*(ri+rj)
    # d/dx_j ... = -2*dx
    # d/dy_j ... = -2*dy
    # d/drj ... = -2*(ri+rj)
    
    def nonoverlap_jac(x_vec):
        xyz_arr = x_vec.reshape(n, 3)
        xi = xyz_arr[:, 0]
        yi = xyz_arr[:, 1]
        ri = xyz_arr[:, 2]
        
        J = np.zeros((num_pairs, 3*n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                sum_r = ri[i] + ri[j]
                
                # Row idx corresponds to pair (i, j)
                J[idx, i*3] = 2 * dx
                J[idx, i*3+1] = 2 * dy
                J[idx, i*3+2] = -2 * sum_r
                
                J[idx, j*3] = -2 * dx
                J[idx, j*3+1] = -2 * dy
                J[idx, j*3+2] = -2 * sum_r
                
                idx += 1
        return J
        
    cons.append((nonoverlap_fun, 0.0, np.inf * np.ones(num_pairs), nonoverlap_jac))
    
    return cons

def _solve_nlp(xyz_init, max_time):
    """
    Solve the NLP problem using trust-constr.
    """
    n = len(xyz_init)
    x0 = xyz_init.flatten()
    
    # Bounds:
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Convert to scipy Bounds object
    from scipy.optimize import Bounds
    lb = np.zeros(3*n)
    ub = np.zeros(3*n)
    for i in range(n):
        lb[i*3] = 0.0
        ub[i*3] = 1.0
        lb[i*3+1] = 0.0
        ub[i*3+1] = 1.0
        lb[i*3+2] = 0.0
        ub[i*3+2] = 0.5
        
    bounds_obj = Bounds(lb, ub)
    
    # Prepare constraints
    cons = _constraints(xyz_init)
    
    # Format constraints for trust-constr
    # It expects a list of NonlinearConstraint or LinearConstraint objects
    from scipy.optimize import NonlinearConstraint, LinearConstraint
    
    trust_cons = []
    
    # First constraint: Linear Boundaries
    A, lb_lin, ub_lin = cons[0]
    trust_cons.append(LinearConstraint(A, lb_lin, ub_lin))
    
    # Second constraint: Nonlinear Non-overlap
    fun_no, lb_no, ub_no, jac_no = cons[1]
    trust_cons.append(NonlinearConstraint(fun_no, lb_no, ub_no, jac=jac_no))
    
    t0 = time.time()
    
    try:
        res = minimize(
            _objective,
            x0,
            method='trust-constr',
            bounds=bounds_obj,
            constraints=trust_cons,
            options={
                'maxiter': 500,
                'gtol': 1e-8,
                'xtol': 1e-10,
                'disp': False
            }
        )
    except Exception as e:
        print(f"NLP Solver Error: {e}")
        return xyz_init
        
    if time.time() - t0 > max_time:
        return xyz_init
        
    if res.success or (hasattr(res, 'x') and res.fun < _objective(xyz_init)):
        new_xyz = res.x.reshape(n, 3)
        # Clip to ensure strict validity
        new_xyz[:, 0] = np.clip(new_xyz[:, 0], 0, 1)
        new_xyz[:, 1] = np.clip(new_xyz[:, 1], 0, 1)
        new_xyz[:, 2] = np.clip(new_xyz[:, 2], 0, 0.5)
        return new_xyz
    else:
        return xyz_init

def _is_valid(xyz, tol=1e-5):
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    n = len(x)
    
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _fix_violations(xyz):
    """
    Heuristic repair if the solver returns slightly invalid points.
    """
    xyz = xyz.copy()
    for _ in range(10):
        changed = False
        # Boundary fix
        for i in range(len(xyz)):
            x, y, r = xyz[i]
            if x < r:
                xyz[i, 2] = x * 0.9
                changed = True
            if x > 1 - r:
                xyz[i, 2] = (1 - x) * 0.9
                changed = True
            if y < r:
                xyz[i, 2] = min(xyz[i, 2], y * 0.9)
                changed = True
            if y > 1 - r:
                xyz[i, 2] = min(xyz[i, 2], (1 - y) * 0.9)
                changed = True
            xyz[i, 2] = max(xyz[i, 2], 0.0)
            
        # Overlap fix: shrink smaller circle
        for i in range(len(xyz)):
            for j in range(i + 1, len(xyz)):
                dx = xyz[i, 0] - xyz[j, 0]
                dy = xyz[i, 1] - xyz[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                req = xyz[i, 2] + xyz[j, 2]
                if d < req:
                    # Shrink the one with larger radius less, or just split the error
                    # Simple: shrink both by half the overlap
                    overlap = req - d
                    if overlap > 0:
                        shrink = overlap / 2.0
                        xyz[i, 2] -= shrink
                        xyz[j, 2] -= shrink
                        xyz[i, 2] = max(xyz[i, 2], 0.0)
                        xyz[j, 2] = max(xyz[j, 2], 0.0)
                        changed = True
        if not changed:
            break
    return xyz

def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    # Initial guess: Hexagonal
    xyz = _hexagonal_init(N)
    
    # Ensure initial guess is valid-ish to help the solver
    xyz = _fix_violations(xyz)
    
    best_xyz = xyz.copy()
    best_score = -np.sum(xyz[:, 2])
    
    # Run NLP solver
    # We allocate most of the time to this single high-quality solve.
    # trust-constr can be slow, so we give it a generous budget but monitor it.
    max_time = 100.0 # Leave 40s buffer
    
    if _time_left() > max_time:
        candidate = _solve_nlp(best_xyz, max_time)
        
        # If the solver failed to improve or produced invalid, try a perturbation
        if _is_valid(candidate, tol=1e-4):
            cand_score = -np.sum(candidate[:, 2])
            if cand_score < best_score: # Minimization
                best_xyz = candidate
                best_score = cand_score
        else:
            # Try to fix it
            candidate = _fix_violations(candidate)
            if _is_valid(candidate, tol=1e-4):
                cand_score = -np.sum(candidate[:, 2])
                if cand_score < best_score:
                    best_xyz = candidate
                    best_score = cand_score
    
    # If we have time left, do a second pass from the best solution
    # Or a different initial guess
    if _time_left() > 20.0:
        # Perturb the best solution
        rng = np.random.RandomState(42)
        perturb = rng.normal(0, 0.01, size=best_xyz.shape)
        perturb[:, 2] = rng.normal(0, 0.005, size=N) # Small radius perturbation
        xyz2 = best_xyz + perturb
        xyz2 = _fix_violations(xyz2)
        
        if _is_valid(xyz2, tol=1e-4):
            candidate2 = _solve_nlp(xyz2, _time_left() - 10.0)
            if _is_valid(candidate2, tol=1e-4):
                cand_score2 = -np.sum(candidate2[:, 2])
                if cand_score2 < best_score:
                    best_xyz = candidate2
                    best_score = cand_score2
    
    # Final cleanup
    best_xyz = _fix_violations(best_xyz)
    
    # Safety check: if invalid, return a known valid grid
    if not _is_valid(best_xyz, tol=1e-5):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        grid = np.zeros((N, 3))
        r = 0.5 / side - 0.01
        for i in range(N):
            row = i // side
            col = i % side
            grid[i, 0] = (col + 0.5) / side
            grid[i, 1] = (row + 0.5) / side
            grid[i, 2] = r
        best_xyz = grid
        
    return best_xyz

# EVOLVE-BLOCK-END
