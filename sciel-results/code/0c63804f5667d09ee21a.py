import numpy as np
import time
from scipy.optimize import linprog

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 such triples
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

def _min_triangle_area(pts):
    """
    pts: array of shape (N, 2) with Cartesian coordinates.
    Returns the minimum area of a triangle formed by any 3 points.
    """
    pi = pts[_triples[:, 0]]
    pj = pts[_triples[:, 1]]
    pk = pts[_triples[:, 2]]
    
    dx1 = pi[:, 0] - pk[:, 0]
    dy1 = pi[:, 1] - pk[:, 1]
    dx2 = pj[:, 0] - pk[:, 0]
    dy2 = pj[:, 1] - pk[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(pts):
    """
    Check if all points are inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    cond1 = y >= -1e-9
    cond2 = SQRT3 * x - y >= -1e-9
    cond3 = -SQRT3 * x - y + SQRT3 >= -1e-9
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def _is_time_up():
        return time.time() > deadline

    best_area = 0.0
    best_pts = None

    # --- Strategy: Geometric Decomposition + LP ---
    # The objective is to maximize the minimum area.
    # Area(i,j,k) = 0.5 * |det(P_i - P_k, P_j - P_k)|
    # This is a piecewise-linear function of the point coordinates.
    # We can solve for the maximum area by iterating over possible "active sets"
    # of constraints, but that's complex. 
    # 
    # Alternative: Use the fact that for a fixed set of signs for the determinants,
    # the problem becomes a Linear Program (LP).
    # However, the number of sign combinations is 2^165, which is too large.
    
    # Instead, we use a heuristic LP-based local search:
    # 1. Start with a random configuration.
    # 2. Identify the "critical" triples (those with the smallest area).
    # 3. Assume the signs of the determinants for all triples remain constant in a neighborhood.
    # 4. Solve an LP to maximize the minimum area under these sign assumptions and triangle constraints.
    # 5. If the solution is valid (signs didn't flip), it's a local optimum for that sign pattern.
    # 6. Perturb and repeat.

    # To make this robust, we will use a "warm start" LP approach.
    # Variables: x_1, y_1, ..., x_11, y_11, t
    # Maximize t
    # Subject to:
    # 1. Triangle constraints (3 linear inequalities per point)
    # 2. For each triple (i,j,k):
    #    s_{ijk} * (det(P_i - P_k, P_j - P_k)) >= 2*t
    #    where s_{ijk} is the sign of the determinant in the current configuration.
    #    Note: 2*t is used because Area = 0.5 * |det|, so |det| >= 2*Area.
    #          If we want Area >= t, then |det| >= 2t.

    # We will implement a function that solves this LP given a sign pattern.
    # If the LP solution violates the sign pattern, we discard it or try a different pattern.
    
    # To get diverse sign patterns, we will:
    # 1. Generate random points.
    # 2. Compute signs.
    # 3. Solve LP.
    # 4. If valid, update best.
    # 5. Perturb the solution and repeat.

    # LP Formulation:
    # Variables: v = [x_1, y_1, ..., x_11, y_11, t] (size 23)
    # Objective: minimize -t (index 22)
    # c = [0, 0, ..., 0, -1]
    
    # Constraints:
    # A_ub * v <= b_ub
    
    # Triangle Constraints for point p_i:
    # 1. y_i >= 0  =>  -y_i <= 0
    # 2. sqrt(3)x_i - y_i >= 0  =>  -sqrt(3)x_i + y_i <= 0
    # 3. -sqrt(3)x_i - y_i + sqrt(3) >= 0  =>  sqrt(3)x_i + y_i <= sqrt(3)
    
    # Area Constraints for triple (i,j,k):
    # det = (x_i - x_k)(y_j - y_k) - (y_i - y_k)(x_j - x_k)
    # det = x_i y_j - x_i y_k - x_k y_j + x_k y_k - y_i x_j + y_i x_k + y_k x_j - y_k x_k
    # This is BILINEAR, not linear!
    # 
    # WAIT. The area of a triangle formed by points is NOT a linear function of the point coordinates.
    # It is bilinear (quadratic).
    # Therefore, the LP approach is INVALID for this problem.
    # The problem is Non-Linear Programming (NLP) with non-convex objective.
    
    # --- Revised Strategy: Differential Evolution + Local Polish ---
    # Since LP is not applicable, and Nelder-Mead is slow, we use Differential Evolution (DE),
    # which is a robust global stochastic optimizer that handles non-smooth, non-convex landscapes well.
    # We will use scipy.optimize.differential_evolution.
    
    # DE is known to be slower per iteration but finds global optima better.
    # We will limit the population size and iterations to fit in the time budget.
    
    # Bounds for each coordinate:
    # x in [0, 1]
    # y in [0, H]
    # But we must ensure points are inside the triangle.
    # DE does not support complex constraints directly, so we use a penalty function.
    
    def objective_de(pts_flat):
        if _is_time_up():
            return 1e9
        pts = pts_flat.reshape(N, 2)
        
        # Check triangle constraints
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalties for being outside
        pen = 0.0
        # y >= 0
        pen += np.sum(np.maximum(0, -y)**2)
        # sqrt(3)x - y >= 0
        d2 = SQRT3 * x - y
        pen += np.sum(np.maximum(0, -d2)**2)
        # -sqrt(3)x - y + sqrt(3) >= 0
        d3 = -SQRT3 * x - y + SQRT3
        pen += np.sum(np.maximum(0, -d3)**2)
        
        if pen > 1e-6:
            return 1e6 + pen
            
        area = _min_triangle_area(pts)
        return -area

    # Bounds for DE
    # x: [0, 1]
    # y: [0, H]
    # This is a bounding box. The penalty handles the triangular constraint.
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, H))  # y

    # Run DE
    # popsize: 15 (default is 15)
    # maxiter: 100 (tune based on time)
    # tol: 1e-8
    
    try:
        from scipy.optimize import differential_evolution
        
        # DE can be slow. We will run it with a tight time limit.
        # We can pass a callback to check time.
        
        def callback(xk, convergence):
            if _is_time_up():
                raise ValueError("Time limit exceeded")

        res = differential_evolution(
            objective_de, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8,
            callback=callback,
            seed=42
        )
        
        if res.success or res.fun < 1e6:
            res_pts = res.x.reshape(N, 2)
            if _is_inside_triangle(res_pts):
                area = _min_triangle_area(res_pts)
                if area > best_area:
                    best_area = area
                    best_pts = res_pts.copy()
        
    except Exception:
        pass

    # If DE didn't finish or didn't find a good solution, fall back to Nelder-Mead on the best found
    # Or just use a few Nelder-Mead restarts if time remains.
    
    if not _is_time_up():
        # Use Nelder-Mead on the best point found by DE (if any) or a random point
        if best_pts is None:
            # Generate a random valid point
            bary = np.random.default_rng(42).dirichlet([1, 1, 1], size=N)
            x = bary[:, 1] + 0.5 * bary[:, 2]
            y = H * bary[:, 2]
            best_pts = np.column_stack([x, y])
            best_area = _min_triangle_area(best_pts)

        # Polish with Nelder-Mead
        def objective_nm(pts_flat):
            if _is_time_up():
                return 1e9
            pts = pts_flat.reshape(N, 2)
            area = _min_triangle_area(pts)
            # Penalty
            x = pts[:, 0]
            y = pts[:, 1]
            pen = np.sum(np.maximum(0, -y)**2) + \
                  np.sum(np.maximum(0, -(SQRT3 * x - y))**2) + \
                  np.sum(np.maximum(0, -(-SQRT3 * x - y + SQRT3))**2)
            return -area + pen * 1000.0

        try:
            res = minimize(objective_nm, best_pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            res_pts = res.x.reshape(N, 2)
            if _is_inside_triangle(res_pts):
                area = _min_triangle_area(res_pts)
                if area > best_area:
                    best_area = area
                    best_pts = res_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        bary = np.ones((N, 3)) / 3.0
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        best_pts = np.column_stack([x, y])
        
    return best_pts

# We need to import minimize for the fallback
from scipy.optimize import minimize
