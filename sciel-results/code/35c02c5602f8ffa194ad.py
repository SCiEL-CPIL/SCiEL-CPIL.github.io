import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 21

def _is_valid(points, radii, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0 or h <= 0:
        return False
    if w + h > 2.0 + tol:
        return False
    for i in range(len(points)):
        x, y = points[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > w + tol or y - r < -tol or y + r > h + tol:
            return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _compute_inradii(points, w, h):
    """Compute inradii for each circle based on Delaunay triangulation and walls."""
    n = len(points)
    inradii = np.zeros(n)
    
    # Add wall points for Delaunay triangulation
    # We need to include the rectangle boundaries in the triangulation
    # To do this, we can add points far outside, but that complicates inradius calculation.
    # Alternative: Use the standard Delaunay of the points, and for each triangle,
    # calculate inradius. For points on the boundary, the inradius is limited by the wall.
    
    # Let's use a simpler heuristic:
    # 1. Delaunay triangulation of the points.
    # 2. For each triangle (i, j, k), compute the inradius.
    # 3. For each point, its max radius is the min of:
    #    - Distance to nearest wall
    #    - Min inradius of all triangles containing that point
    
    # Actually, a more robust way for "maximal packing" is to assume all circles are tangent.
    # The inradius of a triangle formed by 3 tangent circles is 0.
    # Instead, we should maximize the minimum distance to neighbors and walls.
    
    # Let's use the "Voronoi" cell inradius approach, which is equivalent to the largest empty circle centered at the point.
    # For a point p, the max radius is min(
    #    p[0], w - p[0], p[1], h - p[1],
    #    min over all other points q of (dist(p, q) / 2)  <-- This is only if we assume equal sizes or specific contact.
    # )
    # This is not quite right for variable radii.
    
    # Correct approach for variable radii in Delaunay context:
    # The "inradius" of a Delaunay triangle (i, j, k) in the context of circle packing is not standard.
    # However, if we assume the circles are tangent to each other and walls, the radius of circle i is determined by its position and neighbors.
    
    # Let's stick to the objective function used in the inspiration:
    # Maximize sum of radii.
    # The constraint is that circles don't overlap and are inside.
    # We can use a penalty method or SLSQP with constraints.
    
    # For the "inradius" heuristic to guide initialization or polishing:
    # The largest possible radius for a circle at p_i given fixed positions of others is:
    # r_i = min(
    #    p_i[0], w - p_i[0], p_i[1], h - p_i[1],
    #    min_{j != i} (dist(p_i, p_j) / 2)  <-- This assumes r_j is also variable and we are finding a local max.
    # )
    # This is a valid lower bound for the feasible radius if we set r_j=0, but for mutual packing, it's tricky.
    
    # Let's return the "max possible radius" assuming other circles have 0 radius, which is just the distance to the nearest wall/point.
    # This is useful for a "greedy" expansion or as a baseline.
    
    for i in range(n):
        x, y = points[i]
        r_wall = min(x, w - x, y, h - y)
        r_min_neighbor = np.inf
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(points[i] - points[j])
            r_min_neighbor = min(r_min_neighbor, dist / 2.0)
        inradii[i] = min(r_wall, r_min_neighbor)
        
    return inradii

def _delaunay_construction(w, h):
    """Generate a dense initial packing using a hexagonal grid clipped to the rectangle."""
    # Hexagonal packing has a density of pi / (2 * sqrt(3)) ~ 0.9069
    # Let's try to place circles in a hexagonal pattern.
    
    # Estimate number of circles that fit
    # Area of rectangle = w*h
    # Area per circle in hex packing ~ 2 * sqrt(3) * r^2
    # N ~ w*h / (2 * sqrt(3) * r^2)
    # We want N=21.
    
    # Let's just generate a grid of points and select the best 21.
    # Or, generate a hex grid with a specific spacing and see how many fit.
    
    # Better: Start with a square grid, then optimize.
    # The inspiration says "Delaunay-based construction".
    # Let's create a set of candidate points, triangulate, and pick a subset?
    # No, let's just use a hexagonal initial guess.
    
    points = []
    r = 0.1 # Initial radius guess
    dx = 2 * r * 0.95 # Slight overlap in grid to allow optimization to separate
    dy = 2 * r * np.sqrt(3) * 0.95
    
    row = 0
    y = r + dy/2
    while y < h - r/2:
        x_offset = 0 if row % 2 == 0 else dx/2
        x = r + x_offset + dx/2
        while x < w - r/2:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
        
    points = np.array(points)
    
    if len(points) < N:
        # Fallback to square grid if hex grid is too sparse
        side = int(np.sqrt(N)) + 1
        points = []
        for i in range(side):
            for j in range(side):
                if len(points) >= N:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                points.append([x, y])
        points = np.array(points[:N])
    else:
        # Select the N points that are most "central" or just take the first N?
        # Taking the first N from a hex grid is fine.
        points = points[:N]
        
    return points

def _objective(params, w, h):
    """
    Objective function for SLSQP.
    We want to maximize sum of radii.
    Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    """
    n = N
    points = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Penalty for negative radii
    if np.any(radii < 0):
        return 1e6
    
    # Penalty for going out of bounds
    penalty = 0
    for i in range(n):
        x, y = points[i]
        r = radii[i]
        if x - r < 0: penalty += (x - r)**2
        if x + r > w: penalty += (x + r - w)**2
        if y - r < 0: penalty += (y - r)**2
        if y + r > h: penalty += (y + r - h)**2
        
    # Penalty for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
                
    # We want to minimize negative sum of radii + penalty
    return -np.sum(radii) + penalty

def _slsqp_optimize(points0, w, h):
    """Run SLSQP optimization starting from points0."""
    n = N
    # Initial radii: use inradius heuristic
    radii0 = _compute_inradii(points0, w, h)
    
    x0 = np.concatenate([points0.flatten(), radii0])
    
    bounds = []
    for i in range(n):
        bounds.append((0, w)) # x
        bounds.append((0, h)) # y
        bounds.append((0, min(w, h)/2)) # r (max possible radius is half the smaller dimension)
        
    # Constraints
    # We could add explicit constraints for SLSQP, but penalty in objective is often faster/easier to tune.
    # Let's try adding constraints for strict validity.
    
    def constraint_overlap(i, j):
        def con(params):
            pi = params[2*i:2*i+2]
            pj = params[2*j:2*j+2]
            ri = params[2*n+i]
            rj = params[2*n+j]
            dist = np.linalg.norm(pi - pj)
            return dist - ri - rj
        return con
        
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
            
    # Wall constraints
    for i in range(n):
        # x - r >= 0
        def con_left(i=i):
            def fun(params):
                return params[2*i] - params[2*n+i]
            return fun
        constraints.append({'type': 'ineq', 'fun': con_left})
        
        # w - x - r >= 0
        def con_right(i=i):
            def fun(params):
                return w - params[2*i] - params[2*n+i]
            return fun
        constraints.append({'type': 'ineq', 'fun': con_right})
        
        # y - r >= 0
        def con_bottom(i=i):
            def fun(params):
                return params[2*i+1] - params[2*n+i]
            return fun
        constraints.append({'type': 'ineq', 'fun': con_bottom})
        
        # h - y - r >= 0
        def con_top(i=i):
            def fun(params):
                return h - params[2*i+1] - params[2*n+i]
            return fun
        constraints.append({'type': 'ineq', 'fun': con_top})
        
    # Maximize sum of radii -> Minimize -sum(radii)
    def obj(params):
        radii = params[2*n:]
        return -np.sum(radii)
        
    # SLSQP can be slow with many constraints.
    # 21 circles -> 210 overlap constraints + 84 wall constraints = 294 constraints.
    # This might be too slow for SLSQP.
    # Let's rely on the penalty method in the objective and use bounds only.
    
    # Revised strategy: Use L-BFGS-B with penalty, or SLSQP with fewer constraints?
    # The inspiration used SLSQP. Let's try SLSQP but maybe it's too slow.
    # Let's stick to the penalty objective and use L-BFGS-B or SLSQP with just bounds.
    
    # Actually, let's use the penalty objective defined above and minimize it.
    # SLSQP with penalty objective.
    
    try:
        res = minimize(
            _objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        if res.success:
            return res.x
        else:
            return x0
    except Exception:
        return x0

def _expand_individual(points, radii, w, h):
    """Try to expand each circle individually if possible."""
    for i in range(len(points)):
        x, y = points[i]
        r_old = radii[i]
        
        # Max possible radius based on walls
        r_wall = min(x, w - x, y, h - y)
        
        # Max possible radius based on neighbors
        r_neighbor = np.inf
        for j in range(len(points)):
            if i == j:
                continue
            dist = np.linalg.norm(points[i] - points[j])
            # If we expand r_i, we must keep dist >= r_i + r_j
            # So r_i <= dist - r_j
            r_neighbor = min(r_neighbor, dist - radii[j])
            
        r_max = min(r_wall, r_neighbor)
        if r_max > r_old:
            radii[i] = r_max
            
    return radii

def _polish_result(points, radii, w, h, time_limit):
    """Iteratively optimize positions to allow for larger radii."""
    n = N
    best_value = np.sum(radii)
    
    # Local search: perturb positions, re-optimize radii
    for _ in range(10):
        if time.time() > time_limit:
            break
            
        # Try a few random perturbations
        for _ in range(5):
            if time.time() > time_limit:
                break
                
            points_perturbed = points + np.random.normal(0, 0.01, size=points.shape)
            # Ensure points are within bounds
            points_perturbed[:, 0] = np.clip(points_perturbed[:, 0], 0, w)
            points_perturbed[:, 1] = np.clip(points_perturbed[:, 1], 0, h)
            
            # Re-optimize radii for these positions
            radii_new = _compute_inradii(points_perturbed, w, h)
            
            # Check if this is better
            if np.sum(radii_new) > best_value:
                points = points_perturbed
                radii = radii_new
                best_value = np.sum(radii_new)
                
    return points, radii

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # We can vary the aspect ratio w/h.
    # Perimeter = 2(w+h) = 4 => w+h = 2.
    # Let's try a few aspect ratios.
    
    best_result = None
    best_value = -np.inf
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Initial construction
        points0 = _delaunay_construction(w, h)
        
        # Optimization
        if time.time() > deadline - 5:
            break
            
        x_opt = _slsqp_optimize(points0, w, h)
        points = x_opt[:2*N].reshape(N, 2)
        radii = x_opt[2*N:]
        
        # Polish
        points, radii = _polish_result(points, radii, w, h, deadline - 5)
        
        # Expand
        radii = _expand_individual(points, radii, w, h)
        
        # Validate
        if _is_valid(points, radii, w, h):
            value = np.sum(radii)
            if value > best_value:
                best_value = value
                best_result = (points, radii, w, h)
                
    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        points = np.zeros((N, 2))
        radii = np.zeros(N)
        # Simple grid
        side = 5
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                points[count] = [x, y]
                radii[count] = 0.5 / side - 1e-3
                count += 1
        best_result = (points, radii, w, h)
        
    points, radii, w, h = best_result
    
    # Format output: (N, 3) array of [x, y, r]
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [points[i][0], points[i][1], radii[i]]
        
    return result

# EVOLVE-BLOCK-END
