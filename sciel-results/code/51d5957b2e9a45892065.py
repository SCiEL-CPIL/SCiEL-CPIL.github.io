import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
# Vertices of the standard equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute indices for all C(11, 3) = 165 triangles
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

def _min_triangle_area(points):
    """
    Calculate the minimum area of all triangles formed by the points.
    points: shape (N, 2)
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_to_triangle(p):
    """
    Project a single point p to the triangle T using barycentric coordinates clipping.
    """
    v1_p = V1 - p
    v2_p = V2 - p
    w0 = v1_p[0]*v2_p[1] - v1_p[1]*v2_p[0]
    
    p_v0 = p - V0
    v2_v0 = V2 - V0
    w1 = p_v0[0]*v2_v0[1] - p_v0[1]*v2_v0[0]
    
    v1_v0 = V1 - V0
    w2 = v1_v0[0]*p_v0[1] - v1_v0[1]*p_v0[0]
    
    total = w0 + w1 + w2
    
    if total == 0:
        return V0.copy()
        
    l0 = w0 / total
    l1 = w1 / total
    l2 = w2 / total
    
    l0 = max(0.0, l0)
    l1 = max(0.0, l1)
    l2 = max(0.0, l2)
    
    s = l0 + l1 + l2
    if s == 0:
        return V0.copy()
        
    l0 /= s
    l1 /= s
    l2 /= s
    
    return l0*V0 + l1*V1 + l2*V2

def _project_points(points):
    """
    Project all points to the triangle.
    """
    projected = np.array([_project_to_triangle(p) for p in points])
    return projected

def _objective_flat(x):
    """
    Objective function for the optimizer.
    x: flat array of length 2*N
    Returns negative min area (to maximize min area, we minimize -min area)
    """
    points = x.reshape(N, 2)
    points = _project_points(points)
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    def _local_search(points, time_limit, initial_temp=0.003, initial_step=0.015):
        """
        Faster local search with adaptive step size and temperature.
        """
        current_points = points.copy()
        current_score = _min_triangle_area(current_points)
        best_local_points = current_points.copy()
        best_local_score = current_score
        
        temp = initial_temp
        step_size = initial_step
        cooling = 0.999
        min_temp = 0.0003
        min_step = 0.003
        
        stagnation_count = 0
        max_stagnation = 800
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < time_limit:
            # Pick a random point to move
            i = rng.randint(0, N-1)
            
            # Generate a random displacement
            dx = rng.gauss(0, step_size)
            dy = rng.gauss(0, step_size)
            
            new_x = current_points[i][0] + dx
            new_y = current_points[i][1] + dy
            
            # Project back to triangle
            new_pt = _project_to_triangle(np.array([new_x, new_y]))
            
            new_points = current_points.copy()
            new_points[i] = new_pt
            
            new_score = _min_triangle_area(new_points)
            
            # Accept if better
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                stagnation_count = 0
                no_improve_count = 0
                if current_score > best_local_score:
                    best_local_points = current_points.copy()
                    best_local_score = current_score
            else:
                delta = current_score - new_score
                if delta < 1e-10:
                    # No change, accept to explore
                    if rng.random() < 0.3:
                        current_points = new_points
                        current_score = new_score
                else:
                    prob = np.exp(-delta / temp)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
            
            # Adaptive cooling
            temp *= cooling
            step_size *= 0.9995
            
            if temp < min_temp:
                temp = min_temp
            if step_size < min_step:
                step_size = min_step
            
            stagnation_count += 1
            no_improve_count += 1
            
            # If stuck, perturb to escape local optimum
            if stagnation_count > max_stagnation or no_improve_count > max_no_improve:
                for j in range(N):
                    current_points[j] = current_points[j] + np_rng.normal(0, 0.008, 2)
                    current_points[j] = _project_to_triangle(current_points[j])
                current_score = _min_triangle_area(current_points)
                temp = initial_temp * 0.7
                step_size = initial_step * 1.2
                stagnation_count = 0
                no_improve_count = 0
                
        return best_local_points, best_local_score

    def _scipy_optimize(points, time_limit):
        """
        Use scipy SLSQP for a short optimization pass.
        """
        if time.time() > time_limit:
            return points, _min_triangle_area(points)
        
        x0 = points.flatten()
        
        # Define constraints: points must be inside the triangle
        # Using barycentric coordinates: for each point, all barycentric coords >= 0
        # This is complex, so we'll use penalty-based approach instead
        
        def objective(x):
            pts = x.reshape(N, 2)
            pts = _project_points(pts)
            return -_min_triangle_area(pts)
        
        # Run with time limit by checking in callback
        result = None
        try:
            result = minimize(objective, x0, method='SLSQP', 
                           options={'maxiter': 50, 'ftol': 1e-8})
            if result.success or result.fun < objective(x0):
                new_points = _project_points(result.x.reshape(N, 2))
                new_score = _min_triangle_area(new_points)
                if new_score > _min_triangle_area(points):
                    return new_points, new_score
        except Exception:
            pass
        
        return points, _min_triangle_area(points)

    # Generate structured initializations
    def _generate_structured_init(seed_offset=0):
        """
        Generate a structured initial configuration with some symmetry.
        """
        rng_local = random.Random(42 + seed_offset)
        np_rng_local = np.random.RandomState(42 + seed_offset)
        
        # Start with a rough grid-like distribution
        points = np.zeros((N, 2))
        
        # Place points in a pattern: corners, edge midpoints, and interior
        # This gives a good starting point
        # Row 0: 3 points along bottom
        # Row 1: 2 points
        # Row 2: 2 points  
        # Row 3: 2 points
        # Row 4: 2 points (top)
        
        # Actually, let's use a more systematic approach
        # Distribute points in a triangular lattice pattern
        
        # For 11 points, try:
        # Bottom row: 4 points
        # Second row: 3 points
        # Third row: 2 points
        # Fourth row: 1 point
        # Fifth row: 1 point (near top)
        
        # Or use a known good configuration as seed
        # Let's use a mix of random and structured
        
        # Structured: place points at fractional positions
        positions = []
        
        # Bottom edge (y=0): 3 points
        for x_frac in [0.15, 0.5, 0.85]:
            positions.append([x_frac, 0.0])
        
        # Second row (y ~ 0.2*H): 3 points
        y1 = 0.2 * H
        for x_frac in [0.25, 0.5, 0.75]:
            positions.append([x_frac, y1])
        
        # Third row (y ~ 0.4*H): 2 points
        y2 = 0.4 * H
        for x_frac in [0.35, 0.65]:
            positions.append([x_frac, y2])
        
        # Fourth row (y ~ 0.6*H): 2 points
        y3 = 0.6 * H
        for x_frac in [0.4, 0.6]:
            positions.append([x_frac, y3])
        
        # Top: 1 point
        positions.append([0.5, 0.8 * H])
        
        points = np.array(positions)
        
        # Add small random perturbation
        points += np_rng_local.normal(0, 0.02, (N, 2))
        
        # Project to ensure inside
        points = _project_points(points)
        
        return points

    # Run multiple restarts with different strategies
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
            
        # Generate initial points
        if restart == 0:
            # Structured initialization
            points = _generate_structured_init(0)
        elif restart == 1:
            # Different structured initialization
            points = _generate_structured_init(1)
        elif restart % 4 == 2:
            # Perturb the best found so far
            if best_points is not None:
                points = best_points + np_rng.normal(0, 0.08, (N, 2))
                points = _project_points(points)
            else:
                points = _generate_structured_init(restart)
        else:
            # Fresh random or semi-random initialization
            if restart % 2 == 0:
                points = _generate_structured_init(restart)
            else:
                # Random initialization
                points = np.zeros((N, 2))
                for i in range(N):
                    while True:
                        u, v = rng.random(), rng.random()
                        if u + v < 1.0:
                            points[i] = [u + 0.5 * v, H * v]
                            break
        
        # Determine time slice for this restart
        time_slice = min(8.0, deadline - 3.0 - time.time())
        if time_slice <= 0:
            break
            
        # Run local search
        res_points, res_score = _local_search(points, time.time() + time_slice)
        
        if res_score > best_score:
            best_points = res_points.copy()
            best_score = res_score
        
        # If we have time, do a short scipy optimization on the best
        if time.time() < deadline - 5.0 and best_points is not None:
            opt_points, opt_score = _scipy_optimize(best_points, deadline - 3.0)
            if opt_score > best_score:
                best_points = opt_points.copy()
                best_score = opt_score

    # If no optimization ran or failed, return best random
    if best_points is None:
        # Fallback to structured
        best_points = _generate_structured_init(0)
        
    # Final check: ensure points are inside
    best_points = _project_points(best_points)
    best_score = _min_triangle_area(best_points)
    
    # Final polish: run a short local search with very small steps on the best solution
    if time.time() < deadline - 1.5:
        polished_points, polished_score = _local_search(best_points, deadline - 0.5, 
                                                       initial_temp=0.001, initial_step=0.005)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    return best_points
