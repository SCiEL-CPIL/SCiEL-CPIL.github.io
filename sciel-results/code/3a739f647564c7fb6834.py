import numpy as np
import time
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
    This is the standard projection onto the convex hull of V0, V1, V2.
    """
    # Barycentric coordinates with respect to V0, V1, V2
    # p = l0*V0 + l1*V1 + l2*V2
    # We want to find l0, l1, l2 >= 0, sum l_i = 1 minimizing ||p - (l0*V0 + l1*V1 + l2*V2)||^2
    
    # Using the formula for barycentric coordinates:
    # l0 = area(p, V1, V2) / area(V0, V1, V2)
    # l1 = area(V0, p, V2) / area(V0, V1, V2)
    # l2 = area(V0, V1, p) / area(V0, V1, V2)
    
    # Area of triangle (a, b, c) = 0.5 * |(b-a) x (c-a)|
    # Since we divide by the total area, we can omit the 0.5 factor in numerator and denominator.
    
    # Total area * 2 (cross product magnitude)
    # Cross product of (V1-V0) and (V2-V0)
    v1_v0 = V1 - V0
    v2_v0 = V2 - V0
    total_cross = v1_v0[0] * v2_v0[1] - v1_v0[1] * v2_v0[0]
    
    if np.isclose(total_cross, 0.0):
        return V0.copy()
        
    # l0 numerator: cross(V1-p, V2-p)
    v1_p = V1 - p
    v2_p = V2 - p
    cross0 = v1_p[0] * v2_p[1] - v1_p[1] * v2_p[0]
    l0 = cross0 / total_cross
    
    # l1 numerator: cross(p-V0, V2-V0)
    p_v0 = p - V0
    cross1 = p_v0[0] * v2_v0[1] - p_v0[1] * v2_v0[0]
    l1 = cross1 / total_cross
    
    # l2 numerator: cross(V1-V0, p-V0)
    cross2 = v1_v0[0] * p_v0[1] - v1_v0[1] * p_v0[0]
    l2 = cross2 / total_cross
    
    # Clip to [0, 1] and normalize
    # Actually, the standard projection algorithm clips negative weights to 0 and renormalizes.
    # This is equivalent to finding the closest point on the simplex.
    
    l = np.array([l0, l1, l2])
    l = np.maximum(l, 0.0)
    s = np.sum(l)
    if s > 0:
        l /= s
    else:
        # If all are 0 (which shouldn't happen for points in the affine span), return V0
        return V0.copy()
        
    return l[0]*V0 + l[1]*V1 + l[2]*V2

def _project_points(points):
    """
    Project all points to the triangle.
    """
    projected = np.array([_project_to_triangle(p) for p in points])
    return projected

def _smooth_min_triangle_area(points, beta=10.0):
    """
    Calculate a smooth approximation of the minimum triangle area.
    Uses log-sum-exp trick for stability.
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Softmin: -1/beta * log(sum(exp(-beta * areas)))
    y = -beta * areas
    y_max = np.max(y)
    log_sum = y_max + np.log(np.sum(np.exp(y - y_max)))
    softmin = -1.0 / beta * log_sum
    return softmin

def _objective_smooth(x, beta):
    """
    Objective function for the optimizer.
    x: flat array of length 2*N
    Returns negative smooth min area
    """
    points = x.reshape(N, 2)
    points = _project_points(points)
    val = _smooth_min_triangle_area(points, beta)
    return -val

def _local_search(points, time_limit):
    """
    Perform local search using L-BFGS-B with increasing beta.
    """
    current_points = points.copy()
    current_score = _min_triangle_area(current_points)
    
    # Annealing schedule for beta
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    x = current_points.flatten()
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_smooth, 
                x, 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.fun < -current_score + 1e-9:
                x = res.x
                new_points = x.reshape(N, 2)
                new_points = _project_points(new_points)
                new_score = _min_triangle_area(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
        except Exception:
            pass
            
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = -1.0
    
    rng = np.random.RandomState(42)
    
    # Initialization Strategies
    
    def random_init():
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    points[i] = [u + 0.5 * v, H * v]
                    break
        return points

    def repulsion_init():
        points = random_init()
        # Run a few steps of minimizing 1/d^2
        for step in range(50):
            forces = np.zeros((N, 2))
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    d = points[i] - points[j]
                    dist_sq = np.dot(d, d)
                    if dist_sq < 1e-8:
                        dist_sq = 1e-8
                    # Force is proportional to 1/d^3 * d
                    f = d / (dist_sq * np.sqrt(dist_sq))
                    forces[i] += f
            points += 0.01 * forces
            points = _project_points(points)
        return points

    # List of initializers
    initializers = [
        random_init,
        repulsion_init,
        lambda: _project_points(np.array([
            [0.5, 0.0], [0.25, 0.433], [0.75, 0.433],
            [0.125, 0.216], [0.375, 0.216], [0.625, 0.216], [0.875, 0.216],
            [0.25, 0.649], [0.5, 0.649], [0.75, 0.649],
            [0.5, 0.866] # Top vertex
        ])),
        lambda: _project_points(np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, H],
            [0.25, 0.25], [0.75, 0.25], [0.5, 0.5],
            [0.1, 0.5], [0.9, 0.5],
            [0.3, 0.7], [0.7, 0.7], [0.5, 0.9]
        ]))
    ]
    
    # Run optimizations
    num_inits = len(initializers)
    time_per_init = (deadline - start_time - 10.0) / num_inits
    
    for i, init_func in enumerate(initializers):
        if time.time() > deadline - 10.0:
            break
            
        try:
            init_points = init_func()
            if init_points is None:
                continue
                
            # Allocate time for this restart
            time_limit = time.time() + time_per_init
            
            res_points, res_score = _local_search(init_points, time_limit)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                
        except Exception as e:
            pass

    # If we have time left, do a final polish on the best point
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            final_points, final_score = _local_search(best_points, deadline - 2.0)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
        except Exception as e:
            pass

    # Fallback if nothing worked
    if best_points is None:
        best_points = random_init()
        
    # Final projection to ensure validity
    best_points = _project_points(best_points)
    
    return best_points
