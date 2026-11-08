import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
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

def _soft_min_area(points, beta=200.0):
    """
    Calculate a smooth approximation of the minimum area using log-sum-exp.
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    y = -beta * areas
    max_y = np.max(y)
    log_sum_exp = max_y + np.log(np.sum(np.exp(y - max_y)))
    return (1.0 / beta) * log_sum_exp

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
    
    # Clip negative weights to 0
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

def _objective_smooth(x, beta=200.0):
    """
    Smooth objective function for gradient-based optimization.
    """
    points = x.reshape(N, 2)
    points = _project_points(points)
    return _soft_min_area(points, beta)

def _sdp_initialization(rng, time_limit):
    """
    Use SDP/linear programming to maximize minimum pairwise distance squared
    as a convex surrogate. This provides a better global initial point than random starts.
    """
    # Maximize t subject to ||p_i - p_j||^2 >= t for all pairs i<j
    # This is not directly an LP, but we can use a simpler approach:
    # Maximize the minimum distance using iterative convex optimization.
    # For a simpler and faster approach, let's use a greedy placement + local optimization.
    
    # Actually, let's use a simpler but effective initialization:
    # Place points using a space-filling approach with jitter, then polish.
    
    # Alternative: Use scipy.optimize.minimize to maximize the minimum pairwise distance
    # This is a non-convex problem, but starting from a good initial guess helps.
    
    # Let's try a different approach: place points on a grid-like pattern within the triangle
    # and then refine.
    
    # Simple approach: place N points using rejection sampling with a minimum distance constraint
    # This is slow for N=11 but provides a good initial spread.
    
    points = []
    min_dist = 0.1
    max_attempts = 100000
    attempts = 0
    
    while len(points) < N and attempts < max_attempts:
        if time.time() > time_limit:
            break
        attempts += 1
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = np.array([u + 0.5 * v, H * v])
            # Check minimum distance to existing points
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            elif len(points) < N and attempts % 1000 == 0:
                # Accept even if too close, to make progress
                points.append(p)
    
    # If we didn't get enough points, fill with random
    while len(points) < N:
        if time.time() > time_limit:
            break
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            points.append(np.array([u + 0.5 * v, H * v]))
    
    if len(points) < N:
        return None
    
    return np.array(points)

def _polish_smooth(points, time_limit, beta=200.0):
    """
    Use L-BFGS-B to polish the solution using the smooth surrogate.
    """
    x0 = points.flatten()
    
    def callback(xk):
        return time.time() > time_limit
    
    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False, 'callback': callback}
        )
        final_points = _project_points(res.x.reshape(N, 2))
        final_score = _min_triangle_area(final_points)
        return final_points, final_score
    except Exception:
        return points, _min_triangle_area(points)

def _local_search_sa(points, time_limit, initial_temp=0.01, initial_step=0.05):
    current_points = points.copy()
    current_score = _min_triangle_area(current_points)
    best_local_points = current_points.copy()
    best_local_score = current_score
    
    temp = initial_temp
    cooling = 0.999
    step_size = initial_step
    stagnation_count = 0
    max_stagnation = 300
    
    while time.time() < time_limit:
        i = rng.randint(0, N-1)
        
        dx = rng.gauss(0, step_size)
        dy = rng.gauss(0, step_size)
        
        new_points = current_points.copy()
        new_points[i] = [current_points[i][0] + dx, current_points[i][1] + dy]
        new_points[i] = _project_to_triangle(new_points[i])
        
        new_score = _min_triangle_area(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            stagnation_count = 0
            if current_score > best_local_score:
                best_local_points = current_points.copy()
                best_local_score = current_score
        else:
            delta = current_score - new_score
            if delta < 1e-9:
                continue
            prob = np.exp(-delta / temp)
            if rng.random() < prob:
                current_points = new_points
                current_score = new_score
        
        temp *= cooling
        if temp < 0.001:
            temp = 0.005
            step_size = 0.02
        
        stagnation_count += 1
        if stagnation_count > max_stagnation:
            for j in range(N):
                current_points[j] = current_points[j] + rng.gauss(0, 0.01, 2)
                current_points[j] = _project_to_triangle(current_points[j])
            current_score = _min_triangle_area(current_points)
            temp = 0.01
            step_size = 0.05
            stagnation_count = 0
            
    return best_local_points, best_local_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    rng = random.Random(42)
    
    # Phase 1: SDP-style initialization
    sdp_time = min(10.0, deadline - 10.0)
    if sdp_time > 0:
        init_points = _sdp_initialization(rng, time.time() + sdp_time)
        if init_points is not None:
            sa_points, sa_score = _local_search_sa(init_points, time.time() + min(5.0, deadline - 5.0))
            pol_points, pol_score = _polish_smooth(sa_points, time.time() + min(5.0, deadline - 5.0))
            if pol_score > best_score:
                best_points = pol_points
                best_score = pol_score
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Phase 2: Multiple random restarts with hybrid search
    remaining_time = deadline - 15.0 - time.time()
    if remaining_time > 0:
        num_restarts = max(1, int(remaining_time / 3.0))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Generate initial points
            if restart % 3 == 0 and best_points is not None:
                points = best_points + rng.gauss(0, 0.05, (N, 2))
                for i in range(N):
                    points[i] = _project_to_triangle(points[i])
            else:
                points = np.zeros((N, 2))
                for i in range(N):
                    while True:
                        u, v = rng.random(), rng.random()
                        if u + v < 1.0:
                            points[i] = [u + 0.5 * v, H * v]
                            break
            
            # SA phase
            sa_time_slice = min(2.0, deadline - 15.0 - time.time())
            if sa_time_slice <= 0:
                break
            sa_points, sa_score = _local_search_sa(points, time.time() + sa_time_slice)
            
            # Polish phase
            polish_time_slice = min(2.0, deadline - 10.0 - time.time())
            if polish_time_slice <= 0:
                break
            pol_points, pol_score = _polish_smooth(sa_points, time.time() + polish_time_slice)
            
            if pol_score > best_score:
                best_points = pol_points
                best_score = pol_score
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Fallback
    if best_points is None:
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_points = np.array(points)
    
    best_points = _project_points(best_points)
    
    return best_points
