import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices for vectorized area computation
TRI_IDX = np.array(list(combinations(range(N), 3)))

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def soft_min_area(points, beta=50.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * areas))) / beta
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def soft_min_grad(points, beta=50.0):
    """Gradient of soft min area with respect to point positions."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Weights for soft min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient w.r.t. cross product
    # d(areas)/d(cross) = 0.5 * sign(cross)
    sign_cross = np.sign(cross)
    dareas_dcross = 0.5 * sign_cross
    
    # Gradient of soft min w.r.t. areas
    dsoftmin_dareas = -beta * weights
    
    # Combine
    dsoftmin_dcross = dsoftmin_dareas * dareas_dcross
    
    # Now backpropagate through cross = v1[0]*v2[1] - v1[1]*v2[0]
    # v1 = pj - pi, v2 = pk - pi
    # d(cross)/d(pj[0]) = v2[1], d(cross)/d(pj[1]) = -v1[0]
    # d(cross)/d(pk[0]) = v2[1], d(cross)/d(pk[1]) = -v1[0]
    # d(cross)/d(pi[0]) = -(v2[1] - v1[0]) ... wait let me redo this
    
    # cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
    # Let a = pj[0]-pi[0], b = pk[1]-pi[1], c = pj[1]-pi[1], d = pk[0]-pi[0]
    # cross = a*b - c*d
    
    # d(cross)/d(pi[0]) = -b + d = d - b = (pk[0]-pi[0]) - (pk[1]-pi[1])
    # d(cross)/d(pi[1]) = -a + c = c - a = (pj[1]-pi[1]) - (pj[0]-pi[0])
    # d(cross)/d(pj[0]) = b = pk[1]-pi[1]
    # d(cross)/d(pj[1]) = -d = -(pk[0]-pi[0])
    # d(cross)/d(pk[0]) = -c = -(pj[1]-pi[1])
    # d(cross)/d(pk[1]) = a = pj[0]-pi[0]
    
    grad = np.zeros((N, 2))
    
    # For each triangle (i,j,k)
    idx_i = TRI_IDX[:, 0]
    idx_j = TRI_IDX[:, 1]
    idx_k = TRI_IDX[:, 2]
    
    # d(cross)/d(pi)
    dcross_dpi0 = (pk[:, 0] - pi[:, 0]) - (pk[:, 1] - pi[:, 1])
    dcross_dpi1 = (pj[:, 1] - pi[:, 1]) - (pj[:, 0] - pi[:, 0])
    
    # d(cross)/d(pj)
    dcross_dj0 = pk[:, 1] - pi[:, 1]
    dcross_dj1 = -(pk[:, 0] - pi[:, 0])
    
    # d(cross)/d(pk)
    dcross_dk0 = -(pj[:, 1] - pi[:, 1])
    dcross_dk1 = pj[:, 0] - pi[:, 0]
    
    # Accumulate gradients
    for t in range(len(idx_i)):
        g = dsoftmin_dcross[t]
        i = idx_i[t]
        j = idx_j[t]
        k = idx_k[t]
        
        grad[i, 0] += g * dcross_dpi0[t]
        grad[i, 1] += g * dcross_dpi1[t]
        grad[j, 0] += g * dcross_dj0[t]
        grad[j, 1] += g * dcross_dj1[t]
        grad[k, 0] += g * dcross_dk0[t]
        grad[k, 1] += g * dcross_dk1[t]
    
    return grad

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < -eps or x < -eps or x > 1.0 + eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > SQRT3 * (1.0 - x) + eps:
        return False
    return True

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def project_points_to_triangle(points):
    """Project all points onto the triangle."""
    for i in range(len(points)):
        points[i] = project_to_triangle(points[i])
    return points

def constrained_objective(flat_points, beta=50.0):
    """Objective function for optimization with penalty for being outside triangle."""
    points = flat_points.reshape(N, 2)
    points = project_points_to_triangle(points.copy())
    
    # Negative soft min area (we minimize negative to maximize)
    val = -soft_min_area(points, beta)
    
    # Penalty for points outside triangle (should be 0 after projection, but just in case)
    penalty = 0.0
    for p in points:
        if not is_inside_triangle(p, eps=1e-8):
            penalty += 10.0
    
    return val + penalty

def constrained_grad(flat_points, beta=50.0):
    """Gradient for constrained objective."""
    points = flat_points.reshape(N, 2)
    points = project_points_to_triangle(points.copy())
    
    grad = -soft_min_grad(points, beta)
    return grad.flatten()

def lbfgs_optimize(points, beta=50.0, maxiter=100, deadline=None):
    """Optimize using L-BFGS-B with soft min surrogate."""
    flat = points.flatten().copy()
    
    def obj(x):
        if deadline is not None and time.time() > deadline:
            return 1e10  # Large penalty to stop
        return constrained_objective(x, beta)
    
    def grad(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return constrained_grad(x, beta)
    
    try:
        result = minimize(
            obj, flat, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 200}
        )
        if result.success or result.fun < 1e10:
            opt_points = result.x.reshape(N, 2)
            opt_points = project_points_to_triangle(opt_points.copy())
            return opt_points
        else:
            return points
    except Exception:
        return points

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < -eps or x < -eps or x > 1.0 + eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > SQRT3 * (1.0 - x) + eps:
        return False
    return True

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def generate_random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        p = (1.0 - u - v) * V0 + u * V1 + v * V2
        points.append(p)
    return np.array(points)

def local_optimize(points, rng, max_iter=500, step_size=0.05, deadline=None):
    """Local optimization using coordinate descent / hill climbing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Try moving point i in different directions
            best_local = points[i].copy()
            best_local_val = best_val
            
            for _ in range(30):
                if deadline is not None and time.time() > deadline:
                    break
                    
                # Random perturbation
                dx = rng.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        # Update best value
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        # If no improvement in a full pass, reduce step size
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return points, best_val

def simulated_annealing(points, rng, deadline, initial_temp=0.1, cooling=0.9995, max_iter=10000):
    """Simulated annealing to escape local optima."""
    points = points.copy()
    best_points = points.copy()
    best_val = min_triangle_area(points)
    current_val = best_val
    temp = initial_temp
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        i = rng.randint(N)
        dx = rng.uniform(-0.05, 0.05, 2)
        new_p = points[i] + dx
        new_p = project_to_triangle(new_p)
        
        if not is_inside_triangle(new_p, eps=1e-6):
            continue
        
        new_points = points.copy()
        new_points[i] = new_p
        
        new_val = min_triangle_area(new_points)
        delta = new_val - current_val
        
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
            points = new_points
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_points = points.copy()
        
        temp *= cooling
    
    return best_points, best_val

def micro_optimize(points, deadline=None):
    """
    Fine-grained coordinate descent to polish the solution.
    Tries small moves in 8 directions for each point.
    """
    points = points.copy()
    best_val = min_triangle_area(points)
    directions = [
        np.array([1, 0]), np.array([-1, 0]),
        np.array([0, 1]), np.array([0, -1]),
        np.array([1, 1]), np.array([-1, 1]),
        np.array([1, -1]), np.array([-1, -1])
    ]
    # Normalize directions
    directions = [d / np.linalg.norm(d) for d in directions]
    
    step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    
    for step in step_sizes:
        if deadline is not None and time.time() > deadline:
            break
        improved = True
        while improved:
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(N):
                for d in directions:
                    new_p = points[i] + step * d
                    new_p = project_to_triangle(new_p)
                    if not is_inside_triangle(new_p, eps=1e-6):
                        continue
                    new_points = points.copy()
                    new_points[i] = new_p
                    val = min_triangle_area(new_points)
                    if val > best_val + 1e-12:
                        points = new_points
                        best_val = val
                        improved = True
    return points, best_val

def nelder_mead_polish(points, deadline=None):
    """Nelder-Mead polish for high-precision local optimization."""
    if deadline is not None and time.time() > deadline:
        return points, min_triangle_area(points)
    
    flat = points.flatten().copy()
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        pts = x.reshape(N, 2)
        pts = project_points_to_triangle(pts.copy())
        return -min_triangle_area(pts)
    
    try:
        result = minimize(
            objective, flat, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < 1e10:
            opt_points = result.x.reshape(N, 2)
            opt_points = project_points_to_triangle(opt_points.copy())
            return opt_points, -result.fun
        else:
            return points, min_triangle_area(points)
    except Exception:
        return points, min_triangle_area(points)

def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_value = -1
    
    # Try multiple strategies
    strategies = [
        ("structured", generate_structured_initial()),
    ]
    
    # Add random initializations
    for seed in range(8):
        strategies.append((f"random_{seed}", generate_random_points(np.random.RandomState(seed * 42 + 7))))
    
    for name, init_points in strategies:
        if time.time() > deadline - 30:
            break
        
        # Step 1: L-BFGS optimization with soft min surrogate
        lbfgs_points = lbfgs_optimize(init_points, beta=30.0, maxiter=150, deadline=deadline - 5)
        lbfgs_val = min_triangle_area(lbfgs_points)
        
        if lbfgs_val > best_value:
            best_value = lbfgs_val
            best_points = lbfgs_points.copy()
        
        # Step 2: Local optimization (Hill Climbing) from L-BFGS result
        if time.time() < deadline - 20:
            hc_points, hc_val = local_optimize(lbfgs_points, rng, max_iter=200, step_size=0.05, deadline=deadline - 5)
            
            if hc_val > best_value:
                best_value = hc_val
                best_points = hc_points.copy()
        
        # Step 3: Simulated Annealing from the best local optimum
        if time.time() < deadline - 25:
            sa_points, sa_val = simulated_annealing(best_points, rng, deadline, initial_temp=0.03, cooling=0.9997, max_iter=3000)
            
            if sa_val > best_value:
                best_value = sa_val
                best_points = sa_points.copy()
        
        # Step 4: Nelder-Mead polish
        if time.time() < deadline - 15:
            nm_points, nm_val = nelder_mead_polish(best_points, deadline=deadline - 3)
            
            if nm_val > best_value:
                best_value = nm_val
                best_points = nm_points.copy()
    
    # Final micro-optimization pass on the best found
    if best_points is not None and time.time() < deadline - 8:
        final_points, final_val = micro_optimize(best_points, deadline=deadline - 2)
        if final_val > best_value:
            best_value = final_val
            best_points = final_points.copy()
    
    # Final Nelder-Mead polish if time permits
    if best_points is not None and time.time() < deadline - 5:
        nm2_points, nm2_val = nelder_mead_polish(best_points, deadline=deadline - 1)
        if nm2_val > best_value:
            best_value = nm2_val
            best_points = nm2_points.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i, p in enumerate(best_points):
            if not is_inside_triangle(p, eps=1e-6):
                best_points[i] = project_to_triangle(p)
    
    if best_points is None:
        # Fallback
        best_points = generate_structured_initial()
    
    return best_points
