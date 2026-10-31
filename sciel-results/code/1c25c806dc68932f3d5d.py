"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit triangle."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
# Vertices of equilateral triangle with area 1: (0,0), (1,0), (0.5, sqrt(3)/2)
H = np.sqrt(3) / 2.0  # height, area = 0.5*1*H = sqrt(3)/4 ≈ 0.433
# Actually area = 1/2 * base * height = 1/2 * 1 * sqrt(3)/2 = sqrt(3)/4
# The task says "unit-area triangle" and "divides by its area"
# So we work in the triangle (0,0),(1,0),(0.5,sqrt(3)/2) and the evaluator divides by area
# We just need to place points inside this triangle

DEADLINE = time.time() + 130.0  # safety margin before 150s limit

def in_triangle(p):
    """Check if point is inside or on the equilateral triangle."""
    x, y = p
    if y < 0: return False
    if y > H + 1e-10: return False
    # Left edge: from (0,0) to (0.5, H)
    # x >= 2*y/sqrt(3) => x >= y/sqrt(3)*2... let me think
    # Line from (0,0) to (0.5,H): y = (H/0.5)*x = 2H*x = sqrt(3)*x
    # So x <= y/sqrt(3) for left side... wait
    # At y=0, x>=0; at y=H, x>=0.5
    # The left boundary is x = y/(2*H) * 0.5... 
    # Line through (0,0) and (0.5, H): slope = H/0.5 = 2H = sqrt(3)
    # So x = y/sqrt(3)
    # Right boundary: line through (1,0) and (0.5,H): slope = H/(0.5-1) = -2H = -sqrt(3)
    # y = -sqrt(3)*(x-1) => x = 1 - y/sqrt(3)
    if x < 0 - 1e-10: return False
    if x > 1 + 1e-10: return False
    # Left constraint: x >= y/sqrt(3)
    if x < y / np.sqrt(3) - 1e-10: return False
    # Right constraint: x <= 1 - y/sqrt(3)
    if x > 1 - y / np.sqrt(3) + 1e-10: return False
    return True

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def min_triangle_area(points):
    """Compute the minimum area triangle among all C(11,3) = 165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def min_triangle_area_fast(points):
    """Vectorized computation of minimum triangle area."""
    n = len(points)
    areas = np.zeros((n, n, n))
    # Compute all cross products
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                areas[i,j,k] = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    return np.min(areas)

def soft_min_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (negative for soft-min).
    -logsumexp(-areas/beta) * beta approximates min(areas) as beta->inf.
    """
    n = len(points)
    # Collect all triangle areas
    areas_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                a = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas_list.append(a)
    areas = np.array(areas_list)
    if len(areas) == 0:
        return 0.0
    # Soft min: -beta * log(sum(exp(-areas/beta)))
    max_neg = np.max(-areas / beta)  # for numerical stability
    logsum = max_neg + np.log(np.sum(np.exp(-areas / beta - max_neg)))
    return -beta * logsum

def soft_min_area_grad(points, beta=100.0):
    """
    Compute soft-min area and its gradient.
    Returns (value, grad) where grad is shape (2N,).
    """
    n = len(points)
    # Compute all triangle areas and their gradients
    # For triangle (i,j,k), area = 0.5*|cross(pj-pi, pk-pi)|
    # d(area)/dp_m for each m in {i,j,k}
    
    num_tris = n*(n-1)*(n-2)//6
    areas = np.zeros(num_tris)
    # Store triangle indices
    tri_indices = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                a = 0.5 * abs(cross)
                areas[idx] = a
                tri_indices.append((i, j, k, cross))
                idx += 1
    
    if len(areas) == 0:
        return 0.0, np.zeros(2*n)
    
    # Soft min weights: w_t = exp(-area_t/beta) / sum(exp(-area_s/beta))
    neg_areas_over_beta = -areas / beta
    max_val = np.max(neg_areas_over_beta)
    exp_vals = np.exp(neg_areas_over_beta - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # weights[t] for triangle t
    
    soft_min_val = -beta * (max_val + np.log(sum_exp))
    
    # Gradient: d(soft_min)/dp_m = -beta * sum_t weights_t * d(area_t)/dp_m * (-1/beta) ... 
    # Actually: soft_min = -beta * log(sum(exp(-a_t/beta)))
    # d(soft_min)/dp = -beta * [sum(exp(-a_t/beta) * (-1/beta) * da_t/dp)] / sum(exp(-a_t/beta))
    # = sum_t weights_t * da_t/dp
    
    grad = np.zeros(2*n)
    for t, (i, j, k, cross) in enumerate(tri_indices):
        if cross == 0:
            continue
        w = weights[t]
        # area = 0.5 * |cross|
        # d(0.5*|cross|)/dp_m = 0.5 * sign(cross) * d(cross)/dp_m
        sign_cross = np.sign(cross)
        # cross = (pj-pi)_x * (pk-pi)_y - (pj-pi)_y * (pk-pi)_x
        # d(cross)/dpi: -1 * (pk-pi)_y on x, +1 * (pk-pi)_x on y ... let me derive
        # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
        # d(cross)/dxi = -(yk-yi) + (yj-yi)
        # d(cross)/dyi = -(xk-xi) + (xj-xi)
        # d(cross)/dxj = (yk-yi)
        # d(cross)/dyj = -(xk-xi)
        # d(cross)/dxk = -(yj-yi)
        # d(cross)/dyk = (xj-xi)
        
        pi = points[i]; pj = points[j]; pk = points[k]
        
        d_cross_dpi_x = -(pk[1]-pi[1]) + (pj[1]-pi[1])
        d_cross_dpi_y = -(pk[0]-pi[0]) + (pj[0]-pi[0])
        d_cross_dpj_x = (pk[1]-pi[1])
        d_cross_dpj_y = -(pk[0]-pi[0])
        d_cross_dpk_x = -(pj[1]-pi[1])
        d_cross_dpk_y = (pj[0]-pi[0])
        
        factor = 0.5 * sign_cross * w
        
        grad[2*i]   += factor * d_cross_dpi_x
        grad[2*i+1] += factor * d_cross_dpi_y
        grad[2*j]   += factor * d_cross_dpj_x
        grad[2*j+1] += factor * d_cross_dpj_y
        grad[2*k]   += factor * d_cross_dpk_x
        grad[2*k+1] += factor * d_cross_dpk_y
    
    return soft_min_val, grad

def is_valid(points):
    """Check all points are inside the triangle and no two are too close."""
    for p in points:
        if not in_triangle(p):
            return False
    # Check minimum distance between any pair
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                return False
    return True

def project_to_triangle(p):
    """Project a point to the interior of the triangle (simple clamping)."""
    x, y = p
    # Clamp y
    y = max(0.0, min(H, y))
    # Clamp x based on y
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    if x_min > x_max:  # numerical issue
        x_min = 0.0
        x_max = 1.0
    x = max(x_min, min(x_max, x))
    # Add small margin
    margin = 1e-4
    x_min_m = x_min + margin
    x_max_m = x_max - margin
    if x_min_m > x_max_m:
        x = (x_min + x_max) / 2
    else:
        x = max(x_min_m, min(x_max_m, x))
    return np.array([x, y])

def flatten(points):
    """Flatten 2D points to 1D vector."""
    return points.flatten()

def unflatten(x):
    """Unflatten 1D vector to 2D points."""
    return x.reshape(N, 2)

def obj_and_grad(x):
    """Objective and gradient for optimizer. Minimize negative soft-min area."""
    points = unflatten(x)
    # Project to triangle to handle constraints
    projected = np.array([project_to_triangle(p) for p in points])
    val, grad = soft_min_area_grad(projected, beta=50.0)
    # We want to maximize min area, so minimize -val
    return -val, -grad

def local_refine(points, time_limit=5.0):
    """Run L-BFGS-B to refine points."""
    x0 = flatten(points)
    best_x = x0.copy()
    best_val = -soft_min_area(unflatten(x0), beta=50.0)
    
    # Get initial objective
    try:
        f0, g0 = obj_and_grad(x0)
    except Exception:
        return unflatten(x0), best_val
    
    start = time.time()
    try:
        res = minimize(
            lambda x: obj_and_grad(x)[0],
            x0,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if res.x is not None:
            val = -res.fun
            if val > best_val:
                best_x = res.x
                best_val = val
    except Exception:
        pass
    
    elapsed = time.time() - start
    # If we have time, do a second pass with different beta
    if time.time() < DEADLINE - 15 and elapsed < time_limit * 0.8:
        try:
            # Try with higher beta for more precise min
            def obj_high_beta(x):
                pts = unflatten(x)
                proj = np.array([project_to_triangle(p) for p in pts])
                return -soft_min_area(proj, beta=200.0)
            
            res2 = minimize(
                obj_high_beta,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res2.x is not None:
                val2 = -res2.fun
                if val2 > best_val:
                    best_x = res2.x
                    best_val = val2
        except Exception:
            pass
    
    return unflatten(best_x), best_val

def final_polish(points):
    """Final exact min-area polish using coordinate descent."""
    best_points = points.copy()
    best_val = min_triangle_area(best_points)
    
    start = time.time()
    for iteration in range(50):
        if time.time() > DEADLINE - 10:
            break
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in np.linspace(-0.02, 0.02, 5):
                for dy in np.linspace(-0.02, 0.02, 5):
                    new_points = best_points.copy()
                    new_points[i] = project_to_triangle(best_points[i] + np.array([dx, dy]))
                    if not is_valid(new_points):
                        continue
                    val = min_triangle_area(new_points)
                    if val > best_val + 1e-10:
                        best_points = new_points
                        best_val = val
                        improved = True
        if not improved:
            break
    
    return best_points, best_val

def construct():
    """Construct 11 points in the equilateral triangle maximizing min triangle area."""
    global DEADLINE
    DEADLINE = time.time() + 130.0
    
    # Initial points: vertices + centroid + some symmetric points
    # Use a good starting configuration
    init_points = []
    # Vertices
    init_points.append(np.array([0.0, 0.0]))
    init_points.append(np.array([1.0, 0.0]))
    init_points.append(np.array([0.5, H]))
    # Centroid
    init_points.append(np.array([0.5, H/3.0]))
    # Edge midpoints
    init_points.append(np.array([0.25, H/4.0]))
    init_points.append(np.array([0.75, H/4.0]))
    init_points.append(np.array([0.5, H*2/3.0]))
    # Additional symmetric points
    init_points.append(np.array([1/3.0, H/6.0]))
    init_points.append(np.array([2/3.0, H/6.0]))
    init_points.append(np.array([1/3.0, H*2/3.0]))
    init_points.append(np.array([2/3.0, H*2/3.0]))
    
    # Project all to triangle interior
    init_points = np.array([project_to_triangle(p) for p in init_points])
    
    # Ensure validity
    if not is_valid(init_points):
        # Fallback: random points
        rng = np.random.RandomState(42)
        while len(init_points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                p = np.array([u + 0.5*v, H*v])
                if in_triangle(p):
                    init_points.append(p)
    
    best_points = init_points.copy()
    best_score = min_triangle_area(best_points)
    
    # Phase 1: Differential Evolution for global search
    if time.time() < DEADLINE - 60:
        try:
            # Define bounds for each coordinate
            # x in [0,1], y in [0,H]
            bounds = [(0, 1), (0, H)] * N
            
            def de_objective(x):
                points = unflatten(x)
                projected = np.array([project_to_triangle(p) for p in points])
                if not is_valid(projected):
                    return 1.0  # penalty
                return -soft_min_area(projected, beta=50.0)
            
            # Run DE with time limit
            de_start = time.time()
            de_budget = min(40.0, DEADLINE - time.time() - 30)
            
            if de_budget > 10:
                de_res = differential_evolution(
                    de_objective,
                    bounds,
                    maxiter=int(50 + 30 * de_budget / 40),
                    popsize=20,
                    tol=1e-6,
                    seed=42,
                    init='latinhypercube',
                    update_interval=10,
                    callback=lambda xk, convergence: (time.time() > DEADLINE - 45),
                    polish=False  # We'll do our own polish
                )
                
                if de_res.x is not None:
                    de_points = np.array([project_to_triangle(p) for p in unflatten(de_res.x)])
                    if is_valid(de_points):
                        de_score = min_triangle_area(de_points)
                        if de_score > best_score:
                            best_points = de_points
                            best_score = de_score
        except Exception:
            pass
    
    # Phase 2: Local refinement on top candidates
    if time.time() < DEADLINE - 20:
        # Refine the best DE result
        refined, val = local_refine(best_points, time_limit=5.0)
        if is_valid(refined):
            exact_val = min_triangle_area(refined)
            if exact_val > best_score:
                best_points = refined
                best_score = exact_val
    
    # Phase 3: Try a few more random starts + refine
    if time.time() < DEADLINE - 15:
        rng = np.random.RandomState(123)
        for restart in range(3):
            if time.time() > DEADLINE - 10:
                break
            # Random initialization
            rand_points = []
            for _ in range(N):
                for attempt in range(100):
                    u, v = rng.random(), rng.random()
                    if u + v < 0.9:
                        p = np.array([u + 0.5*v, H*v])
                        if in_triangle(p):
                            rand_points.append(p)
                            break
            if len(rand_points) < N:
                continue
            rand_points = np.array(rand_points)
            if not is_valid(rand_points):
                continue
            
            refined_r, val_r = local_refine(rand_points, time_limit=3.0)
            if is_valid(refined_r):
                exact_val_r = min_triangle_area(refined_r)
                if exact_val_r > best_score:
                    best_points = refined_r
                    best_score = exact_val_r
    
    # Phase 4: Final polish
    if time.time() < DEADLINE - 5:
        polished, val_p = final_polish(best_points)
        if is_valid(polished):
            best_points = polished
            best_score = min_triangle_area(polished)
    
    # Final validation
    if not is_valid(best_points):
        # Emergency fallback
        best_points = init_points
    
    return best_points

# Verify at import time that construct works
if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Points shape: {pts.shape}")
    print(f"All valid: {is_valid(pts)}")
    print(f"Min area: {min_triangle_area(pts):.6f}")
    print(f"Score (area/sqrt(3)/4): {min_triangle_area(pts)/(np.sqrt(3)/4):.6f}")
