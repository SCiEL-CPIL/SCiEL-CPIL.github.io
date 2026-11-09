import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)  # shape (165, 3)

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    # Edge AB: y >= 0
    p[:, 1] = np.maximum(p[:, 1], 0.0)
    
    # Edge AC: y <= sqrt(3)*x
    sqrt3 = np.sqrt(3)
    mask_ac = p[:, 1] > sqrt3 * p[:, 0]
    if np.any(mask_ac):
        px = p[mask_ac, 0]
        py = p[mask_ac, 1]
        dot = px + py * sqrt3
        px_proj = dot / 4.0
        py_proj = dot * sqrt3 / 4.0
        px_proj = np.clip(px_proj, 0.0, 0.5)
        py_proj = px_proj * sqrt3
        p[mask_ac, 0] = px_proj
        p[mask_ac, 1] = py_proj
    
    # Edge BC: y <= sqrt(3)*(1-x)
    mask_bc = p[:, 1] > sqrt3 * (1.0 - p[:, 0])
    if np.any(mask_bc):
        px = p[mask_bc, 0]
        py = p[mask_bc, 1]
        d = (sqrt3 * (px - 1.0) + py) / 4.0
        px_proj = px - d * sqrt3
        py_proj = py - d
        px_proj = np.clip(px_proj, 0.5, 1.0)
        py_proj = sqrt3 * (1.0 - px_proj)
        p[mask_bc, 0] = px_proj
        p[mask_bc, 1] = py_proj
        
    return p

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    p_i = points[_TRIPLETS[:, 0]]
    p_j = points[_TRIPLETS[:, 1]]
    p_k = points[_TRIPLETS[:, 2]]
    
    v12 = p_j - p_i
    v13 = p_k - p_i
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def objective_soft(x, beta):
    """
    x: flattened array of shape (2*N,)
    Returns negative soft-min area (log-sum-exp) to be minimized.
    Uses a large beta to approximate the min closely while remaining differentiable.
    """
    points = x.reshape(N, 2)
    
    # Soft penalty for being outside triangle
    sqrt3 = np.sqrt(3)
    viol1 = np.maximum(0, -points[:, 1])
    viol2 = np.maximum(0, points[:, 1] - sqrt3 * points[:, 0])
    viol3 = np.maximum(0, points[:, 1] - sqrt3 * (1.0 - points[:, 0]))
    penalty = 10.0 * np.sum(viol1**2 + viol2**2 + viol3**2)
    
    p_i = points[_TRIPLETS[:, 0]]
    p_j = points[_TRIPLETS[:, 1]]
    p_k = points[_TRIPLETS[:, 2]]
    
    v12 = p_j - p_i
    v13 = p_k - p_i
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for soft min
    # min(a) = -lim_{b->inf} (1/b) log(sum(exp(-b*a)))
    # We want to maximize min area, so minimize -soft_min_area
    # soft_min_area = -(1/beta) * log(sum(exp(-beta * areas)))
    # Objective = -soft_min_area + penalty = (1/beta) * log(sum(exp(-beta * areas))) + penalty
    
    # To avoid overflow, we can subtract max, but here areas are small.
    # Actually, for numerical stability:
    # logsumexp(-beta * areas) = max(-beta*areas) + log(sum(exp(-beta*areas - max(-beta*areas))))
    # max(-beta*areas) = -beta * min(areas)
    
    min_area = np.min(areas)
    # If min_area is 0, logsumexp is log(165) approx? No.
    # Let's just compute directly with clipping if necessary, but areas are ~0.01-0.1
    
    # To be safe against overflow in exp:
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    soft_min_area = -log_sum_exp / beta
    
    return penalty - soft_min_area

def _local_refine_soft(x0, deadline, beta_start=100.0):
    """
    Run L-BFGS-B with soft objective.
    """
    try:
        # Start with moderate beta, then increase?
        # For simplicity, use a fixed high beta for the main optimization.
        # The soft min with high beta is very close to min but smooth.
        beta = beta_start
        
        res = minimize(objective_soft, x0, args=(beta,), method='L-BFGS-B',
                       options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12})
        
        pts = project_to_triangle(res.x.reshape(N, 2))
        area = min_triangle_area(pts)
        return area, pts
    except Exception:
        return -1, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Symmetric grid layouts (1-2-3-5)
    p_top = np.array([0.5, h])
    y3 = 0.75 * h
    p_r3 = np.array([[0.4, y3], [0.6, y3]])
    y2 = 0.5 * h
    p_r2 = np.array([[0.3, y2], [0.5, y2], [0.7, y2]])
    y1 = 0.25 * h
    x1s = np.linspace(0.15, 0.85, 5)
    p_r1 = np.array(list(zip(x1s, np.full(5, y1))))
    guess1 = np.vstack([p_top, p_r3, p_r2, p_r1])
    starts.append(guess1.flatten())
    starts.append((guess1 + rng.normal(0, 0.05, guess1.shape)).flatten())
    
    # 2. Different layout: 4-4-3
    y_a = 0.2 * h
    xa = np.linspace(0.1, 0.9, 4)
    y_b = 0.5 * h
    xb = np.linspace(0.2, 0.8, 4)
    y_c = 0.8 * h
    xc = np.linspace(0.35, 0.65, 3)
    guess2 = np.vstack([np.array(list(zip(xa, np.full(4, y_a)))),
                        np.array(list(zip(xb, np.full(4, y_b)))),
                        np.array(list(zip(xc, np.full(3, y_c))))])
    starts.append(guess2.flatten())
    starts.append((guess2 + rng.normal(0, 0.04, guess2.shape)).flatten())
    
    # 3. Corner-inclusive
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    midpoints = np.array([[0.5, 0.0], [0.75, h/2], [0.25, h/2]])
    center = np.array([[0.5, h/3]])
    extra = rng.random((4, 2)) * np.array([0.6, 0.5*h]) + np.array([0.2, 0.1*h])
    guess3 = np.vstack([corners, midpoints, center, extra])
    starts.append(project_to_triangle(guess3).flatten())
    
    # 4. Random starts
    for _ in range(10):
        rp = rng.random((N, 2)) * np.array([1.0, h])
        starts.append(project_to_triangle(rp).flatten())
        
    # 5. Sobol-like quasi-random (approximated with stratified random)
    # Just use more randoms with different seeds
    for i in range(5):
        rng_i = np.random.default_rng(100 + i)
        rp = rng_i.random((N, 2)) * np.array([1.0, h])
        starts.append(project_to_triangle(rp).flatten())

    # Run local refinement on each start
    for x0 in starts:
        if time.time() > deadline - 5.0:
            break
        area, pts = _local_refine_soft(x0, deadline, beta_start=50.0)
        if pts is not None and area > best_score:
            best_score = area
            best_points = pts
    
    # If we have a good solution, try to polish with higher beta (closer to true min)
    if best_points is not None and time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        # Increase beta to sharpen the landscape for final polish
        area, pts = _local_refine_soft(x_best, deadline, beta_start=500.0)
        if pts is not None and area > best_score:
            best_score = area
            best_points = pts
            
        # One more polish with even higher beta
        if time.time() < deadline - 1.0:
            area, pts = _local_refine_soft(best_points.flatten(), deadline, beta_start=1000.0)
            if pts is not None and area > best_score:
                best_score = area
                best_points = pts

    if best_points is None:
        # Fallback
        pts = np.random.default_rng(7).random((N, 2)) * np.array([1.0, h])
        best_points = project_to_triangle(pts)
    
    return best_points
