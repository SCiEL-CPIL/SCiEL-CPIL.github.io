# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize, differential_evolution

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def smooth_objective(x, beta=0.05):
    """
    Smooth surrogate for maximizing (d_min/d_max)^2.
    Uses soft-min and soft-max via log-sum-exp.
    """
    points = x.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Soft min: -beta * log(sum(exp(-d/beta)))
    # Soft max: beta * log(sum(exp(d/beta)))
    # We want to maximize soft_min / soft_max
    
    # Numerical stability
    neg_d_over_beta = -dists / beta
    neg_d_over_beta -= np.max(neg_d_over_beta)
    soft_min = -beta * np.log(np.sum(np.exp(neg_d_over_beta)))
    
    d_over_beta = dists / beta
    d_over_beta -= np.max(d_over_beta)
    soft_max = beta * np.log(np.sum(np.exp(d_over_beta)))
    
    if soft_max <= 0:
        return 0.0
    
    ratio = soft_min / soft_max
    return ratio * ratio

def objective(x, beta=0.05):
    """Objective for minimization (negative of score)."""
    return -smooth_objective(x, beta)

def objective_exact(x):
    """Exact objective for final evaluation."""
    points = x.reshape(-1, 2)
    return -calculate_score(points)

def hexagonal_init(n, scale=1.0, rotation=0.0):
    """Generate a hexagonal grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    theta = rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def concentric_circles_init(n, scale=1.0):
    """Generate points on concentric circles."""
    # 4 points on inner circle, 12 on outer
    pts = []
    # Center
    pts.append([0, 0])
    # Inner circle: 5 points
    for k in range(5):
        angle = 2 * np.pi * k / 5
        pts.append([0.4 * scale * np.cos(angle), 0.4 * scale * np.sin(angle)])
    # Outer circle: 10 points
    for k in range(10):
        angle = 2 * np.pi * k / 10
        pts.append([scale * np.cos(angle), scale * np.sin(angle)])
    return np.array(pts[:n])

def random_init(n, seed=42):
    """Generate a random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    r = rng.rand(n) ** 0.5
    theta = rng.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def square_init(n, scale=1.0):
    """Generate a square grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([j * scale, i * scale])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def circle_init(n, scale=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x, y))

def polish_l_bfgs_b(x0, beta=0.02, maxiter=5000):
    """Polish with L-BFGS-B on smooth surrogate."""
    try:
        res = minimize(
            objective, 
            x0, 
            args=(beta,),
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x
    except:
        return x0

def polish_exact(x0, maxiter=2000):
    """Final polish with exact objective using Powell (derivative-free)."""
    try:
        res = minimize(
            objective_exact, 
            x0, 
            method='Powell', 
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        return res.x
    except:
        return x0

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 580
    
    best_result = None
    best_score = -1.0
    
    def update_best(points):
        nonlocal best_result, best_score
        score = calculate_score(points)
        if score > best_score:
            best_score = score
            best_result = points.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # Hexagonal grids with various rotations
    for rot in [0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, np.pi/12]:
        initializations.append(hexagonal_init(n, rotation=rot))
        
    # Square grid
    initializations.append(square_init(n))
    
    # Points on a circle
    initializations.append(circle_init(n))
    
    # Concentric circles
    initializations.append(concentric_circles_init(n))
    
    # Random points
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        initializations.append(random_init(n, seed=seed))
        
    # Jittered hexagonal grids
    rng = np.random.RandomState(123)
    for _ in range(8):
        base = hexagonal_init(n, rotation=rng.rand() * np.pi)
        jitter = rng.randn(n, 2) * 0.15
        initializations.append(base + jitter)
    
    # Update best with initializations
    for init_pts in initializations:
        update_best(init_pts)
    
    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 200:
        bounds = [(-1, 1)] * (n * 2)
        try:
            de_res = differential_evolution(
                objective, 
                bounds, 
                args=(0.1,),  # Larger beta for global search
                maxiter=100, 
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < 0:
                de_points = de_res.x.reshape(-1, 2)
                update_best(de_points)
        except Exception as e:
            pass
    
    # Phase 2: Local optimization from best points and good initializations
    # Use L-BFGS-B with decreasing beta for smooth optimization
    candidates = [best_result] if best_result is not None else []
    
    # Add top initializations by score
    scored_inits = []
    for init_pts in initializations:
        scored_inits.append((calculate_score(init_pts), init_pts))
    scored_inits.sort(reverse=True)
    for _, pts in scored_inits[:8]:
        candidates.append(pts)
    
    # Deduplicate candidates (keep unique ones)
    unique_candidates = []
    for cand in candidates:
        is_dup = False
        for uc in unique_candidates:
            if np.allclose(cand, uc, atol=1e-6):
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)
    
    for i, cand in enumerate(unique_candidates):
        if time.time() > deadline - 100:
            break
        
        x0 = cand.flatten()
        
        # Progressive beta reduction
        for beta in [0.05, 0.03, 0.02, 0.01]:
            if time.time() > deadline - 80:
                break
            polished = polish_l_bfgs_b(x0, beta=beta, maxiter=2000)
            update_best(polished.reshape(-1, 2))
            x0 = polished
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 20 and best_result is not None:
        x0 = best_result.flatten()
        
        # Multiple Powell polishes with very tight tolerances
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            polished = polish_exact(x0, maxiter=1000)
            update_best(polished.reshape(-1, 2))
            x0 = polished
    
    # Fallback
    if best_result is None:
        best_result = hexagonal_init(n)
    
    return best_result

# EVOLVE-BLOCK-END
