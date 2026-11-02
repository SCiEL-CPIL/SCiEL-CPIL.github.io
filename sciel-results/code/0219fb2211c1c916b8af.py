"""Rewritten program for Heilbronn triangles with discrete + continuous hybrid search."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

DEADLINE_MARGIN = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_areas_cart(points):
    """Calculate all triangle areas."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    for _ in range(3):
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0, x)
        x = np.minimum(1, x)
    
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    bary = np.maximum(0, bary)
    bsum = bary.sum(axis=1, keepdims=True)
    bsum[bsum < 1e-12] = 1.0
    bary = bary / bsum
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    soft_min = _soft_min_area_cart(cart, alpha)
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _cart_objective(flat_cart, alpha=100):
    """Objective function in Cartesian coordinates."""
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    soft_min = _soft_min_area_cart(cart, alpha)
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _grid_search_discrete(grid_size=60):
    """Discrete grid search over barycentric coordinates."""
    # Generate grid points in barycentric space
    # For a triangle, we need b0+b1+b2=1, all >= 0
    # We can parameterize with two variables: b1, b2 (b0 = 1-b1-b2)
    # b1 in [0,1], b2 in [0, 1-b1]
    
    best_score = 0
    best_bary = None
    
    # Create a 2D grid for (b1, b2)
    b1_vals = np.linspace(0, 1, grid_size)
    
    # For each point, we'll try to place it on the grid
    # But 11 points on a grid is complex, so instead we do a structured search
    
    # Alternative: use the fact that optimal points often lie on a regular grid
    # Try placing points at regular intervals
    
    # For an equilateral triangle, a natural grid is based on barycentric coordinates
    # where each coordinate is a multiple of 1/grid_size
    
    # Try a few structured configurations
    configs = []
    
    # Configuration 1: points on a triangular lattice
    for k in range(1, grid_size + 1):
        step = 1.0 / k
        pts = []
        for i in range(k + 1):
            for j in range(k + 1 - i):
                b0 = i * step
                b1 = j * step
                b2 = 1 - b0 - b1
                if b2 >= -1e-9:
                    pts.append([b0, b1, b2])
        if len(pts) >= N:
            configs.append(pts)
    
    # Evaluate each configuration by selecting N points that maximize min area
    for pts in configs:
        if len(pts) < N:
            continue
        pts = np.array(pts)
        # Select N points from the grid points
        # Use a greedy approach: start with corners and add points that maximize min distance
        selected = [0, 1, len(pts) - 1]  # Start with three corners (approx)
        # Actually, let's just randomly sample N points from the grid and evaluate
        for trial in range(50):
            idx = np.random.choice(len(pts), N, replace=False)
            bary = pts[idx]
            cart = _bary_to_cart(bary)
            cart = _project_cartesian(cart)
            if not _is_valid_cart(cart):
                continue
            score = _vectorized_min_area_cart(cart)
            if score > best_score:
                best_score = score
                best_bary = bary.copy()
    
    return best_bary, best_score

def _local_refinement(bary, max_iter=10000):
    """Local refinement using Nelder-Mead in Cartesian space."""
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    best_score = _vectorized_min_area_cart(cart)
    best_cart = cart.copy()
    
    # Try multiple alpha values for the soft minimum
    for alpha in [50, 100, 200, 500]:
        res = minimize(_cart_objective, cart.flatten(), method='Nelder-Mead',
                      args=(alpha,),
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        
        new_cart = res.x.reshape(N, 2)
        new_cart = _project_cartesian(new_cart)
        new_score = _vectorized_min_area_cart(new_cart)
        
        if new_score > best_score:
            best_score = new_score
            best_cart = new_cart.copy()
    
    return best_cart, best_score

def _hill_climbing(cart, n_iter=2000, step_size=0.01):
    """Hill climbing with random perturbations."""
    best_cart = cart.copy()
    best_score = _vectorized_min_area_cart(best_cart)
    
    for i in range(n_iter):
        # Random perturbation
        new_cart = best_cart + np.random.randn(N, 2) * step_size
        new_cart = _project_cartesian(new_cart)
        
        if not _is_valid_cart(new_cart):
            continue
        
        new_score = _vectorized_min_area_cart(new_cart)
        
        if new_score > best_score:
            best_score = new_score
            best_cart = new_cart.copy()
            # Decrease step size on success
            step_size *= 0.95
        else:
            # Increase step size on failure (with cap)
            step_size = min(step_size * 1.05, 0.05)
    
    return best_cart, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Phase 1: Discrete grid search for a good initial point
    best_bary, best_score = _grid_search_discrete(grid_size=40)
    
    if best_bary is None:
        best_bary = _random_bary()
        best_cart = _bary_to_cart(best_bary)
        best_cart = _project_cartesian(best_cart)
        best_score = _vectorized_min_area_cart(best_cart)
    else:
        best_cart = _bary_to_cart(best_bary)
        best_cart = _project_cartesian(best_cart)
        best_score = _vectorized_min_area_cart(best_cart)
    
    # Phase 2: Local refinement from the best discrete solution
    if time.time() < deadline - 30:
        best_cart, best_score = _local_refinement(best_bary, max_iter=5000)
    
    # Phase 3: Multiple restarts with diverse initializations
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if r % 3 == 0:
            # Random barycentric
            x0_bary = _random_bary()
        elif r % 3 == 1:
            # Near the best solution
            x0_bary = best_bary + np.random.randn(N, 3) * 0.1
            x0_bary = np.maximum(0, x0_bary)
            x0_bary /= x0_bary.sum(axis=1, keepdims=True)
        else:
            # Structured: spread points
            x0_bary = np.zeros((N, 3))
            for i in range(N):
                # Place points at roughly equal barycentric intervals
                angle = 2 * np.pi * i / N
                x0_bary[i] = [0.3 + 0.3 * np.cos(angle), 
                              0.3 + 0.3 * np.sin(angle) * 0.5, 
                              0.4]
                x0_bary[i] = np.maximum(0, x0_bary[i])
                x0_bary[i] /= x0_bary[i].sum()
        
        x0_cart = _bary_to_cart(x0_bary)
        x0_cart = _project_cartesian(x0_cart)
        
        # Nelder-Mead optimization
        for alpha in [50, 100, 200]:
            if time.time() > deadline - 10:
                break
            res = minimize(_cart_objective, x0_cart.flatten(), method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            new_cart = res.x.reshape(N, 2)
            new_cart = _project_cartesian(new_cart)
            new_score = _vectorized_min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_cart = new_cart.copy()
                best_bary = _cart_to_bary(best_cart)
        
        # Hill climbing refinement
        if time.time() < deadline - 15:
            hc_cart, hc_score = _hill_climbing(best_cart, n_iter=1000, step_size=0.005)
            if hc_score > best_score:
                best_score = hc_score
                best_cart = hc_cart.copy()
                best_bary = _cart_to_bary(best_cart)
    
    # Phase 4: Final fine-tuning
    if time.time() < deadline - 5:
        for alpha in [200, 500, 1000]:
            res = minimize(_cart_objective, best_cart.flatten(), method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            
            new_cart = res.x.reshape(N, 2)
            new_cart = _project_cartesian(new_cart)
            new_score = _vectorized_min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_cart = new_cart.copy()
    
    # Ensure final result is valid
    best_cart = _project_cartesian(best_cart)
    
    return best_cart
# EVOLVE-BLOCK-END
