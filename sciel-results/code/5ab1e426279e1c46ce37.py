"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    Returns 0 if any point is outside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Check if points are inside the triangle
    # Vertices: (0,0), (1,0), (0.5, H)
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * x + sqrt(3)
    
    viol1 = np.maximum(0, -y)
    viol2 = np.maximum(0, y - np.sqrt(3) * x)
    viol3 = np.maximum(0, y + np.sqrt(3) * x - np.sqrt(3))
    
    if np.any(viol1 + viol2 + viol3 > 1e-9):
        return 0.0

    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    
    return np.min(areas)

def objective(params):
    """
    Objective function for minimization.
    Minimize -min_area.
    """
    points = params.reshape((N, 2))
    area = min_triangle_area_vec(points)
    
    if area == 0.0:
        return 1.0
        
    return -area

def random_valid_params():
    """Generate random points inside the triangle using barycentric coordinates."""
    u = np.random.random(N)
    v = np.random.random(N)
    # Ensure u + v <= 1
    # Simple rejection or scaling
    # Scaling is faster: if u+v > 1, scale both by 1/(u+v)
    sums = u + v
    mask = sums > 1
    if np.any(mask):
        u[mask] /= sums[mask]
        v[mask] /= sums[mask]
    
    # Convert to Cartesian
    # P = u*A + v*B + w*C, A=(0,0), B=(1,0), C=(0.5, H)
    # x = 0*u + 1*v + 0.5*(1-u-v) = 0.5 + 0.5v - 0.5u
    # y = 0*u + 0*v + H*(1-u-v) = H(1-u-v)
    x = 0.5 + 0.5 * v - 0.5 * u
    y = H * (1 - u - v)
    
    return np.column_stack((x, y)).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Phase 1: Global Search with Differential Evolution
    # DE is good for finding a rough global optimum
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    try:
        # DE might be slow, so limit iterations and use callback
        def callback(x, convergence):
            return time.time() > deadline - 20.0 # Stop if only 20s left
        
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4,
            callback=callback
        )
        
        if de_result.success or de_result.fun < 0: # If it found a valid solution
            score = -de_result.fun
            if score > best_score:
                best_score = score
                best_params = de_result.x
    except Exception:
        pass

    # Phase 2: Local Refinement from multiple starts
    # We use the DE result and several random starts
    
    starts = []
    if best_params is not None:
        starts.append(best_params)
    
    # Add random starts
    num_random_starts = 50
    for _ in range(num_random_starts):
        if time.time() > deadline - 10.0:
            break
        starts.append(random_valid_params())
    
    for params in starts:
        if time.time() > deadline - 5.0:
            break
        
        # Local minimization
        try:
            # Nelder-Mead is robust for non-smooth functions
            res = minimize(
                objective, 
                params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.fun < 0: # Valid solution
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            continue

    # Final refinement of the best candidate
    if best_params is not None and time.time() < deadline - 2.0:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to random
        best_params = random_valid_params()

    return best_params.reshape((N, 2))

# EVOLVE-BLOCK-END
