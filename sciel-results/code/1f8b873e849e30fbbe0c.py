import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time

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
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def bary_to_cartesian(bary):
    """
    Convert barycentric coordinates to Cartesian in the unit equilateral triangle.
    bary: (N, 3) array where each row sums to 1 and all components >= 0
    """
    # Vertices of equilateral triangle
    # A = (0,0), B = (1,0), C = (0.5, H)
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * H
    return np.column_stack((x, y))

def params_to_points(params):
    """
    Convert flat params (2*N) to Cartesian points.
    Params are (u, v) for each point, representing barycentric coords (1-u-v, u, v).
    """
    params = np.array(params).reshape(N, 2)
    u = params[:, 0]
    v = params[:, 1]
    
    # Ensure u+v <= 1
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u = u.copy()
        v = v.copy()
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
    
    alpha = 1.0 - u - v
    beta = u
    gamma = v
    bary = np.column_stack((alpha, beta, gamma))
    return bary_to_cartesian(bary)

def objective(params):
    """
    Objective to minimize: negative of min triangle area.
    """
    points = params_to_points(params)
    return -min_triangle_area_vec(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Diverse Local Search using Sobol sequence initialization
    # Generate 100 diverse starting points in the 22D parameter space
    n_starts = 100
    sobol = qmc.Sobol(d=2 * N, scramble=True, seed=42)
    # Generate points in [0, 1]^(2N)
    # We need enough points. Sobol generates 2^m points.
    # 100 points is not a power of 2. We can generate 128 and take 100.
    m = int(np.ceil(np.log2(n_starts)))
    sobol_points = sobol.random(2**m)
    sobol_points = sobol_points[:n_starts]
    
    local_results = []
    
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        start_params = sobol_points[i]
        
        # Run Nelder-Mead
        try:
            result = minimize(
                objective,
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            score = -result.fun
            local_results.append((score, result.x.copy()))
            
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
                
        except Exception:
            pass

    # Sort local results by score
    local_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Global Search with Differential Evolution seeded by best local results
    if time.time() < deadline - 20.0 and len(local_results) > 0:
        # Use top 5 local results as initial population for DE
        top_k = 5
        init_pop = [res[1] for res in local_results[:top_k]]
        
        # Add some random points to ensure diversity
        for _ in range(5):
            init_pop.append(rng.random(2 * N))
            
        init_pop = np.array(init_pop)
        
        bounds = [(0.0, 1.0)] * (2 * N)
        
        try:
            # DE with small population to stay within time
            result_de = differential_evolution(
                objective, 
                bounds, 
                maxiter=30, 
                popsize=10, 
                seed=42, 
                tol=1e-6,
                polish=False,
                init=init_pop
            )
            
            de_score = -result_de.fun
            if de_score > best_score:
                best_score = de_score
                best_params = result_de.x.copy()
                
        except Exception:
            pass

    # Phase 3: Final Polish
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            result_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            final_score = -result_final.fun
            if final_score > best_score:
                best_score = final_score
                best_params = result_final.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = rng.random(2 * N)

    # Convert to points
    points = params_to_points(best_params)
    
    return points
