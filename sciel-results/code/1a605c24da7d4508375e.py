"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Pre-compute indices for all triangles (i, j, k) with i < j < k
_tri_indices = list(combinations(range(N), 3))
_tri_idx = np.array(_tri_indices)
_i_idx = _tri_idx[:, 0]
_j_idx = _tri_idx[:, 1]
_k_idx = _tri_idx[:, 2]

def _min_area_fast(pts):
    """
    Calculate the minimum area of all triangles formed by the points.
    pts: numpy array of shape (N, 2)
    """
    pi = pts[_i_idx]
    pj = pts[_j_idx]
    pk = pts[_k_idx]
    
    # Vectorized cross product calculation
    # area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective(flat_pts):
    """
    Objective function for optimization.
    Minimizes negative min area + penalty for points outside the triangle.
    """
    pts = flat_pts.reshape(N, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Triangle constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (line from (0,0) to (0.5, sqrt(3)/2) extended? No, line from (0,0) to (1,0) is y=0. Line from (0,0) to (0.5, sqrt(3)/2) is y = sqrt(3)x. Line from (1,0) to (0.5, sqrt(3)/2) is y = -sqrt(3)(x-1))
    
    # Let's verify the lines.
    # Side 1: (0,0) to (1,0) -> y = 0. Condition: y >= 0.
    # Side 2: (0,0) to (0.5, sqrt(3)/2). Slope = sqrt(3)/2 / 0.5 = sqrt(3). Equation: y = sqrt(3) * x. Condition: y <= sqrt(3) * x.
    # Side 3: (1,0) to (0.5, sqrt(3)/2). Slope = (sqrt(3)/2 - 0) / (0.5 - 1) = (sqrt(3)/2) / (-0.5) = -sqrt(3). Equation: y - 0 = -sqrt(3)(x - 1). Condition: y <= -sqrt(3)(x - 1).
    
    pen1 = np.sum(np.maximum(0, -y))
    pen2 = np.sum(np.maximum(0, y - np.sqrt(3) * x))
    pen3 = np.sum(np.maximum(0, y + np.sqrt(3) * (x - 1)))
    
    penalty = 100.0 * (pen1 + pen2 + pen3)
    
    min_a = _min_area_fast(pts)
    return -min_a + penalty

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Bounds for Differential Evolution
    # x in [0, 1], y in [0, 0.866]
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    
    rng = np.random.default_rng(42)
    
    # Run multiple DE instances to escape local minima
    num_de_runs = 10
    
    for run in range(num_de_runs):
        if time.time() > deadline - 15:
            break
        
        try:
            # Differential Evolution
            # popsize and maxiter control the number of evaluations
            # 15 * 100 = 1500 evaluations per run. Fast.
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42 + run,
                maxiter=100, 
                popsize=15,
                tol=1e-6,
                init='latinhypercube'
            )
            
            if time.time() > deadline - 10:
                break
                
            pts = res.x.reshape(N, 2)
            score = -res.fun
            
            if score > best_score:
                best_score = score
                best_pts = pts
                
        except Exception:
            pass
    
    # If DE found a solution, polish it with L-BFGS-B
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res_opt = minimize(
                _objective, 
                best_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            pts_opt = res_opt.x.reshape(N, 2)
            score_opt = -res_opt.fun
            
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
        except Exception:
            pass

    # If we still have time, do a few local random restarts from the best solution
    if best_pts is not None and time.time() < deadline - 2:
        current_best = best_pts.copy()
        current_score = best_score
        T = 0.05
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            # Perturb a random point
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_best.copy()
            new_pts[idx] += delta
            
            new_score = -_objective(new_pts.flatten())
            
            # Accept if better or with probability exp((new-old)/T)
            if new_score > current_score or (new_score - current_score) / T > np.log(rng.random()):
                current_best = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_best
            T *= 0.95

    if best_pts is None:
        # Fallback: Place points in a grid
        # This is a very poor solution but ensures we return something valid
        pts = []
        for i in range(4):
            for j in range(3):
                if len(pts) < N:
                    x = 0.1 + i * 0.2
                    y = 0.1 + j * 0.2
                    pts.append([x, y])
        best_pts = np.array(pts[:N])
        
    return best_pts
# EVOLVE-BLOCK-END
