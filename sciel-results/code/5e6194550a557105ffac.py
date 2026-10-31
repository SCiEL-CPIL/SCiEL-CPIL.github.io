import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
# Vertices of equilateral triangle
# A = (0,0), B = (1,0), C = (0.5, sqrt(3)/2)
V_A = np.array([0.0, 0.0])
V_B = np.array([1.0, 0.0])
V_C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute triangle indices for speed
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # Shape (165, 3)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    # x = u*0 + v*1 + w*0.5
    # y = u*0 + v*0 + w*sqrt(3)/2
    x = v + 0.5 * w
    y = (np.sqrt(3) / 2.0) * w
    return np.column_stack((x, y))

def project_to_triangle(bary):
    """
    Project barycentric coordinates (u, v) to the valid region u>=0, v>=0, u+v<=1.
    bary: (N, 2)
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip u and v to [0, 1]
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # If u + v > 1, project to the line u + v = 1
    # We can scale u and v such that u' + v' = 1
    # Simplest projection: if u+v > 1, let s = u+v. u' = u/s, v' = v/s.
    # This preserves the direction from the third vertex C.
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        
    return np.column_stack((u, v))

def calculate_min_area(points):
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) Cartesian coordinates
    """
    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def objective_de(x):
    """
    Objective for differential evolution.
    x: flat array of 2*N barycentric coordinates (u1, v1, u2, v2, ...)
    Returns negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = project_to_triangle(bary)
    points = bary_to_cart(bary)
    min_area = calculate_min_area(points)
    return -min_area

def objective_soft(x, beta):
    """
    Objective for local optimization (SLSQP/L-BFGS-B).
    Uses soft-min to smooth the landscape.
    x: flat array of 2*N barycentric coordinates
    """
    bary = x.reshape(N, 2)
    # Soft penalty for going out of bounds? 
    # DE handles bounds, but local opt might drift.
    # We rely on bounds in minimize.
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for u+v > 1 is handled by bounds? 
    # Bounds in minimize are per-variable. 
    # We need u>=0, v>=0, u+v<=1.
    # L-BFGS-B doesn't support u+v<=1 directly.
    # We must use a penalty or transform.
    # Let's use a penalty in the objective for local opt.
    viol = np.maximum(0.0, u + v - 1.0)
    penalty = 1000.0 * np.sum(viol**2)
    
    points = bary_to_cart(bary)
    min_area, _ = calculate_min_area_soft(points, beta)
    
    return -min_area + penalty

def calculate_min_area_soft(points, beta):
    """
    Calculate soft-min area for smoothness.
    """
    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, use log-sum-exp trick
    m = np.min(areas)
    if m < 1e-12:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - (1.0 / beta) * log_sum
    return soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_x = None
    
    # Bounds for differential evolution: u in [0,1], v in [0,1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run Differential Evolution
    # popsize=150, maxiter=2000 might be too slow if not careful, but N=11 is small.
    # Let's try popsize=50, maxiter=1000 first.
    # tol=1e-6
    
    try:
        res_de = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=2000, 
            popsize=50, 
            tol=1e-6, 
            polish=False # We will polish separately
        )
        
        if time.time() < deadline:
            best_x = res_de.x
            best_score = -res_de.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't run or failed, try a random start
    if best_x is None:
        rng = np.random.default_rng(42)
        # Random valid barycentric points
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        best_x = np.concatenate([u, v])
        points = bary_to_cart(best_x.reshape(N, 2))
        best_score = calculate_min_area(points)

    # Local Polish with SLSQP
    # SLSQP can handle constraints, but we used penalty.
    # Let's try to use SLSQP with constraints for better precision.
    # Constraints: u_i >= 0, v_i >= 0, u_i + v_i <= 1
    
    # Actually, L-BFGS-B is faster. Let's stick with penalty-based L-BFGS-B or SLSQP with penalty.
    # SLSQP is slower but handles constraints.
    # Let's use SLSQP with explicit constraints for the final polish.
    
    if time.time() < deadline - 5.0:
        # Define constraints for SLSQP
        # Inequality: g(x) >= 0
        # u_i >= 0 -> x[2i] >= 0
        # v_i >= 0 -> x[2i+1] >= 0
        # 1 - u_i - v_i >= 0 -> 1 - x[2i] - x[2i+1] >= 0
        
        constraints = []
        for i in range(N):
            # u_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
            # v_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
            # u_i + v_i <= 1
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
        
        bounds_slsqp = [(0.0, 1.0)] * (2 * N)
        
        # Use a high beta for soft min to approximate hard min
        beta = 1000.0
        
        try:
            res_local = minimize(
                objective_soft, 
                best_x, 
                args=(beta,), 
                method='SLSQP', 
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res_local.success and -res_local.fun > best_score:
                # Verify with hard min
                bary = res_local.x.reshape(N, 2)
                points = bary_to_cart(bary)
                hard_score = calculate_min_area(points)
                if hard_score > best_score:
                    best_score = hard_score
                    best_x = res_local.x
                    
        except Exception as e:
            print(f"Local optimization failed: {e}")
            
    # Final check
    if best_x is None:
        # Fallback
        rng = np.random.default_rng(1)
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        best_x = np.concatenate([u, v])
        
    bary = best_x.reshape(N, 2)
    points = bary_to_cart(bary)
    
    return points

if __name__ == "__main__":
    pts = construct()
    print(pts)
