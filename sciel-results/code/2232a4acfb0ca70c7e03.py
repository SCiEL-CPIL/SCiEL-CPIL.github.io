import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
h = np.sqrt(3) / 2

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, h])

def barycentric_to_cartesian(u):
    """
    u: array of shape (N, 3) with u >= 0 and sum(u, axis=1) == 1.
    Returns array of shape (N, 2) of Cartesian coordinates.
    """
    # P = u0*V0 + u1*V1 + u2*V2
    # V0 is origin, so P = u1*V1 + u2*V2
    # V1 = [1, 0], V2 = [0.5, h]
    # P_x = u1 * 1 + u2 * 0.5
    # P_y = u1 * 0 + u2 * h
    return np.column_stack((u[:, 1] + 0.5 * u[:, 2], h * u[:, 2]))

def cartesian_to_barycentric(p):
    """
    p: array of shape (N, 2)
    Returns array of shape (N, 3) of barycentric coordinates (unnormalized, may have negative values if outside).
    """
    # Solve: p = u1*V1 + u2*V2
    # p_x = u1 + 0.5 u2
    # p_y = h u2
    # u2 = p_y / h
    # u1 = p_x - 0.5 u2
    # u0 = 1 - u1 - u2
    u2 = p[:, 1] / h
    u1 = p[:, 0] - 0.5 * u2
    u0 = 1.0 - u1 - u2
    return np.column_stack((u0, u1, u2))

def min_triangle_area_cartesian(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    min_area = float('inf')
    n = len(points)
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v12 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v13 = p3 - p1
                cross = v12[0] * v13[1] - v12[1] * v13[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective_barycentric(u_flat):
    """
    u_flat: flattened array of shape (3*N,)
    Returns negative min area.
    """
    u = u_flat.reshape(N, 3)
    # Normalize to ensure they are valid barycentric coordinates
    # Since we use DE with bounds [0,1] for each component, we must normalize
    # to sum to 1 to be on the simplex.
    sums = np.sum(u, axis=1, keepdims=True)
    # Avoid division by zero (shouldn't happen with bounds [0,1] but safe)
    sums = np.where(sums == 0, 1, sums)
    u_norm = u / sums
    
    points = barycentric_to_cartesian(u_norm)
    return -min_triangle_area_cartesian(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    best_score = -1
    best_u_flat = None
    
    # Define bounds for Differential Evolution
    # Each point has 3 barycentric coords. We allow them to be in [0, 1]
    # and normalize. This maps the hypercube to the simplex.
    bounds = [(0.0, 1.0)] * (3 * N)
    
    # 1. Global Search using Differential Evolution
    # DE is robust and doesn't require gradients.
    # We use a moderate popsize and maxiter to find a good basin.
    try:
        res_de = differential_evolution(
            objective_barycentric, 
            bounds, 
            seed=42,
            maxiter=50, 
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False, # We will polish manually with L-BFGS-B
            init='sobol'
        )
        if res_de.fun < 0:
            curr_area = -res_de.fun
            if curr_area > best_score:
                best_score = curr_area
                best_u_flat = res_de.x
    except Exception as e:
        pass
    
    # 2. Local Refinement using L-BFGS-B
    # L-BFGS-B is efficient for smooth problems.
    # We will refine the best solution from DE and also try a few random starts.
    
    def objective_for_lbfgsb(u_flat):
        return objective_barycentric(u_flat)
    
    # Define constraints? No, we rely on the normalization inside the objective.
    # However, L-BFGS-B assumes a smooth landscape. The normalization u/sum(u) 
    # is smooth everywhere except at 0. Since bounds are [0,1] and we start from valid points,
    # we should be fine. But to be safer, we can add a small penalty if any u_i < 0 
    # (though bounds prevent this) or if sum is small.
    # Actually, the mapping from [0,1]^3N to the simplex is smooth and well-behaved 
    # as long as we don't hit the origin.
    
    candidates = []
    if best_u_flat is not None:
        candidates.append(best_u_flat)
    
    # Add a few random starts for local refinement
    rng = np.random.default_rng(123)
    for _ in range(5):
        rand_u = rng.random((N, 3))
        candidates.append(rand_u.flatten())
        
    # Add a symmetric guess
    # 11 points.
    # Try to place them roughly evenly.
    # Barycentric coords for centroid is (1/3, 1/3, 1/3).
    # We can generate points on a grid in barycentric space.
    # Or just use the DE result.
    
    for x0 in candidates:
        if time.time() > deadline:
            break
        
        # Ensure x0 is valid (non-negative)
        x0 = np.clip(x0, 0.0, 1.0)
        
        try:
            res_lbfgsb = minimize(
                objective_for_lbfgsb, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            curr_area = -res_lbfgsb.fun
            if curr_area > best_score:
                best_score = curr_area
                best_u_flat = res_lbfgsb.x
        except Exception:
            pass
            
    # 3. Further refinement if time permits
    if time.time() < deadline - 5.0 and best_u_flat is not None:
        # Run DE again with tighter bounds around the best solution?
        # Or just more local restarts from the best solution with small perturbations.
        rng2 = np.random.default_rng(999)
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb the best solution
            perturb = rng2.normal(0, 0.01, best_u_flat.shape)
            x0 = np.clip(best_u_flat + perturb, 0.0, 1.0)
            
            try:
                res_lbfgsb = minimize(
                    objective_for_lbfgsb, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                curr_area = -res_lbfgsb.fun
                if curr_area > best_score:
                    best_score = curr_area
                    best_u_flat = res_lbfgsb.x
            except Exception:
                pass

    # Final conversion
    if best_u_flat is None:
        # Fallback: Centroid
        u = np.ones((N, 3)) / 3.0
        best_u_flat = u.flatten()
        
    u_best = best_u_flat.reshape(N, 3)
    sums = np.sum(u_best, axis=1, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    u_norm = u_best / sums
    
    points = barycentric_to_cartesian(u_norm)
    
    return points

# EVOLVE-BLOCK-END
