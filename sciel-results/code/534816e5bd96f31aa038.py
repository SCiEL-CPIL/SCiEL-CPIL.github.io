"""Rewritten program using barycentric coordinates, L-BFGS-B, and continuation on soft-min parameter."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 130.0  # safety margin before 150s kill

# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def barycentric_to_cartesian(bary):
    """Convert barycentric coords (n,3) to cartesian (n,2)."""
    return bary[:, 0] * V[0] + bary[:, 1] * V[1] + bary[:, 2] * V[2]


def triangle_areas(points):
    """Compute all C(n,3) triangle areas for n points."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                areas.append(area)
    return np.array(areas)


def soft_min(areas, t):
    """Smooth approximation of min(areas) using log-sum-exp."""
    return -np.log(np.sum(np.exp(-t * areas))) / t


def objective(bary_flat, t, penalty_weight):
    """Objective: minimize -soft_min(areas) + penalty for barycentric constraints."""
    bary = bary_flat.reshape(N, 3)
    # Soft penalty for barycentric constraints: sum >= 1, each >= 0
    # Use smooth max(0, violation)
    sum_viol = np.maximum(0.0, 1.0 - np.sum(bary, axis=1))
    comp_viol = np.maximum(0.0, -bary)
    penalty = penalty_weight * (np.sum(sum_viol ** 2) + np.sum(comp_viol ** 2))
    
    points = barycentric_to_cartesian(bary)
    areas = triangle_areas(points)
    sm = soft_min(areas, t)
    return -sm + penalty


def construct():
    start_time = time.time()
    
    # Initial points: vertices + centroid + 6 others
    # Use barycentric coordinates
    bary_init = np.zeros((N, 3))
    # Place at vertices
    bary_init[0] = [1, 0, 0]
    bary_init[1] = [0, 1, 0]
    bary_init[2] = [0, 0, 1]
    # Centroid
    bary_init[3] = [1/3, 1/3, 1/3]
    # Midpoints of edges
    bary_init[4] = [0.5, 0.5, 0]
    bary_init[5] = [0.5, 0, 0.5]
    bary_init[6] = [0, 0.5, 0.5]
    # Inner points
    bary_init[7] = [0.4, 0.4, 0.2]
    bary_init[8] = [0.2, 0.4, 0.4]
    bary_init[9] = [0.4, 0.2, 0.4]
    bary_init[10] = [0.3, 0.3, 0.4]
    
    # Continuation on t (soft-min parameter)
    t_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    penalty_weight = 10.0
    
    best_bary = bary_init.copy()
    best_score = -np.inf
    
    for t in t_values:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Multiple restarts for each t
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            if restart == 0:
                x0 = best_bary.flatten().copy()
            else:
                # Perturb from best
                x0 = best_bary.flatten().copy()
                noise = np.random.randn(N * 3) * 0.1
                x0 = x0 + noise
                # Re-normalize barycentric coords to sum to 1
                for i in range(N):
                    s = np.sum(x0[i*3:(i+1)*3])
                    if s > 1e-10:
                        x0[i*3:(i+1)*3] = np.maximum(0, x0[i*3:(i+1)*3]) / np.sum(np.maximum(0, x0[i*3:(i+1)*3]))
                    else:
                        x0[i*3:(i+1)*3] = [1/3, 1/3, 1/3]
            
            res = minimize(
                objective, x0, args=(t, penalty_weight),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000}
            )
            
            if res.success or res.fun < best_score:
                bary_new = res.x.reshape(N, 3)
                # Check if points are inside triangle
                if np.all(bary_new >= -1e-6) and np.all(np.abs(np.sum(bary_new, axis=1) - 1) < 0.1):
                    points = barycentric_to_cartesian(bary_new)
                    areas = triangle_areas(points)
                    min_area = np.min(areas)
                    if min_area > best_score:
                        best_score = min_area
                        best_bary = bary_new.copy()
    
    # Convert to cartesian
    points = barycentric_to_cartesian(best_bary)
    
    # Final check: ensure points are inside triangle
    for i in range(N):
        p = points[i]
        # Check barycentric coordinates
        # Solve for barycentric coords
        A = np.array([[V[1][0]-V[0][0], V[2][0]-V[0][0]],
                      [V[1][1]-V[0][1], V[2][1]-V[0][1]]])
        b = np.array([p[0]-V[0][0], p[1]-V[0][1]])
        try:
            lam = np.linalg.solve(A, b)
            bary_check = np.array([1 - lam[0] - lam[1], lam[0], lam[1]])
            if np.any(bary_check < -1e-6):
                # Clamp to boundary
                bary_check = np.maximum(0, bary_check)
                s = np.sum(bary_check)
                if s > 1e-10:
                    bary_check = bary_check / s
                    points[i] = barycentric_to_cartesian(bary_check.reshape(1, 3))[0]
        except:
            pass
    
    return points


# Pre-generate with fixed seed for reproducibility
np.random.seed(42)
