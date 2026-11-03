import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 13

# Precompute indices for all triples
TRIPLES = np.array([(i, j, k) for i in range(N - 2) for j in range(i + 1, N - 1) for k in range(j + 1, N)])
NUM_TRIPLES = len(TRIPLES)

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def angles_to_points(angles):
    """
    Convert angles to points on the unit circle.
    """
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def objective(theta):
    """
    Objective function to minimize: -score.
    We use a smooth approximation of the minimum to help gradients.
    However, for N=13, the exact min is often sufficient for BFGS if initialized well,
    but a soft-min (log-sum-exp) is smoother.
    Let's try a simple penalty: -min_area / hull_area.
    To make it smooth for optimization, we can use -log(sum(exp(-areas/k))) / log(N) approx?
    Actually, standard minimize with BFGS works well on -min(areas) if we provide numerical gradients or use L-BFGS-B.
    Since we don't have analytical gradients easily for the hull area part (which is piecewise linear),
    we will rely on finite differences via BFGS/L-BFGS-B.
    
    To improve smoothness, we can optimize the sum of the smallest few areas, 
    or just use the exact score. The score is continuous but not differentiable at points 
    where the identity of the minimum triangle changes. BFGS handles this reasonably well.
    
    We normalize by hull area.
    """
    points = angles_to_points(theta)
    
    # Calculate hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e6 # Large penalty
    except Exception:
        return 1e6

    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    # We want to maximize min_area / hull_area
    # So we minimize - (min_area / hull_area)
    if min_area < 1e-10:
        return 1e6
        
    return - (min_area / hull_area)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Strategy: Multi-start optimization on angular domain
    # The points are constrained to lie on the boundary of the convex hull.
    # This is a strong assumption, but for "maximize min triangle area / hull area",
    # pushing points to the boundary usually helps maximize the areas.
    # A regular polygon is the symmetric solution.
    
    # Initial guess: Regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    num_restarts = 0
    max_restarts = 50 # Cap restarts, but controlled by time
    
    while time.time() < deadline - 5.0:
        # Generate a perturbed initial guess
        # Small perturbations to explore near the regular polygon
        # Larger perturbations to explore other local minima
        if num_restarts < 5:
            # Fine tuning around regular polygon
            perturbation = np.random.uniform(-0.05, 0.05, N)
        else:
            # Coarser search
            perturbation = np.random.uniform(-0.5, 0.5, N)
            
        theta0 = base_angles + perturbation
        # Ensure sorted angles? 
        # angles_to_points handles any order, but convex hull calculation 
        # works regardless of input order. 
        # However, for the optimization to be well-behaved, keeping them 
        # roughly in circular order is good. The perturbation is small enough 
        # that order is preserved for the first few restarts.
        # For larger perturbations, we might shuffle, but that's fine.
        
        # Optimize
        # BFGS is good for small dimensions (N=13).
        # L-BFGS-B is also good.
        # We use BFGS.
        try:
            res = minimize(
                objective, 
                theta0, 
                method='BFGS', 
                options={'maxiter': 1000, 'disp': False}
            )
        except Exception:
            continue
            
        if res.success or res.fun < best_score * -1: # res.fun is negative score
            # res.fun is -score. So if res.fun < -best_score, it's better.
            # i.e. if -res.fun > best_score
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = angles_to_points(res.x)
        
        num_restarts += 1
        
        # If we've done a few high-quality runs, we can stop if time is tight
        if num_restarts >= 10 and time.time() > deadline - 20.0:
            break

    # Final refinement: Take the best point found, convert to angles, and do one more high-precision pass
    if best_points is not None and time.time() < deadline - 2.0:
        # Recover angles from best_points
        # Note: best_points are on the unit circle (approx, due to optimization on angles)
        # We can just use the angles from the last successful res.x if we saved them, 
        # or re-derive them. 
        # Simpler: Just re-run minimize starting from the current best angles.
        # To get angles from points:
        x, y = best_points[:, 0], best_points[:, 1]
        angles = np.arctan2(y, x)
        # Sort angles to ensure continuity? 
        # The optimizer might have rotated or permuted. 
        # Since the objective is rotationally symmetric and permutation invariant (for the set of points),
        # we can just use the angles as is, or sort them to be safe for the next step.
        angles = np.sort(angles)
        
        try:
            res_final = minimize(
                objective, 
                angles, 
                method='BFGS', 
                options={'maxiter': 2000, 'disp': False, 'gtol': 1e-10}
            )
            if -res_final.fun > best_score:
                best_score = -res_final.fun
                best_points = angles_to_points(res_final.x)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = angles_to_points(base_angles)
        
    return best_points
