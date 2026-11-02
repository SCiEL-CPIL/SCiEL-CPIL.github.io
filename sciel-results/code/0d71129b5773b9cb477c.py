import numpy as np
from scipy.spatial import ConvexHull
import time
import random

N = 14

# Pre-compute indices for all unique triplets (i, j, k) with i < j < k
triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            triplets.append((i, j, k))
triplets = np.array(triplets)
I_IDX = triplets[:, 0]
J_IDX = triplets[:, 1]
K_IDX = triplets[:, 2]

def compute_score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        # Vectorized triangle area calculation
        p1 = points[I_IDX]
        p2 = points[J_IDX]
        p3 = points[K_IDX]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Area = 0.5 * |cross(v1, v2)|
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        min_area = np.min(areas)
        
        # Convex hull area
        hull = ConvexHull(points)
        hull_area = hull.area
        
        if hull_area < 1e-10:
            return 0.0
            
        return min_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with a regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = compute_score(best_points)
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    step_size = 0.05
    
    iteration = 0
    while time.time() < deadline:
        # Generate a new candidate by perturbing current points
        # We perturb all points with Gaussian noise
        noise = np.random.normal(0, step_size, (N, 2))
        new_points = current_points + noise
        
        # Optional: Keep points within a reasonable bounding box to avoid numerical issues
        # Since the problem is scale invariant, we can normalize if needed, 
        # but simple perturbation usually works if step_size is small.
        
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_points = new_points
                    current_score = new_score
            # If delta >= 0, it's already handled above, but just in case
            elif new_score > current_score:
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()

        # Cooling
        T *= cooling_rate
        
        # Adapt step size if stuck? 
        # Simple heuristic: if no improvement in last 1000 steps, increase step size
        # For simplicity and robustness, we'll keep it fixed or slowly decay.
        # Let's decay step size slightly as well
        step_size *= 0.9999
        
        iteration += 1
        if T < T_min:
            break
            
    return best_points

# EVOLVE-BLOCK-START
# The code above is the complete program. 
# We need to wrap it in the EVOLVE-BLOCK as per instructions.
# However, the instruction says "Rewrite the program from scratch... Output the COMPLETE program".
# And "Keep the function names and signatures of the seed program".
# The seed program had:
# N = 14
# # EVOLVE-BLOCK-START
# import random
# import numpy as np
# def construct(): ...
# # EVOLVE-BLOCK-END

# So I will place the code inside the EVOLVE-BLOCK markers in the final output.
# EVOLVE-BLOCK-END
