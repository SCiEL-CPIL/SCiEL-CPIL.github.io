"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # Create indices for all triples
    # i < j < k
    # This is 364 combinations.
    # We can generate all pairs and compute determinants.
    
    # Method:
    # For each i, for each j > i, for each k > j
    # This is fine in Python for N=14.
    
    # Faster vectorized approach:
    # Get all pairs (i, j)
    idx_i, idx_j = np.triu_indices(n, k=1)
    # This gives pairs (i, j) with i < j.
    # We need triples.
    
    # Let's stick to a simple loop but optimized with numpy operations inside?
    # Actually, for N=14, a pure Python loop of 364 items is negligible compared to ConvexHull.
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices points[i], points[j], points[k]
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # 2 * Area = | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or few points, return 0 or small value
        return 1e-6

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def construct():
    """
    Use Simulated Annealing to find 14 points in the unit square that
    maximize the minimum triangle area relative to the convex hull area.
    """
    
    def _sa_optimize(init_points, n_iter=50000, T_start=0.01, T_end=1e-6):
        best_points = init_points.copy()
        best_score = _score(best_points)
        current_points = init_points.copy()
        current_score = best_score
        
        # Pre-compute indices for triangles to speed up _min_triangle_area?
        # No, we call _score which calls _min_triangle_area.
        # Let's make _min_triangle_area faster by caching indices?
        # Or just inline the calculation in the SA loop for max speed?
        
        # Let's optimize the inner loop of SA.
        # We need to calculate score for current and neighbor.
        
        # To make it faster, we can pre-compute the list of triples.
        triples = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
        
        def _fast_min_area(pts):
            min_a = float('inf')
            for i, j, k in triples:
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if a < min_a:
                    min_a = a
            return min_a
            
        def _fast_score(pts):
            min_a = _fast_min_area(pts)
            try:
                hull_area = ConvexHull(pts).volume
            except:
                hull_area = 1e-6
            if hull_area <= 0:
                return 0.0
            return min_a / hull_area

        # Re-initialize with fast functions
        best_score = _fast_score(best_points)
        current_score = best_score
        
        for t in range(n_iter):
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (t / n_iter)
            
            # Pick a random point to perturb
            idx = np.random.randint(N)
            old_x, old_y = current_points[idx]
            
            # Gaussian perturbation
            # Scale of perturbation can be adaptive, but fixed small step is okay
            # Step size should be related to T? No, usually fixed or decreasing.
            # Let's use a fixed small step, e.g., 0.05
            step = 0.05
            new_x = old_x + np.random.normal(0, step)
            new_y = old_y + np.random.normal(0, step)
            
            # Clip to [0, 1]
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            
            # Save old values
            current_points[idx] = [new_x, new_y]
            
            new_score = _fast_score(current_points)
            
            # Acceptance probability
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp((new - old) / T)
                # Since we maximize, delta = new - old is negative.
                delta = new_score - current_score
                # If delta is very negative, exp is 0.
                # Avoid underflow
                if delta < -100 * T:
                    prob = 0.0
                else:
                    prob = np.exp(delta / T)
                
                if np.random.random() < prob:
                    current_score = new_score
                else:
                    # Reject, revert
                    current_points[idx] = [old_x, old_y]
                    
        return best_points

    # Multiple restarts
    best_overall_points = None
    best_overall_score = 0.0
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        # Initialization: Random points in [0.1, 0.9] to avoid boundary issues initially
        # Or grid-based init?
        # Let's try a mix: 50% random, 50% grid-like
        
        if restarts % 2 == 0:
            # Grid init
            # 14 points in a 4x4 grid (16 slots), pick 14
            # Or 3x5 grid
            xs = np.linspace(0.1, 0.9, 4)
            ys = np.linspace(0.1, 0.9, 4)
            grid_points = np.array([[x, y] for x in xs for y in ys])
            # Shuffle and pick 14
            np.random.shuffle(grid_points)
            init_points = grid_points[:14].copy()
            # Add small noise
            init_points += np.random.normal(0, 0.05, (14, 2))
            init_points = np.clip(init_points, 0.0, 1.0)
        else:
            # Random init
            init_points = np.random.uniform(0.1, 0.9, (N, 2))
            
        # Run SA
        # Adjust iterations based on time remaining
        time_elapsed = time.time() - start_time
        time_remaining = time_limit - time_elapsed
        # Estimate time per iteration? 
        # Let's just run a fixed number, say 10000, and check time after.
        # If it takes too long, we break.
        
        res_points = _sa_optimize(init_points, n_iter=10000, T_start=0.01, T_end=1e-6)
        
        res_score = _score(res_points)
        
        if res_score > best_overall_score:
            best_overall_score = res_score
            best_overall_points = res_points.copy()
            
        # Check if we should stop
        if time.time() - start_time > time_limit:
            break
            
    if best_overall_points is None:
        # Fallback
        best_overall_points = np.random.uniform(0, 1, (N, 2))
        
    return best_overall_points

# EVOLVE-BLOCK-END
