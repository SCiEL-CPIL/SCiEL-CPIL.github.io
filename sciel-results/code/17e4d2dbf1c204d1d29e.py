"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    n = len(points)
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate minimum triangle area
    min_area = float('inf')
    # Iterate over all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle = 0.5 * |cross(p2-p1, p3-p1)|
                cross = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _vectorized_score(points):
    """Vectorized score calculation for speed."""
    n = len(points)
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate all triangle areas vectorized
    # Use broadcasting to compute cross products for all triples
    # This is memory intensive for large N, but N=14 is small (364 triples)
    # We can just use the loop version if vectorization is tricky, but let's try a fast loop with precomputed differences
    # Actually, for N=14, the loop is fast enough. The bottleneck is ConvexHull.
    # Let's stick to the loop for reliability but maybe optimize the inner loop?
    # No, the loop is fine. The main improvement will be in the search strategy.
    
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            for k in range(j+1, n):
                pk = points[k]
                cross = abs((pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area / hull_area

def construct():
    # Improved search: Simulated Annealing with Regular Polygon Initialization
    # and Adaptive Step Size
    
    best_points = None
    best_score = 0.0
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Number of restarts
    restarts = 5
    
    for r in range(restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with a regular polygon (strong prior for max-min distance/area)
        # Rotate slightly for each restart to break symmetry
        angle_offset = np.random.uniform(0, 2*np.pi)
        points = np.column_stack([
            np.cos(np.linspace(0, 2*np.pi, N, endpoint=False) + angle_offset),
            np.sin(np.linspace(0, 2*np.pi, N, endpoint=False) + angle_offset)
        ])
        
        # Add small random perturbation to break perfect symmetry
        points += np.random.normal(0, 0.01, points.shape)
        
        current_score = _score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Simulated Annealing parameters
        T_start = 0.5
        T_end = 1e-6
        steps = 20000
        alpha = (T_end / T_start) ** (1.0 / steps)
        
        T = T_start
        step_size = 0.1
        accept_window = []
        window_size = 100
        
        for i in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            # Propose move: pick random point, Gaussian perturbation
            idx = np.random.randint(N)
            old_point = points[idx].copy()
            perturbation = np.random.normal(0, step_size, 2)
            points[idx] = old_point + perturbation
            
            new_score = _score(points)
            
            # Metropolis acceptance
            delta = new_score - current_score
            if delta > 0 or (T > 0 and np.random.random() < np.exp(delta / T)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                accept_window.append(1)
            else:
                points[idx] = old_point
                accept_window.append(0)
                
            # Adaptive step size
            if len(accept_window) >= window_size:
                accept_rate = np.mean(accept_window)
                if accept_rate > 0.6:
                    step_size *= 1.1
                elif accept_rate < 0.4:
                    step_size *= 0.9
                step_size = np.clip(step_size, 0.001, 1.0)
                accept_window = []
                
            # Cool down
            T *= alpha
            
        # Local refinement: Hill Climbing from the best point found in this restart
        # to polish the result
        if time.time() - start_time > time_limit:
            break
            
        current_points = points.copy()
        current_score = _score(current_points)
        
        for i in range(5000):
            if time.time() - start_time > time_limit:
                break
            idx = np.random.randint(N)
            old_point = current_points[idx].copy()
            step_size_hc = 0.01
            perturbation = np.random.normal(0, step_size_hc, 2)
            current_points[idx] = old_point + perturbation
            
            new_score = _score(current_points)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                current_points[idx] = old_point
    
    return best_points
# EVOLVE-BLOCK-END
