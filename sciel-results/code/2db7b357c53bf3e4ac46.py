"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from collections import Counter

def _score(points):
    """
    Compute the max multiplicity of a positive doubled area.
    points: list of (x, y) tuples or np.array of shape (N, 2)
    """
    if isinstance(points, list):
        points = np.array(points)
    n = len(points)
    if n < 3:
        return 0
    
    # Vectorized computation of all triangle determinants
    # i < j < k
    # We can compute all pairwise differences or just use broadcasting for triples.
    # For N=30, there are 4060 triples. Broadcasting a (30, 30, 2) or similar might be heavy but doable.
    # Better: compute area for all pairs (i, j) and then for each k? No, that's O(N^3).
    
    # Let's use a slightly optimized loop or vectorized chunks if possible, 
    # but for N=30, a pure Python loop with numpy ops inside is often faster than complex vectorization due to overhead.
    # However, we can pre-compute coordinates.
    
    x = points[:, 0]
    y = points[:, 1]
    
    areas = []
    # To speed up, we can just iterate and append to a list, then count.
    # 4060 iterations is very fast.
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        # Vectorize the rest for this i
        # j > i, k > j
        # This is tricky to fully vectorize without generating all j,k pairs.
        # Let's stick to a simple loop but use local variables for speed.
        pass

    # Alternative: Generate all combinations indices and vectorize
    # For N=30, itertools.combinations is fast enough.
    # Let's try a hybrid: pre-compute differences?
    
    # Actually, let's just use the itertools approach but ensure it's tight.
    # Or better, use numpy to compute all pairwise cross products?
    # det(i,j,k) = (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i)
    
    # Let's use a simple loop but optimized.
    counts = Counter()
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n - 1):
            xj, yj = x[j], y[j]
            # Vectorize over k
            xk = x[j+1:]
            yk = y[j+1:]
            
            # det = (xj-xi)*(yk-yi) - (xk-xi)*(yj-yi)
            dx_j = xj - xi
            dy_j = yj - yi
            dx_k = xk - xi
            dy_k = yk - yi
            
            dets = abs(dx_j * dy_k - dx_k * dy_j)
            
            # Filter out zeros
            dets = dets[dets > 0]
            if len(dets) > 0:
                # Update counter
                c = Counter(dets.tolist())
                for k, v in c.items():
                    counts[k] += v
                    
    if not counts:
        return 0
    return max(counts.values())

def _random_points(n=30, bounds=30):
    """Generate n distinct random points in [-bounds, bounds]^2."""
    pts = set()
    while len(pts) < n:
        x = random.randint(-bounds, bounds)
        y = random.randint(-bounds, bounds)
        pts.add((x, y))
    return list(pts)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin
    
    best_score = 0
    best_points = None
    
    # Initialize with a few random starts
    n_starts = 5
    
    for start_idx in range(n_starts):
        if time.time() - start_time > time_limit:
            break
            
        points = _random_points(30)
        current_score = _score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points[:]
            
        # Local Search / Hill Climbing
        # Try moving each point to a random nearby location or swapping
        stuck_count = 0
        max_iterations = 10000
        
        while stuck_count < 100 and time.time() - start_time < time_limit:
            improved = False
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = points[idx]
            
            # Try a few random moves for this point
            moves_tried = 0
            while moves_tried < 50 and not improved:
                # Random move: shift by -2 to 2
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                    
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Check bounds
                if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                    moves_tried += 1
                    continue
                    
                # Check distinctness
                if any(points[i] == (new_x, new_y) for i in range(30) if i != idx):
                    moves_tried += 1
                    continue
                
                # Make the move
                points[idx] = (new_x, new_y)
                
                # Evaluate
                new_score = _score(points)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    # Update global best
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points[:]
                    stuck_count = 0
                else:
                    # Revert
                    points[idx] = (old_x, old_y)
                    moves_tried += 1
                    
            if not improved:
                stuck_count += 1
                
    if best_points is None:
        # Fallback
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points
# EVOLVE-BLOCK-END
