# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    
    Uses local search with random restarts to minimize convex 7-gons.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    k = 7
    deadline = time.time() + 600  # 600s limit, use 620 for safety? No, 630 is kill. Use 620.
    
    # Precompute indices for combinations
    # We will generate combinations on the fly or precompute all 7-combinations
    # Precomputing 4.2M combinations of indices takes memory but is fast to iterate.
    # Let's generate them once.
    all_combos = list(combinations(range(n), k))
    
    def is_valid(points):
        """Check validity constraints."""
        if points.shape != (n, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Check minimum distance
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for l in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[l]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    def count_convex_7gons(points):
        """Count convex 7-gons using precomputed combinations and vectorized cross products."""
        # Compute cross products for all triples (i, j, k)
        # cross[i, j, k] = (p_j - p_i) x (p_k - p_j)
        # We need (p_j - p_i) x (p_k - p_i) ? No, the turn is p_i -> p_j -> p_k.
        # Vector v1 = p_j - p_i, v2 = p_k - p_j.
        # cross = v1_x * v2_y - v1_y * v2_x
        
        # To speed up, we can compute the cross product for the specific combinations.
        # But 4.2M * 7 is 30M operations.
        # Let's try to vectorize the check for a batch of combinations?
        # Or just loop in Python but efficiently.
        
        count = 0
        # Precompute pairwise differences?
        # p[i] - p[j]
        
        for combo in all_combos:
            # combo is (i0, i1, i2, i3, i4, i5, i6)
            # Check turns:
            # (i0, i1, i2), (i1, i2, i3), ..., (i5, i6, i0), (i6, i0, i1)
            
            # Unpack for speed
            i0, i1, i2, i3, i4, i5, i6 = combo
            
            # Turn 1: i0 -> i1 -> i2
            v1x = points[i1, 0] - points[i0, 0]
            v1y = points[i1, 1] - points[i0, 1]
            v2x = points[i2, 0] - points[i1, 0]
            v2y = points[i2, 1] - points[i1, 1]
            cp1 = v1x * v2y - v1y * v2x
            if cp1 <= 1e-9:
                continue
                
            # Turn 2: i1 -> i2 -> i3
            # v1 is now i2-i1, v2 is i3-i2
            v1x = points[i2, 0] - points[i1, 0]
            v1y = points[i2, 1] - points[i1, 1]
            v2x = points[i3, 0] - points[i2, 0]
            v2y = points[i3, 1] - points[i2, 1]
            cp2 = v1x * v2y - v1y * v2x
            if cp2 <= 1e-9:
                continue

            # Turn 3: i2 -> i3 -> i4
            v1x = points[i3, 0] - points[i2, 0]
            v1y = points[i3, 1] - points[i2, 1]
            v2x = points[i4, 0] - points[i3, 0]
            v2y = points[i4, 1] - points[i3, 1]
            cp3 = v1x * v2y - v1y * v2x
            if cp3 <= 1e-9:
                continue

            # Turn 4: i3 -> i4 -> i5
            v1x = points[i4, 0] - points[i3, 0]
            v1y = points[i4, 1] - points[i3, 1]
            v2x = points[i5, 0] - points[i4, 0]
            v2y = points[i5, 1] - points[i4, 1]
            cp4 = v1x * v2y - v1y * v2x
            if cp4 <= 1e-9:
                continue

            # Turn 5: i4 -> i5 -> i6
            v1x = points[i5, 0] - points[i4, 0]
            v1y = points[i5, 1] - points[i4, 1]
            v2x = points[i6, 0] - points[i5, 0]
            v2y = points[i6, 1] - points[i5, 1]
            cp5 = v1x * v2y - v1y * v2x
            if cp5 <= 1e-9:
                continue

            # Turn 6: i5 -> i6 -> i0
            v1x = points[i6, 0] - points[i5, 0]
            v1y = points[i6, 1] - points[i5, 1]
            v2x = points[i0, 0] - points[i6, 0]
            v2y = points[i0, 1] - points[i6, 1]
            cp6 = v1x * v2y - v1y * v2x
            if cp6 <= 1e-9:
                continue

            # Turn 7: i6 -> i0 -> i1
            v1x = points[i0, 0] - points[i6, 0]
            v1y = points[i0, 1] - points[i6, 1]
            v2x = points[i1, 0] - points[i0, 0]
            v2y = points[i1, 1] - points[i0, 1]
            cp7 = v1x * v2y - v1y * v2x
            if cp7 <= 1e-9:
                continue

            count += 1
            
        return count

    def local_search(points, max_iters=1000):
        """Perform local search to reduce convex 7-gons."""
        best_points = points.copy()
        best_count = count_convex_7gons(best_points)
        
        # Use a fixed seed for reproducibility in local search steps? 
        # No, global seed is set outside.
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try moving a random point
            idx = np.random.randint(n)
            orig = best_points[idx].copy()
            
            # Try a few random perturbations
            improved = False
            # Adaptive step size
            step_size = 0.05 if it < 100 else 0.01
            
            for trial in range(30):
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                new_x = np.clip(orig[0] + dx, 0, 1)
                new_y = np.clip(orig[1] + dy, 0, 1)
                best_points[idx] = [new_x, new_y]
                
                if not is_valid(best_points):
                    best_points[idx] = orig
                    continue
                
                new_count = count_convex_7gons(best_points)
                if new_count < best_count:
                    best_count = new_count
                    improved = True
                    break
                else:
                    best_points[idx] = orig
            
            if not improved:
                # Try larger moves or restart
                if it > 100:
                    break
        
        return best_points, best_count
    
    # Start with random initialization
    np.random.seed(42)
    best_points = None
    best_count = float('inf')
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        # Random initialization
        points = np.random.rand(n, 2)
        if not is_valid(points):
            continue
        
        # Local search
        result_points, result_count = local_search(points, max_iters=200)
        
        if result_count < best_count:
            best_count = result_count
            best_points = result_points.copy()
        
        if best_count == 1: # Theoretical minimum
            break
    
    # If no valid result found, return a simple valid configuration
    if best_points is None or not is_valid(best_points):
        # Create a valid configuration: points on a slightly perturbed grid
        best_points = np.zeros((n, 2))
        cols = 6
        rows = 6
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                best_points[count, 0] = (j + 0.5) / cols + np.random.uniform(-0.01, 0.01)
                best_points[count, 1] = (i + 0.5) / rows + np.random.uniform(-0.01, 0.01)
                count += 1
        # Add remaining points
        while count < n:
            best_points[count, 0] = np.random.uniform(0.05, 0.95)
            best_points[count, 1] = np.random.uniform(0.05, 0.95)
            count += 1
        
        # Ensure validity
        for attempt in range(1000):
            if is_valid(best_points):
                break
            # Perturb a random point
            idx = np.random.randint(n)
            best_points[idx, 0] += np.random.uniform(-0.01, 0.01)
            best_points[idx, 1] += np.random.uniform(-0.01, 0.01)
            best_points[idx] = np.clip(best_points[idx], 0.01, 0.99)
    
    return best_points


# EVOLVE-BLOCK-END
