import random
import numpy as np
from collections import Counter
import time

random.seed(42)

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area (absolute determinant).
    """
    
    def compute_areas(points):
        # points is a numpy array of shape (N, 2)
        # Compute doubled area for all triples
        # Using broadcasting: 
        # p1 = points[:, None, :] (N, 1, 2)
        # p2 = points[None, :, :] (1, N, 2)
        # p3 = points (N, 2) -> this doesn't work directly for triples i<j<k
        
        # Efficient way:
        # x1, y1 = points[:, 0], points[:, 1]
        # x2, y2 = ...
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # Doubled Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        
        N = len(points)
        # We can use the formula: 2*Area = |(P2-P1) x (P3-P1)|
        # Let's generate indices i, j, k with i < j < k
        # This is slow in Python loops. 
        # Alternative: 
        # For each pair (i, j), the area with k is |cross(p_j - p_i, p_k - p_i)|
        # This is still O(N^3).
        # N=30, N^3 = 27000. This is very fast in NumPy if vectorized properly.
        
        # Let's just use a loop over i, and vectorize j, k?
        # Or just use itertools if it's fast enough? 
        # 30 choose 3 is 4060. 
        # 4060 iterations in Python is negligible. 
        # The bottleneck in the previous code was likely the `Counter` update or the list creation.
        # Let's stick to a Python loop but optimized.
        
        areas = []
        pts = points
        for i in range(N-2):
            x1, y1 = pts[i]
            for j in range(i+1, N-1):
                x2, y2 = pts[j]
                # Vectorize over k > j
                # pts[k] for k in j+1..N-1
                x3s = pts[j+1:, 0]
                y3s = pts[j+1:, 1]
                # doubled_area = abs(x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
                # = abs(x1*y2 - x1*y3 + x2*y3 - x2*y1 + x3*y1 - x3*y2)
                # = abs( (x1*y2 - x2*y1) + y3*(x2-x1) + x3*(y1-y2) )
                base = x1*y2 - x2*y1
                dy = y1 - y2
                dx = x2 - x1
                areas_k = np.abs(base + y3s*dx + x3s*dy)
                # Filter out 0
                areas.extend(areas_k[areas_k > 0].tolist())
        
        return areas

    def get_max_multiplicity(points):
        areas = compute_areas(points)
        if not areas:
            return 0, 0
        counts = Counter(areas)
        max_mult, max_area = counts.most_common(1)[0]
        return max_mult, max_area

    # Initial configuration: 5x6 grid
    # Centered at 0,0 to allow moves
    w, h = 5, 6
    xs = np.arange(w) - w//2
    ys = np.arange(h) - h//2
    points = np.array([(x, y) for x in xs for y in ys], dtype=int)
    
    current_points = points.copy()
    current_mult, current_area = get_max_multiplicity(current_points)
    
    best_points = current_points.copy()
    best_mult = current_mult
    
    # Local Search
    # Time limit: 50 seconds
    start_time = time.time()
    max_time = 50.0
    
    # Precompute valid empty spots? No, just check on the fly.
    # Grid bounds: -30 to 30
    
    while time.time() - start_time < max_time:
        # Pick a random point index
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Try to move this point to a random new location
        # Sample a few candidates
        improved = False
        for _ in range(10):
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = np.array([new_x, new_y])
            
            # Check collision
            if np.any(np.all(current_points == new_point, axis=1)):
                continue
                
            # Save old point
            current_points[idx] = new_point
            
            new_mult, new_area = get_max_multiplicity(current_points)
            
            if new_mult > current_mult:
                current_mult = new_mult
                current_area = new_area
                if new_mult > best_mult:
                    best_mult = new_mult
                    best_points = current_points.copy()
                improved = True
                break # Accept and restart search from here
            else:
                # Revert
                current_points[idx] = np.array([old_x, old_y])
        
        # Optional: Try swapping two points
        if not improved and random.random() < 0.1:
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 != idx2:
                p1 = current_points[idx1].copy()
                p2 = current_points[idx2].copy()
                current_points[idx1] = p2
                current_points[idx2] = p1
                
                new_mult, new_area = get_max_multiplicity(current_points)
                
                if new_mult > current_mult:
                    current_mult = new_mult
                    current_area = new_area
                    if new_mult > best_mult:
                        best_mult = new_mult
                        best_points = current_points.copy()
                else:
                    # Revert
                    current_points[idx1] = p1
                    current_points[idx2] = p2

    return [tuple(p) for p in best_points]
