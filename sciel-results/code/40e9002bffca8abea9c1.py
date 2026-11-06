"""Improved program using structured lattice initialization and enhanced local search."""
import numpy as np
from collections import Counter
import random
from itertools import combinations

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Generate candidate point sets from structured lattice patterns.
    2. Use vectorized computation to quickly evaluate max multiplicity.
    3. Apply enhanced local search with structure-aware move generation.
    """
    
    def compute_max_multiplicity(points):
        """Compute the maximum multiplicity of any positive doubled area using vectorized ops."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        # Generate all valid triples
        idx_i, idx_j, idx_k = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
        
        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        idx_k = np.array(idx_k)
        
        # Compute areas
        xi = pts[idx_i, 0]
        yi = pts[idx_i, 1]
        xj = pts[idx_j, 0]
        yj = pts[idx_j, 1]
        xk = pts[idx_k, 0]
        yk = pts[idx_k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Filter positive areas
        positive_mask = areas > 0
        if not np.any(positive_mask):
            return 0
        
        positive_areas = areas[positive_mask]
        counter = Counter(positive_areas.tolist())
        return max(counter.values())
    
    # Generate diverse candidate sets from structured patterns
    candidates = []
    
    # Pattern 1: Rectangular grids
    for w in range(4, 9):
        for h in range(4, 9):
            if w * h >= 30:
                pts = [(x, y) for x in range(w) for y in range(h)]
                if len(pts) >= 30:
                    candidates.append(pts[:30])
    
    # Pattern 2: Centered grids
    for size in range(5, 9):
        center = 15
        pts = []
        for x in range(-size//2, size//2 + 1):
            for y in range(-size//2, size//2 + 1):
                pts.append((center + x, center + y))
        if len(pts) >= 30:
            candidates.append(pts[:30])
    
    # Pattern 3: Diagonal stripes
    for step in [1, 2, 3]:
        for offset in range(0, 30, step):
            pts = []
            for i in range(30):
                x = i % 6
                y = (i // 6) + offset
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            if len(pts) == 30 and len(set(pts)) == 30:
                candidates.append(pts)
    
    # Pattern 4: Hexagonal-like lattice
    for scale in [1, 2, 3]:
        pts = []
        for row in range(6):
            for col in range(5):
                x = col * 2 * scale + (row % 2) * scale
                y = row * scale
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) >= 30:
            candidates.append(pts[:30])
    
    # Pattern 5: Arithmetic progressions in 2D
    for dx in [1, 2, 3]:
        for dy in [1, 2, 3]:
            for ox in range(0, 30, 5):
                for oy in range(0, 30, 5):
                    pts = []
                    for i in range(30):
                        x = ox + (i % 6) * dx
                        y = oy + (i // 6) * dy
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                        else:
                            break
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Evaluate all candidates
    random.seed(42)
    best_score = 0
    best_points = None
    
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        score = compute_max_multiplicity(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = compute_max_multiplicity(best_points)
    
    # Enhanced local search
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    random.shuffle(all_valid_points)
    
    for iteration in range(8):
        improved = False
        current_set = set(best_points)
        
        # Try replacing one point at a time with better candidates
        for i in range(30):
            old_point = best_points[i]
            
            # Try structured replacements: points near old_point or on same lattice
            candidates_to_try = []
            
            # Points in a neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point:
                            candidates_to_try.append(np_)
            
            # Also try some random points
            for _ in range(10):
                np_ = random.choice(all_valid_points)
                if np_ not in current_set and np_ != old_point:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:15]:  # Limit tries per position
                new_points = best_points[:]
                new_points[i] = new_point
                
                score = compute_max_multiplicity(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    current_set = set(best_points)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_points
