import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Initial configuration: 6x5 grid
    points = [(x, y) for x in range(6) for y in range(5)]
    points = np.array(points)
    
    def compute_area_counts(pts):
        """Compute counts of absolute doubled areas for all triples."""
        n = len(pts)
        areas = np.zeros(n * (n-1) * (n-2) // 6, dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Doubled area = |det|
                    d = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - 
                            (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                    if d > 0:
                        areas[idx] = d
                        idx += 1
        areas = areas[:idx]
        if len(areas) == 0:
            return 0, 0
        counts = Counter(areas.tolist())
        max_count = max(counts.values())
        best_area = max(counts, key=counts.get)
        return max_count, best_area
    
    def fast_score(pts):
        """Fast computation of max area multiplicity."""
        n = len(pts)
        # Use vectorized computation
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - 
                            (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())
    
    # Better: use numpy for speed
    def score_numpy(pts):
        n = len(pts)
        # Compute all pairwise differences
        # For efficiency, compute determinants
        areas = []
        for i in range(n):
            pi = pts[i]
            for j in range(i+1, n):
                pij = pts[j] - pi
                for k in range(j+1, n):
                    pik = pts[k] - pi
                    d = abs(pij[0]*pik[1] - pij[1]*pik[0])
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        c = Counter(areas)
        return c.most_common(1)[0][1]
    
    # Even faster: precompute
    def score_fast(pts):
        n = len(pts)
        area_counts = {}
        for i in range(n):
            for j in range(i+1, n):
                dx1 = pts[j,0] - pts[i,0]
                dy1 = pts[j,1] - pts[i,1]
                for k in range(j+1, n):
                    dx2 = pts[k,0] - pts[i,0]
                    dy2 = pts[k,1] - pts[i,1]
                    d = abs(dx1*dy2 - dy1*dx2)
                    if d > 0:
                        area_counts[d] = area_counts.get(d, 0) + 1
        if not area_counts:
            return 0
        return max(area_counts.values())
    
    best_score = score_fast(points)
    best_points = points.copy()
    
    # Simulated annealing
    T = 5.0
    cooling = 0.995
    step_size = 3
    
    # All valid positions
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random point and move it
        idx = random.randint(0, 29)
        old_pos = points[idx].copy()
        
        # Generate new position near old
        if random.random() < 0.7:
            # Small move
            new_pos = old_pos + np.array([random.randint(-step_size, step_size), 
                                           random.randint(-step_size, step_size)])
        else:
            # Larger move
            new_pos = old_pos + np.array([random.randint(-8, 8), 
                                           random.randint(-8, 8)])
        
        # Clamp to bounds
        new_pos = np.clip(new_pos, -30, 30)
        new_pos = new_pos.astype(np.int32)
        
        # Check for collision
        if any(np.array_equal(points[i], new_pos) for i in range(30) if i != idx):
            continue
        
        points[idx] = new_pos
        new_score = score_fast(points)
        
        if new_score >= best_score:
            best_score = new_score
            best_points = points.copy()
        elif random.random() < np.exp((new_score - best_score) / T):
            # Accept worse solution
            pass
        else:
            points[idx] = old_pos
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            step_size = max(1, step_size - 1)
    
    # Final local search polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(30):
            for j in range(i+1, 30):
                # Try swapping
                points[i], points[j] = points[j], points[i]
                new_score = score_fast(points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = points.copy()
                    improved = True
                else:
                    points[i], points[j] = points[j], points[i]
    
    return [tuple(p) for p in best_points]
