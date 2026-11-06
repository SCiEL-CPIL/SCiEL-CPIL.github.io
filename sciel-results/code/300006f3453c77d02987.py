"""Rewritten program using max-clique / greedy heuristic for equal-area triangles."""
import numpy as np
from itertools import combinations

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Precompute a good base set of 30 points.
    2. For candidate target areas A, build a graph where an edge exists between two points
       if there exists a third point in the set such that the triple has doubled area A.
       Actually, we need a hypergraph approach: for a fixed A, find the maximum number of
       triples with doubled area A. This is equivalent to finding a large subset of points
       where many triples have area A.
    
    Simpler heuristic: 
    - Start with a structured set (e.g., grid).
    - Compute all triple areas.
    - Find the most common positive area A*.
    - Try to improve by swapping points to increase the count of triples with area A*.
    
    Even simpler but effective: use a lattice structure. Points on a 2D lattice tend to have
    many equal-area triangles. For example, a 6x5 grid gives many equal areas.
    
    Let's use a more systematic search:
    - Generate candidate point sets from structured patterns (grids, arithmetic progressions).
    - For each candidate set, compute the max multiplicity.
    - Also try local optimization: start from a good seed, swap points to improve.
    """
    
    def doubled_area(p1, p2, p3):
        """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    
    def max_multiplicity(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        from collections import Counter
        areas = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    a = doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        if not areas:
            return 0
        counter = Counter(areas)
        return max(counter.values())
    
    def vectorized_areas(points):
        """Compute all doubled areas using NumPy for speed."""
        n = len(points)
        pts = np.array(points)
        # Compute all triples
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = abs(int(pts[i,0]*(pts[j,1]-pts[k,1]) + pts[j,0]*(pts[k,1]-pts[i,1]) + pts[k,0]*(pts[i,1]-pts[j,1])))
                    if a > 0:
                        areas.append(a)
        from collections import Counter
        if not areas:
            return 0
        return max(Counter(areas).values())
    
    # Start with a structured set: 6x5 grid
    base_set = [(x, y) for x in range(6) for y in range(5)]
    best_score = vectorized_areas(base_set)
    best_points = base_set[:]
    
    # Try different grid configurations
    import random
    random.seed(42)
    
    # Try various structured sets
    candidates = []
    
    # Grids of different sizes
    for w in range(4, 8):
        for h in range(4, 8):
            if w * h >= 30:
                # Take first 30 from the grid
                pts = [(x, y) for x in range(w) for y in range(h)][:30]
                if len(pts) == 30:
                    candidates.append(pts)
    
    # Arithmetic progression based sets
    for step in range(1, 6):
        for offset_x in range(0, 30, step):
            for offset_y in range(0, 30, step):
                pts = []
                for i in range(30):
                    x = offset_x + (i % 6) * step
                    y = offset_y + (i // 6) * step
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
                    else:
                        break
                if len(pts) == 30:
                    # Check for duplicates
                    if len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Evaluate all candidates
    for cand in candidates:
        score = vectorized_areas(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
    
    # Local optimization: try swapping points
    # For efficiency, we'll do a few rounds of local search
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    for iteration in range(5):  # 5 rounds of local search
        improved = False
        current_set = set(best_points)
        
        # Try replacing one point at a time
        for i in range(30):
            old_point = best_points[i]
            
            # Try a few random replacements
            for _ in range(20):
                new_point = random.choice(all_valid_points)
                if new_point in current_set:
                    continue
                
                # Create new set
                new_points = best_points[:]
                new_points[i] = new_point
                
                # Quick check: if many points are collinear, skip
                score = vectorized_areas(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    current_set = set(best_points)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_points

# EVOLVE-BLOCK-END
