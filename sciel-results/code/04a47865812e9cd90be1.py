"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import defaultdict

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _evaluate(points):
    """Find the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    area_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    area_counts[a] = area_counts.get(a, 0) + 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def _greedy_select_for_target(all_points, target_area, k=30):
    """
    Greedy set-cover style selection: pick points that maximize the number of triangles
    with exactly target_area. We use a greedy approach where at each step we add the point
    that contributes the most new target-area triangles with the already-selected points.
    """
    n = len(all_points)
    if n < k:
        return None
    
    # Precompute area matrix: area[i][j] = doubled area of triangle formed by 
    # reference point (0,0) with points i and j
    # Actually, we need areas relative to pairs of selected points.
    # Better approach: precompute for each pair (i,j), what areas they form with each other point k.
    
    # For efficiency, precompute area between each pair of points
    # area[i][j] = doubled area of triangle with vertices all_points[i], all_points[j], and a fixed reference
    # But we need triangles formed by 3 points from our set.
    
    # Alternative: For each candidate point p, count how many triangles of area target_area 
    # it forms with pairs from the already selected set.
    
    # Precompute: for each point index i, and for each pair (j,k) where j<k, 
    # the area of triangle (i,j,k). This is O(n^3) which is too much for n~3600.
    
    # Instead, use a smarter approach: 
    # Fix a reference point r. For any two other points p, q, the area of triangle (r,p,q) is determined.
    # But we're not fixing a reference.
    
    # Let's use a different greedy: 
    # 1. Start with an empty set S.
    # 2. At each step, for each candidate point p not in S, compute how many new triangles 
    #    of area target_area are formed by adding p to S.
    #    A new triangle is formed by p and two points already in S.
    #    So we need: for each pair (a,b) in S, check if area(p,a,b) == target_area.
    # 3. Add the point that maximizes this count.
    # 4. Repeat until we have k points.
    
    # Precompute area lookup: area_lookup[p][a][b] would be too big.
    # Instead, for each point p, precompute a set of pairs (a,b) such that area(p,a,b) == target_area.
    # But that's still O(n^3) total.
    
    # For n=3600, n^3 = 46.6 billion, which is too much.
    # We need a smarter approach.
    
    # Alternative: Use the fact that area(p,a,b) = |det(a-p, b-p)| = |det(a, b) - det(a, p) - det(p, b)|
    # where det(u,v) = u_x * v_y - u_y * v_x (treating points as vectors from origin).
    
    # Let's precompute for each point p, the vector v_p = (x_p, y_p).
    # Then area(p,a,b) = |det(v_a - v_p, v_b - v_p)| = |det(v_a, v_b) - det(v_a, v_p) - det(v_p, v_b) + det(v_p, v_p)|
    # Since det(v_p, v_p) = 0:
    # area(p,a,b) = |det(v_a, v_b) - det(v_a, v_p) - det(v_p, v_b)|
    # Note: det(v_p, v_b) = -det(v_b, v_p), so:
    # area(p,a,b) = |det(v_a, v_b) - det(v_a, v_p) + det(v_b, v_p)|
    
    # Let d(p,q) = det(v_p, v_q) = x_p * y_q - y_p * x_q
    # Then area(p,a,b) = |d(a,b) - d(a,p) + d(b,p)| = |d(a,b) + d(p,a) + d(b,p)| ... let me re-derive.
    
    # Actually: det(v_a - v_p, v_b - v_p) = det(v_a, v_b) - det(v_a, v_p) - det(v_p, v_b) + det(v_p, v_p)
    # = d(a,b) - d(a,p) - d(p,b)
    # And d(p,b) = -d(b,p), so:
    # = d(a,b) - d(a,p) + d(b,p)
    
    # So area(p,a,b) = |d(a,b) - d(a,p) + d(b,p)|
    
    # For fixed p and target area A, we need: d(a,b) - d(a,p) + d(b,p) = ±A
    # i.e., d(a,b) = A + d(a,p) - d(b,p) or d(a,b) = -A + d(a,p) - d(b,p)
    
    # This is still complex. Let me try a simpler greedy with limited precomputation.
    
    # For the greedy step, given current set S (size s), and candidate p:
    # We need to count pairs (a,b) in S such that area(p,a,b) == target_area.
    # This is O(s^2) per candidate, and we have O(n) candidates, so O(n*s^2) per iteration.
    # With s up to 30 and n up to 3600, this is 3600 * 900 = 3.24 million per iteration, times 30 iterations = ~100 million.
    # This might be borderline but let's try with optimizations.
    
    selected = []
    selected_set = set()
    
    for step in range(k):
        best_point = None
        best_count = -1
        
        for idx, p in enumerate(all_points):
            if idx in selected_set:
                continue
            
            # Count new triangles of area target_area formed by p with pairs in selected
            count = 0
            s = len(selected)
            for i in range(s):
                for j in range(i+1, s):
                    a = selected[i]
                    b = selected[j]
                    area = _doubled_area(p, a, b)
                    if area == target_area:
                        count += 1
            
            if count > best_count:
                best_count = count
                best_point = idx
        
        if best_point is None:
            break
        
        selected.append(all_points[best_point])
        selected_set.add(best_point)
    
    if len(selected) != k:
        return None
    return selected

def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area.
    
    Strategy: Use greedy set-cover approach for various target areas.
    """
    # Generate all candidate points in [-30, 30]^2
    all_points = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            all_points.append((x, y))
    
    n = len(all_points)
    
    best_score = 0
    best_points = None
    
    # Try various target areas
    target_areas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]
    
    for target_area in target_areas:
        points = _greedy_select_for_target(all_points, target_area, k=30)
        if points is None:
            continue
        
        score = _evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Also try some heuristic configurations as fallback
    if best_points is None or best_score < 10:
        # Try small grids
        for w in range(3, 11):
            for h in range(3, 11):
                if w * h < 30:
                    continue
                points = []
                x_start = -w // 2
                y_start = -h // 2
                for x in range(x_start, x_start + w):
                    for y in range(y_start, y_start + h):
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            points.append((x, y))
                        if len(points) == 30:
                            break
                    if len(points) == 30:
                        break
                if len(points) != 30 or len(set(points)) != 30:
                    continue
                score = _evaluate(points)
                if score > best_score:
                    best_score = score
                    best_points = points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
