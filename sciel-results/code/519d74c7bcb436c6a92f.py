import numpy as np
from math import gcd
import random
import time

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2:
        return None
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    return _normalize_line(A, B, C)

def _calculate_score(points):
    """
    Calculate the sum of the top 27 line incidences for the given points.
    """
    N = len(points)
    line_counts = {}
    
    # Generate all unique lines
    for i in range(N):
        for j in range(i + 1, N):
            norm = _line_from_points(points[i], points[j])
            if norm is not None:
                if norm not in line_counts:
                    line_counts[norm] = 0
                line_counts[norm] += 1 # This is not right, this counts pairs, not incidences.
    
    # We need incidence count: number of points on the line.
    # The loop above counts pairs.
    # If a line has k points, it has k*(k-1)/2 pairs.
    # We can recover k from pairs, but it's easier to just count points directly for each unique line.
    
    unique_lines = list(line_counts.keys())
    
    counts = []
    for line in unique_lines:
        A, B, C = line
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        counts.append(count)
        
    # Sort counts descending
    counts.sort(reverse=True)
    
    # Sum top 27
    if len(counts) >= 27:
        return sum(counts[:27])
    else:
        # If fewer than 27 lines, sum all (and pad with 0? No, we need 27 lines.
        # If there are fewer than 27 unique lines, we can't form 27 distinct lines.
        # But with 54 points, there are always >= 27 lines unless points are very degenerate.
        # Actually, if all points are on 1 line, there is 1 line.
        # The problem requires returning 27 lines.
        # If we don't have 27 distinct lines, we must add "dummy" lines with 0 or 1 incidence?
        # The task says "lines a (27, 3) array".
        # It doesn't say lines must have >= 2 points.
        # So if we have < 27 unique lines from pairs, we can add arbitrary lines (e.g. x=large_number) with 0 points.
        # So sum is just sum of all available line counts.
        return sum(counts)

def _generate_initial_points():
    # Start with a 6x9 grid, which is a good seed
    xs = np.arange(6, dtype=np.int64)
    ys = np.arange(9, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _perturb_point(points, idx, delta_range=3):
    """
    Create a new points array with one point perturbed.
    Returns None if the new point collides with an existing one.
    """
    new_points = points.copy()
    p = points[idx]
    dx = random.randint(-delta_range, delta_range)
    dy = random.randint(-delta_range, delta_range)
    if dx == 0 and dy == 0:
        return None
    new_p = (p[0] + dx, p[1] + dy)
    
    # Check collision
    for i in range(len(points)):
        if i == idx:
            continue
        if points[i][0] == new_p[0] and points[i][1] == new_p[1]:
            return None
            
    new_points[idx] = new_p
    return new_points

def _select_top_k_lines(points, k=27):
    """
    Select the k distinct lines with the highest incidences.
    If there are fewer than k unique lines, pad with arbitrary lines.
    """
    N = len(points)
    line_counts = {}
    
    # Find all unique lines
    for i in range(N):
        for j in range(i + 1, N):
            norm = _line_from_points(points[i], points[j])
            if norm is not None:
                if norm not in line_counts:
                    line_counts[norm] = 0
                # We just need the set of lines. Counting is done separately for scoring.
                # But for selection, we need the count.
                # Let's just store the line.
                line_counts[norm] = True
                
    unique_lines = list(line_counts.keys())
    
    # Calculate counts for these lines
    scored_lines = []
    for line in unique_lines:
        A, B, C = line
        count = np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)
        scored_lines.append((count, line))
        
    # Sort by count descending
    scored_lines.sort(key=lambda x: -x[0])
    
    selected = []
    for count, line in scored_lines:
        if len(selected) >= k:
            break
        selected.append(line)
        
    # Pad if necessary
    if len(selected) < k:
        # Add arbitrary lines that don't intersect the points? Or just any valid line.
        # To be safe, add lines like x = 1000 + i
        used_norms = set(selected)
        i = 0
        while len(selected) < k:
            # Vertical line x = 1000 + i
            # A=1, B=0, C=-(1000+i)
            norm = (1, 0, -(1000 + i))
            if norm not in used_norms:
                selected.append(norm)
                used_norms.add(norm)
            i += 1
            
    lines = np.array(selected, dtype=np.int64)
    return lines

def szemeredi_trotter_incidences():
    """
    Optimize point configuration using Simulated Annealing.
    """
    points = _generate_initial_points()
    current_score = _calculate_score(points)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    start_time = time.time()
    max_time = 550  # Leave 10s buffer
    
    best_score = current_score
    best_points = points.copy()
    
    while T > T_min and (time.time() - start_time) < max_time:
        idx = random.randint(0, len(points) - 1)
        new_points = _perturb_point(points, idx, delta_range=4)
        
        if new_points is None:
            T *= cooling
            continue
            
        new_score = _calculate_score(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                
        T *= cooling
        
    lines = _select_top_k_lines(best_points, 27)
    
    # Final validation/rounding
    best_points = np.round(best_points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    return best_points, lines
