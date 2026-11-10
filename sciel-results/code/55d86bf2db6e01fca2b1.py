"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter, defaultdict

def _canonical_line(p1, p2):
    """Return a canonical tuple (A, B, C) for the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Normalize by GCD
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Ensure a unique sign representation
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _score(points):
    """
    Calculate the score: number of lines with exactly 4 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    
    score = 0
    for count in line_counts.values():
        if count >= 10:  # C(5,2)=10 pairs for 5 points
            return -1
        if count == 6:  # C(4,2)=6 pairs for 4 points
            score += 1
    return score

def _count_lines_with_k_points(points, k):
    """Count lines with exactly k points."""
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    
    count = 0
    for count_pairs in line_counts.values():
        if count_pairs == k * (k - 1) // 2:
            count += 1
    return count

def _has_five_collinear(points):
    """Check if any line has 5 or more points."""
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    
    for count_pairs in line_counts.values():
        if count_pairs >= 10:  # C(5,2)=10
            return True
    return False

def _line_counts_from_points(points):
    """Get line pair counts for a set of points."""
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    return line_counts

def _would_create_five_collinear(points, new_point, line_counts):
    """
    Check if adding new_point to points would create a line with 5+ points.
    points: list of existing points
    new_point: candidate point to add
    line_counts: Counter of line pair counts for existing points
    """
    for existing_point in points:
        line = _canonical_line(existing_point, new_point)
        # Count how many existing points are on this line
        # We need to count points on the line, not pairs
        # A line with k points has C(k,2) pairs
        # If we add new_point, and there are already m points on this line,
        # then after adding, there will be m+1 points
        # We need to check if m+1 >= 5, i.e., m >= 4
        
        # Count existing points on this line
        count_on_line = 0
        for p in points:
            if p == existing_point:
                continue
            # Check if p is on the line through existing_point and new_point
            x1, y1 = existing_point
            x2, y2 = new_point
            xp, yp = p
            # Cross product to check collinearity
            if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                count_on_line += 1
        
        # Including existing_point, total on line = count_on_line + 1
        # After adding new_point, total = count_on_line + 2
        if count_on_line + 2 >= 5:
            return True
    
    return False

def _backtrack_build(points, line_counts, target_size, best_points, best_score, deadline, 
                     search_range=30, max_candidates=1000):
    """
    Backtracking search to build a valid configuration with target_size points.
    Returns (best_points, best_score) found during this search.
    """
    n = len(points)
    
    if n == target_size:
        score = _score(points)
        if score > best_score[0]:
            best_score[0] = score
            best_points[0] = [p[:] for p in points]
        return
    
    if time.time() > deadline:
        return
    
    # Generate candidate points to try
    # Strategy: try points that are likely to form lines with existing points
    candidates = []
    
    # First, try to extend existing lines
    # For each pair of existing points, find integer points on the line within range
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_step = dx // g
            dy_step = dy // g
            
            # Find integer points on this line within range
            # Parametric: (x1 + t*dx_step, y1 + t*dy_step) for integer t
            # Need |x|, |y| <= search_range
            for t in range(-search_range, search_range + 1):
                x = x1 + t * dx_step
                y = y1 + t * dy_step
                if abs(x) <= search_range and abs(y) <= search_range:
                    # Check if this point is already in points
                    pt = (x, y)
                    if pt not in points:
                        candidates.append(pt)
    
    # Also add some random candidates
    for _ in range(50):
        x = random.randint(-search_range, search_range)
        y = random.randint(-search_range, search_range)
        pt = (x, y)
        if pt not in points:
            candidates.append(pt)
    
    # Deduplicate
    candidates = list(set(candidates))
    
    # Limit candidates to avoid excessive branching
    if len(candidates) > max_candidates:
        # Prioritize candidates that are on lines with existing points
        def candidate_priority(pt):
            # Count how many existing points are on lines through pt
            count = 0
            for p in points:
                line = _canonical_line(p, pt)
                # Check if other points are on this line
                for q in points:
                    if q != p:
                        if _canonical_line(p, q) == line:
                            count += 1
                            break
            return -count
        
        candidates.sort(key=candidate_priority)
        candidates = candidates[:max_candidates]
    
    # Try each candidate
    for candidate in candidates:
        if time.time() > deadline:
            return
        
        # Check if adding this point would create 5 collinear
        if _would_create_five_collinear(points, candidate, line_counts):
            continue
        
        # Add the point
        points.append(candidate)
        
        # Update line counts
        for existing_point in points[:-1]:
            line = _canonical_line(existing_point, candidate)
            line_counts[line] += 1
        
        # Recurse
        _backtrack_build(points, line_counts, target_size, best_points, best_score, 
                        deadline, search_range, max_candidates)
        
        # Undo
        points.pop()
        for existing_point in points:
            line = _canonical_line(existing_point, candidate)
            line_counts[line] -= 1
    
    return

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Backtracking search
    # Start with a small seed and grow
    seed_configs = [
        # 4x4 grid corners
        [(0, 0), (0, 3), (3, 0), (3, 3)],
        # Random small seeds
        [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(4)],
        [(random.randint(-15, 15), random.randint(-15, 15)) for _ in range(4)],
    ]
    
    for seed in seed_configs:
        if time.time() > deadline:
            break
        
        # Deduplicate seed
        seed = list(set(seed))
        if len(seed) < 4:
            continue
        
        # Check seed validity
        if _has_five_collinear(seed):
            continue
        
        points = [list(p) for p in seed]
        line_counts = _line_counts_from_points(points)
        
        # Run backtracking
        bp = [None]
        bs = [-1]
        _backtrack_build(points, line_counts, 16, bp, bs, deadline, 
                        search_range=25, max_candidates=500)
        
        if bs[0] > best_score:
            best_score = bs[0]
            best_points = bp[0]
    
    # Strategy 2: Local search from best found
    if best_points is not None:
        current_points = [p[:] for p in best_points]
        current_score = best_score
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve < max_no_improve:
            idx = random.randint(0, 15)
            old_point = current_points[idx]
            
            # Generate new point
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            new_point = [new_x, new_y]
            
            # Check distinctness
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
            
            current_points[idx] = new_point
            new_score = _score(current_points)
            
            if new_score == -1:
                current_points[idx] = old_point
                no_improve += 1
            elif new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                no_improve = 0
            else:
                current_points[idx] = old_point
                no_improve += 1
    
    # Fallback
    if best_points is None or best_score < 0:
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_points

# EVOLVE-BLOCK-END
