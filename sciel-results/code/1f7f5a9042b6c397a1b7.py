import time
import random
import numpy as np
from collections import Counter

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    if dy == 0:
        return ('h', y1)
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return ('s', dx, dy, c)

def _evaluate(points):
    """
    Returns (is_valid, score).
    is_valid: 16 distinct integer points, |x|,|y|<=1000, no 5 collinear.
    score: number of lines containing exactly 4 points.
    """
    n = len(points)
    if n != 16:
        return False, 0
    
    # Check bounds and distinctness
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    if any(abs(x) > 1000 or abs(y) > 1000 for x, y in points):
        return False, 0
        
    pts_set = set((x, y) for x, y in points)
    if len(pts_set) != 16:
        return False, 0

    # Count pairs per line
    line_counts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            key = _line_key(xi, yi, xj, yj)
            line_counts[key] = line_counts.get(key, 0) + 1

    score = 0
    for key, pair_count in line_counts.items():
        # C(k, 2) = pair_count
        # k=4 => 6
        # k=5 => 10
        if pair_count == 6:
            score += 1
        elif pair_count >= 10:
            return False, 0
            
    return True, score

def _find_candidates(points, exclude_idx):
    """
    Find candidate points for position exclude_idx.
    Strategy: Find lines with exactly 3 points among the other 15 points.
    Candidate points are integer points on these lines, within bounds, not occupied.
    """
    others = [points[i] for i in range(16) if i != exclude_idx]
    others_set = set((x, y) for x, y in others)
    
    # Find all lines defined by pairs of 'others'
    # Count how many 'others' are on each line
    line_counts = {}
    for i in range(15):
        x1, y1 = others[i]
        for j in range(i + 1, 15):
            x2, y2 = others[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    # Identify lines with exactly 3 points (C(3,2)=3 pairs)
    candidate_lines = []
    for key, count in line_counts.items():
        if count == 3:
            candidate_lines.append(key)
            
    if not candidate_lines:
        return []
        
    candidates = []
    # Limit the number of lines we process to avoid explosion, though 15 points is small
    # Randomly sample some lines if there are too many
    if len(candidate_lines) > 20:
        candidate_lines = random.sample(candidate_lines, 20)
        
    for key in candidate_lines:
        if key[0] == 'v':
            x = key[1]
            # y can be any integer in [-1000, 1000]
            # We pick a few random y values
            for _ in range(5):
                y = random.randint(-1000, 1000)
                if (x, y) not in others_set:
                    candidates.append([x, y])
        elif key[0] == 'h':
            y = key[1]
            for _ in range(5):
                x = random.randint(-1000, 1000)
                if (x, y) not in others_set:
                    candidates.append([x, y])
        else:
            _, dx, dy, c = key
            # Line: dy*x - dx*y = c  =>  dy*x - c = dx*y
            # We need integer solutions.
            # Pick a random x in range, check if y is integer and in range.
            for _ in range(20):
                x = random.randint(-1000, 1000)
                numerator = dy * x - c
                if numerator % dx == 0:
                    y = numerator // dx
                    if -1000 <= y <= 1000 and (x, y) not in others_set:
                        candidates.append([x, y])
                        
    return candidates

def construct():
    best_score = 0
    best_points = None
    deadline = _START_TIME + _DEADLINE

    # Initial: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    is_valid, score = _evaluate(grid)
    if is_valid:
        best_score = score
        best_points = [list(p) for p in grid]
    else:
        # Fallback, though grid should be valid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = 0

    random.seed(42)
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Local search
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_point = current_points[idx]
        
        # Try smart candidates first
        candidates = _find_candidates(current_points, idx)
        if candidates:
            new_point = random.choice(candidates)
        else:
            # Random move
            if random.random() < 0.5:
                # Small perturbation
                new_point = [old_point[0] + random.randint(-5, 5), 
                             old_point[1] + random.randint(-5, 5)]
            else:
                # Large jump
                new_point = [random.randint(-1000, 1000), 
                             random.randint(-1000, 1000)]
                
        if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
            continue
            
        # Check collision
        if any(new_point == p for i, p in enumerate(current_points) if i != idx):
            continue
            
        current_points[idx] = new_point
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
            # Accept if better or equal (to allow wandering)
            if new_score >= current_score or random.random() < 0.1:
                current_score = new_score
            else:
                current_points[idx] = old_point
        else:
            current_points[idx] = old_point

    return best_points
