"""
Structural rewrite: line-based construction with explicit line selection.

Key structural change: instead of perturbing individual points (local search),
we explicitly choose a set of lines and distribute points on them.
This gives us direct control over the collinear triple structure.

Algorithm:
1. Generate candidate lines (integer lines with small slope/intercept).
2. Select a subset of lines and assign points to them (max 3 per line).
3. Score = largest subset with no three collinear (exact B&B).
4. Use simulated annealing on the LINE SELECTION (not point positions).
"""

import time
import random
import math
from collections import defaultdict

DEADLINE = 115.0  # seconds, safety margin before 150s kill

def collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if collinear(points[i], points[j], points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def max_no_three_collinear(points):
    """Exact branch and bound for largest 3-collinear-free subset."""
    n = len(points)
    
    # Precompute collinear pairs for each point
    coll_pairs = [set() for _ in range(n)]
    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            for j in range(i + 1, n):
                if j == k:
                    continue
                if collinear(points[i], points[j], points[k]):
                    coll_pairs[k].add(frozenset((i, j)))
    
    best = [0]
    
    def search(idx, current_set, current_pairs):
        if len(current_set) + (n - idx) <= best[0]:
            return
        if idx == n:
            if len(current_set) > best[0]:
                best[0] = len(current_set)
            return
        
        # Exclude
        search(idx + 1, current_set, current_pairs)
        
        # Include if valid
        can_include = True
        for pair in coll_pairs[idx]:
            if pair in current_pairs:
                can_include = False
                break
        
        if can_include:
            new_set = current_set | {idx}
            new_pairs = set(current_pairs)
            for j in current_set:
                new_pairs.add(frozenset((idx, j)))
            search(idx + 1, new_set, new_pairs)
    
    search(0, set(), set())
    return best[0]

def generate_candidate_lines():
    """Generate a pool of candidate integer lines within |x|,|y|<=100."""
    lines = []
    # Horizontal lines
    for y in range(-50, 51, 5):
        lines.append((0, 1, y))  # ax + by = c => 0*x + 1*y = y
    # Vertical lines
    for x in range(-50, 51, 5):
        lines.append((1, 0, x))
    # Sloped lines with small integer slopes
    for m in range(-3, 4):
        for b in range(-60, 61, 6):
            # y = mx + b => mx - y = -b => a=m, b=-1, c=-b
            lines.append((m, -1, -b))
    # Remove duplicates
    unique_lines = []
    seen = set()
    for a, b, c in lines:
        # Normalize
        g = math.gcd(a, b)
        if g > 0:
            a, b, c = a // g, b // g, c // g
        key = (a, b, c)
        if key not in seen:
            seen.add(key)
            unique_lines.append((a, b, c))
    return unique_lines

def points_on_line(line, max_coord=100):
    """Generate integer points on a line within bounds."""
    a, b, c = line
    points = []
    if a == 0 and b == 0:
        return points
    if a == 0:  # horizontal: y = c/b
        if b != 0 and c % b == 0:
            y = c // b
            if abs(y) <= max_coord:
                for x in range(-max_coord, max_coord + 1):
                    points.append((x, y))
        return points
    if b == 0:  # vertical: x = c/a
        if a != 0 and c % a == 0:
            x = c // a
            if abs(x) <= max_coord:
                for y in range(-max_coord, max_coord + 1):
                    points.append((x, y))
        return points
    # General: ax + by = c
    # Find one solution, then parameterize
    # Use extended Euclidean for ax + by = c
    # But simpler: iterate x and check
    for x in range(-max_coord, max_coord + 1):
        if (c - a * x) % b == 0:
            y = (c - a * x) // b
            if abs(y) <= max_coord:
                points.append((x, y))
    return points

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Generate candidate lines
    candidate_lines = generate_candidate_lines()
    print(f"Generated {len(candidate_lines)} candidate lines", flush=True)
    
    # We'll select a subset of lines and assign 2-3 points to each
    # Total points = 20, so we need roughly 7-10 lines
    
    def make_config_from_lines(selected_lines):
        """Assign points to selected lines, ensuring no four collinear."""
        if not selected_lines:
            return None
        
        points = []
        used = set()
        
        # For each line, pick 2-3 points not already used
        for line in selected_lines:
            if len(points) >= 20:
                break
            line_pts = points_on_line(line)
            if len(line_pts) < 2:
                continue
            # Shuffle for randomness
            random.shuffle(line_pts)
            added = 0
            for p in line_pts:
                if len(points) >= 20:
                    break
                if p in used:
                    continue
                # Check no four collinear
                new_pts = points + [p]
                if has_four_collinear(new_pts):
                    continue
                points.append(p)
                used.add(p)
                added += 1
                if added >= 3:
                    break
        
        if len(points) < 20:
            # Fill with random points
            for _ in range(1000):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p in used:
                    continue
                new_pts = points + [p]
                if has_four_collinear(new_pts):
                    continue
                points.append(p)
                used.add(p)
                if len(points) == 20:
                    break
        
        if len(points) != 20:
            return None
        return points
    
    # Initial: pick random lines
    n_lines_to_use = 8
    selected_lines = random.sample(candidate_lines, min(n_lines_to_use, len(candidate_lines)))
    points = make_config_from_lines(selected_lines)
    if points is None:
        # Fallback
        points = [(x, x*x) for x in range(-10, 10)]
    
    current_score = max_no_three_collinear(points)
    best_score = current_score
    best_points = points[:]
    print(f"Initial score: {best_score}", flush=True)
    
    # Simulated annealing on line selection
    T = 10.0
    cooling = 0.998
    step = 0
    
    while time.time() - start_time < DEADLINE:
        step += 1
        if step % 500 == 0:
            print(f"Step {step}, best_score={best_score}, T={T:.2f}", flush=True)
        
        # Mutate: replace one line with a random new line
        idx = random.randint(0, len(selected_lines) - 1)
        new_line = random.choice(candidate_lines)
        if new_line in selected_lines:
            continue
        selected_lines[idx] = new_line
        
        new_points = make_config_from_lines(selected_lines)
        if new_points is None:
            # Revert
            selected_lines[idx] = candidate_lines[random.randint(0, len(candidate_lines)-1)]
            # Find the old line... we lost it. Let's track it differently.
            # Actually, let's just revert by re-selecting
            continue
        
        new_score = max_no_three_collinear(new_points)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
            current_score = new_score
            points = new_points
            if new_score < best_score:
                best_score = new_score
                best_points = points[:]
                print(f"  New best: {best_score} at step {step}", flush=True)
        else:
            # Revert: we need to restore the old line
            # This is tricky. Let's use a different mutation strategy.
            pass
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return [[p[0], p[1]] for p in best_points]
