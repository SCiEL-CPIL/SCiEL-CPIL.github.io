"""Rewritten program: line-based simulated annealing for 20-point no-four-collinear."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 130.0

def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _has_four_collinear(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            ax, ay = points[i]
            bx, by = points[j]
            dx = bx - ax
            dy = by - ay
            g = abs(dx)
            if g == 0:
                g = abs(dy)
            else:
                from math import gcd
                g = gcd(g, abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            key = (ax, ay, dx, dy)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    for line_pts in lines.values():
        if len(line_pts) >= 4:
            return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    # conflict_mask[i] = bitmask of points that form a collinear triple with i
    # Actually: for each pair (i,j), the set of k such that i,j,k collinear
    # We need: no three in subset collinear
    # conflict[i] = set of j such that there exists k with i,j,k collinear
    # Better: for each point i, conflict_with[i] = set of points j where
    #         adding j to a subset containing i would create a triple
    # Simplest: for each pair (i,j), store the set of points collinear with both
    # Then a subset is valid iff for every pair (i,j) in subset, no third point in subset is collinear
    
    # Precompute: for each pair (i,j), the set of other points collinear with i and j
    collinear_with_pair = {}
    for i in range(n):
        for j in range(i+1, n):
            collinear_with_pair[(i,j)] = set()
            for k in range(n):
                if k != i and k != j:
                    if _cross(points[i], points[j], points[k]) == 0:
                        collinear_with_pair[(i,j)].add(k)
    
    # For branch and bound:
    # A subset S is valid iff for all pairs (i,j) in S, collinear_with_pair[(i,j)] ∩ S = ∅
    # Equivalently: for all i in S, for all j in S with j > i, collinear_with_pair[(i,j)] ∩ S = ∅
    
    # Simpler check for a candidate subset:
    def is_valid(subset):
        s = set(subset)
        for i in subset:
            for j in subset:
                if j <= i:
                    continue
                if collinear_with_pair.get((i,j), set()) & s:
                    return False
        return True
    
    # Branch and bound with good ordering
    best = [0]
    
    # Order points by "conflict degree" - points involved in more collinear triples first
    # This helps prune earlier
    degree = [0] * n
    for (i,j), coll_set in collinear_with_pair.items():
        for k in coll_set:
            degree[i] += 1
            degree[j] += 1
            degree[k] += 1
    
    order = sorted(range(n), key=lambda x: -degree[x])
    
    def bnb(subset, available_mask, depth):
        if available_mask == 0:
            if len(subset) > best[0]:
                best[0] = len(subset)
            return
        
        # Upper bound
        remaining_count = bin(available_mask).count('1')
        if len(subset) + remaining_count <= best[0]:
            return
        
        # Try to add points in order
        for idx in order:
            if not (available_mask & (1 << idx)):
                continue
            
            # Check if adding idx is valid with current subset
            valid = True
            for i in subset:
                if i > idx:
                    key = (idx, i)
                else:
                    key = (i, idx)
                if collinear_with_pair.get(key, set()) & set(subset):
                    valid = False
                    break
            if not valid:
                continue
            
            new_subset = subset + [idx]
            new_available = available_mask & ~(1 << idx)
            bnb(new_subset, new_available, depth+1)
            
            # Also try without idx
            bnb(subset, new_available, depth+1)
            return  # Only need to explore one branch per level in this formulation
    
    # Actually, standard B&B: at each level, either include or exclude the next point
    # Let me redo this properly
    
    best[0] = 0
    
    def bnb2(subset, start_idx):
        if len(subset) + (n - start_idx) <= best[0]:
            return
        
        if start_idx == n:
            if len(subset) > best[0]:
                best[0] = len(subset)
            return
        
        # Exclude order[start_idx]
        bnb2(subset, start_idx + 1)
        
        # Include order[start_idx]
        idx = order[start_idx]
        # Check validity
        valid = True
        for i in subset:
            if i > idx:
                key = (idx, i)
            else:
                key = (i, idx)
            coll_set = collinear_with_pair.get(key, set())
            for k in coll_set:
                if k in subset:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            bnb2(subset + [idx], start_idx + 1)
    
    bnb2([], 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    return _max_no_three_collinear(points)

def _find_initial():
    """Find a valid initial 20-point configuration."""
    for _ in range(10000):
        points = []
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p not in points:
                    break
        if not _has_four_collinear(points):
            return [list(p) for p in points]
    # Fallback: grid-like
    points = []
    for i in range(20):
        x = i * 5
        y = (i * 7) % 101 - 50
        points.append([x, y])
    return points

def _get_lines(points):
    """Get all lines with their point sets."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            ax, ay = points[i]
            bx, by = points[j]
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                continue
            from math import gcd
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            key = (ax, ay, dx, dy)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 15  # safety margin
    
    points = _find_initial()
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    if time.time() > deadline:
        return best_points
    
    # Simulated annealing
    current_score = best_score
    current_points = [p[:] for p in points]
    
    # Temperature schedule
    T = 5.0
    T_min = 0.01
    cooling = 0.97
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Move a point to a random position
            idx = random.randint(0, 19)
            old_p = current_points[idx][:]
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            new_p = [x, y]
            if new_p in current_points:
                continue
            current_points[idx] = new_p
            
        elif move_type < 0.8:
            # Move a point along an existing line to another point on that line
            lines = _get_lines(current_points)
            # Find lines with at least 2 points
            multi_lines = [lp for lp in lines.values() if len(lp) >= 2]
            if not multi_lines:
                continue
            line_pts = random.choice(multi_lines)
            idx = random.choice(list(line_pts))
            old_p = current_points[idx][:]
            # Move to a random position on the same line
            # Get the line equation from two points
            pts_on_line = list(line_pts)
            if len(pts_on_line) >= 2:
                p1 = current_points[pts_on_line[0]]
                p2 = current_points[pts_on_line[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Random t parameter
                t = random.randint(-50, 50)
                x = p1[0] + t * dx
                y = p1[1] + t * dy
                if -100 <= x <= 100 and -100 <= y <= 100:
                    new_p = [x, y]
                    if new_p not in current_points:
                        current_points[idx] = new_p
                    else:
                        continue
                else:
                    continue
            else:
                continue
                
        else:
            # Swap two points
            i, j = random.sample(range(20), 2)
            current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
        
        # Check validity
        if _has_four_collinear(current_points):
            # Revert
            if move_type < 0.5:
                current_points[idx] = old_p
            elif move_type < 0.8:
                current_points[idx] = old_p
            else:
                current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
            continue
        
        # Evaluate
        new_score = _score(current_points)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            if move_type < 0.5:
                current_points[idx] = old_p
            elif move_type < 0.8:
                current_points[idx] = old_p
            else:
                current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
        
        # Cool down
        T = max(T_min, T * cooling)
        
        if time.time() > deadline:
            break
    
    return best_points
# EVOLVE-BLOCK-END
