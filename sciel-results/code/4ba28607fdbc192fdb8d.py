import time
import random
import sys

# Set random seed for reproducibility during development, but evaluator might use different
# random.seed(42) 

_DEADLINE = 115.0  # Leave 5s margin for 120s limit
_start_time = 0.0

def _elapsed():
    return time.time() - _start_time

def _check_no_four_collinear(points):
    """
    Check if no four points are collinear.
    Returns True if valid (no 4 collinear), False otherwise.
    Also returns the list of indices involved in any 4-collinear line if invalid.
    """
    n = len(points)
    # For each pair, count how many other points are on the line
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            # Check all other points
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _count_triples(points):
    """
    Count the number of collinear triples.
    This is used as a proxy for the objective.
    """
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            # If there are 'count' points on this line, the number of triples on this line is C(count, 3)
            # But we are counting unique triples of points.
            # Each triple is defined by 3 points. 
            # If we iterate over pairs (i,j), we count the third point k.
            # This counts each triple 3 times (once for each pair in the triple).
            # So we can just count the number of (i,j,k) such that i<j<k and collinear.
            # Or simply: for each line with m points, add C(m,3).
            # But we must avoid double counting lines.
            # Since we iterate over pairs, and determine the line, we might count the same line multiple times.
            # Better: Use a set of frozensets of indices for triples.
            
    # Efficient way:
    # For each pair (i,j), find all k.
    # To avoid overcounting, we can just count the number of valid (i,j,k) with i<j<k.
    # Let's just use the set method for correctness, it's N=20, so it's fast enough.
    triple_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            for k in range(j + 1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    triple_set.add((i, j, k))
    return len(triple_set)

def _repair_no_four_collinear(points, max_iters=1000):
    """
    If points have 4 collinear, move one of them to a random valid location.
    """
    n = len(points)
    for _ in range(max_iters):
        if _elapsed() > _DEADLINE - 5:
            break
        if _check_no_four_collinear(points):
            return True
        
        # Find a 4-collinear line
        found_line = None
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                line_pts = [i, j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        line_pts.append(k)
                if len(line_pts) >= 4:
                    found_line = line_pts
                    break
            if found_line:
                break
        
        if not found_line:
            break
            
        # Move one of the points on the line to a random new position
        idx = random.choice(found_line)
        old_x, old_y = points[idx]
        
        # Try to find a new position
        for _ in range(50):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            # Check duplicate
            is_dup = False
            for k in range(n):
                if k == idx:
                    continue
                if points[k][0] == new_x and points[k][1] == new_y:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Tentatively move
            points[idx] = [new_x, new_y]
            
            # Check if this move creates a NEW 4-collinear or fixes the old one without creating others
            # Actually, just check if the whole config is valid now.
            # It's expensive but N=20 is small.
            if _check_no_four_collinear(points):
                return True
            # If not valid, revert and try again
            points[idx] = [old_x, old_y]
            
        # If we couldn't fix it in 50 tries, revert and break
        points[idx] = [old_x, old_y]
        break
        
    return _check_no_four_collinear(points)

def _local_search_maximize_triples(points, time_limit=5.0):
    """
    Local search to maximize the number of collinear triples.
    """
    start = time.time()
    current_points = [list(p) for p in points]
    current_triples = _count_triples(current_points)
    
    n = len(current_points)
    
    while time.time() - start < time_limit:
        if _elapsed() > _DEADLINE - 5:
            break
            
        improved = False
        # Try moving each point
        for i in range(n):
            if time.time() - start > time_limit:
                break
            old_x, old_y = current_points[i]
            best_new_pos = None
            best_new_triples = current_triples
            
            # Try small perturbations and some random jumps
            # Small perturbations
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                        continue
                    
                    # Check duplicate
                    is_dup = False
                    for k in range(n):
                        if k == i:
                            continue
                        if current_points[k][0] == new_x and current_points[k][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                        
                    # Tentative move
                    current_points[i] = [new_x, new_y]
                    
                    # Check validity (no 4 collinear)
                    if not _check_no_four_collinear(current_points):
                        current_points[i] = [old_x, old_y]
                        continue
                        
                    new_triples = _count_triples(current_points)
                    
                    if new_triples > best_new_triples:
                        best_new_triples = new_triples
                        best_new_pos = [new_x, new_y]
                        
            # Random jump
            for _ in range(10):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
                is_dup = False
                for k in range(n):
                    if k == i:
                        continue
                    if current_points[k][0] == new_x and current_points[k][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                    
                current_points[i] = [new_x, new_y]
                
                if not _check_no_four_collinear(current_points):
                    current_points[i] = [old_x, old_y]
                    continue
                    
                new_triples = _count_triples(current_points)
                
                if new_triples > best_new_triples:
                    best_new_triples = new_triples
                    best_new_pos = [new_x, new_y]
                    
            if best_new_pos is not None:
                current_points[i] = best_new_pos
                current_triples = best_new_triples
                improved = True
                break # Restart loop for next point or next iteration
                
        if not improved:
            # Stuck in local optimum. Try a random restart or perturbation?
            # For now, just break and let the outer loop try another candidate.
            break
            
    return current_points

def _generate_candidate_parabola(shift1, shift2):
    """
    Generate points on two parabolas.
    """
    points = []
    seen = set()
    
    # y = x^2
    for x in range(-4, 5):
        y = x * x
        if (x, y) not in seen:
            points.append([x, y])
            seen.add((x, y))
            
    # y = x^2 + shift1
    for x in range(-4, 5):
        y = x * x + shift1
        if (x, y) not in seen and len(points) < 20:
            points.append([x, y])
            seen.add((x, y))
            
    # y = x^2 + shift2
    for x in range(-4, 5):
        y = x * x + shift2
        if (x, y) not in seen and len(points) < 20:
            points.append([x, y])
            seen.add((x, y))
            
    # Fill remaining
    if len(points) < 20:
        # Add points on y = 2x^2 or other curves?
        # Or just random nearby
        for _ in range(20 - len(points)):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in seen:
                points.append([x, y])
                seen.add((x, y))
                
    return points[:20]

def _generate_candidate_grid():
    """
    Generate points from a small grid, perturbed.
    """
    points = []
    seen = set()
    # 5x4 grid
    for x in range(-2, 3):
        for y in range(-2, 2):
            if (x, y) not in seen:
                points.append([x, y])
                seen.add((x, y))
                
    # Perturb to break 4-collinear
    # We will rely on _repair_no_four_collinear
    return points[:20]

def _generate_candidate_random():
    """
    Generate 20 random points.
    """
    points = []
    seen = set()
    while len(points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            points.append([x, y])
            seen.add((x, y))
    return points

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points = None
    best_triple_count = -1
    
    candidates = []
    
    # 1. Parabola candidates
    shifts = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 5), (3, 4), (1, 5), (2, 6)]
    for s1, s2 in shifts:
        if _elapsed() > _DEADLINE - 10:
            break
        pts = _generate_candidate_parabola(s1, s2)
        if _repair_no_four_collinear(pts):
            candidates.append(pts)
            
    # 2. Grid candidates
    for _ in range(5):
        if _elapsed() > _DEADLINE - 10:
            break
        pts = _generate_candidate_grid()
        if _repair_no_four_collinear(pts):
            candidates.append(pts)
            
    # 3. Random candidates
    for _ in range(10):
        if _elapsed() > _DEADLINE - 10:
            break
        pts = _generate_candidate_random()
        if _repair_no_four_collinear(pts):
            candidates.append(pts)
            
    if not candidates:
        # Fallback: hardcoded valid config
        # This is a known valid config (no 4 collinear)
        best_points = [
            [0, 0], [1, 1], [2, 4], [3, 9], [-1, 1], [-2, 4], [-3, 9],
            [4, 16], [-4, 16], [1, 2], [2, 5], [3, 10], [-1, 2], [-2, 5],
            [-3, 10], [4, 17], [-4, 17], [0, 1], [5, 25], [-5, 25]
        ]
        # Verify it's valid, if not, try to repair
        if not _check_no_four_collinear(best_points):
             _repair_no_four_collinear(best_points)
        return best_points

    # 4. Local Search on candidates
    # We have limited time. Allocate time per candidate.
    # If we have 20 candidates and 100s left, 5s each.
    num_candidates = len(candidates)
    time_left = _DEADLINE - _elapsed() - 5
    time_per_candidate = max(0.5, time_left / num_candidates)
    
    for pts in candidates:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Run local search
        improved_pts = _local_search_maximize_triples(pts, time_limit=time_per_candidate)
        
        if not _check_no_four_collinear(improved_pts):
            # If local search broke validity (shouldn't happen, but just in case)
            continue
            
        triples = _count_triples(improved_pts)
        
        if triples > best_triple_count:
            best_triple_count = triples
            best_points = improved_pts
            
    if best_points is None:
        # If no candidate survived, return the first valid one we had
        best_points = candidates[0]
        
    return best_points

# Ensure the function is named construct
# The evaluator calls construct()

if __name__ == "__main__":
    # Quick test
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Points: {pts}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
    print(f"Triples: {_count_triples(pts)}")
