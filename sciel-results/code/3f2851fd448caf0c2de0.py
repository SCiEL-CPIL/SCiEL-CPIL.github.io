import time
import random
from itertools import combinations
from math import gcd

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def count_4point_lines(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def fast_count(points):
    """Fast count of 4-point lines using line key counting."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def evaluate_perturbation(points, idx, new_pos):
    """Evaluate the score change when moving point idx to new_pos."""
    old_pos = points[idx]
    if old_pos == new_pos:
        return None  # No change
    
    # Check if new_pos collides with another point
    for i in range(len(points)):
        if i != idx and points[i] == new_pos:
            return None  # Collision
    
    # Count lines affected
    # Lines involving the old point
    old_line_counts = {}
    new_line_counts = {}
    
    for i in range(len(points)):
        if i == idx:
            continue
        k_old = get_line_key(old_pos, points[i])
        if k_old is not None:
            old_line_counts[k_old] = old_line_counts.get(k_old, 0) + 1
        k_new = get_line_key(new_pos, points[i])
        if k_new is not None:
            new_line_counts[k_new] = new_line_counts.get(k_new, 0) + 1
    
    # Compute the delta in 4-point line count
    # For each line that had the old point, check if removing it changes count from 4 to 3 or 3 to 4
    # For each line that gains the new point, check if adding it changes count from 3 to 4 or 4 to 5
    
    delta = 0
    invalid = False
    
    # Lines lost (had old point)
    for line, cnt in old_line_counts.items():
        # The line had cnt other points + old point = cnt+1 total
        # If cnt+1 == 4, we lose a 4-point line
        if cnt + 1 == 4:
            delta -= 1
        # If cnt+1 == 5, we were invalid, but we check validity separately
        if cnt + 1 == 5:
            # This was already invalid, removing might fix it
            pass
    
    # Lines gained (have new point)
    for line, cnt in new_line_counts.items():
        # The line had cnt other points + new point = cnt+1 total
        # If cnt+1 == 4, we gain a 4-point line
        if cnt + 1 == 4:
            delta += 1
        # If cnt+1 == 5, we create a 5-point line (invalid)
        if cnt + 1 == 5:
            invalid = True
    
    # Also need to check lines that don't involve point idx but might have been affected
    # Actually, lines not involving idx are unchanged
    
    # But we also need to check if the old point was part of a 5-line
    # and the new point creates a 5-line
    
    # Check if any line not involving idx already has 5+ points
    # We can't do this efficiently without full recount
    # For now, approximate: if invalid flag is set, reject
    # Otherwise, return delta
    
    if invalid:
        return None  # Invalid
    
    return delta

def local_search(points, deadline, max_iter=None):
    """Local search to improve the configuration."""
    current_points = [p[:] for p in points]
    current_score = fast_count(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    if max_iter is None:
        max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try moving each point to a nearby location
        idx = random.randint(0, 15)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_x = current_points[idx][0] + dx
        new_y = current_points[idx][1] + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_pos = [new_x, new_y]
        
        # Check collision
        collision = False
        for i in range(16):
            if i != idx and current_points[i] == new_pos:
                collision = True
                break
        if collision:
            continue
        
        # Evaluate
        old_pos = current_points[idx][:]
        current_points[idx] = new_pos
        new_score = fast_count(current_points)
        current_points[idx] = old_pos
        
        if new_score < 0:
            continue
        
        if new_score > current_score:
            current_points[idx] = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        elif new_score == current_score and random.random() < 0.3:
            current_points[idx] = new_pos
    
    return best_points, best_score

def random_configuration(deadline, max_iter=100):
    """Generate a random valid configuration with some structure."""
    for attempt in range(max_iter):
        if time.time() > deadline:
            break
        
        points = []
        # Generate points in a small box to encourage collinearities
        box = random.randint(3, 8)
        while len(points) < 16:
            x = random.randint(-box, box)
            y = random.randint(-box, box)
            if [x, y] not in points:
                points.append([x, y])
        
        score = fast_count(points)
        if score > 0:
            return points, score
    
    return None, -1

def grid_configuration():
    """Generate a 4x4 grid configuration."""
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Start with 4x4 grid
    best_points, best_score = grid_configuration()
    
    # Local search from grid
    best_points, best_score = local_search(best_points, deadline, max_iter=10000)
    
    # Try random restarts
    for _ in range(50):
        if time.time() > deadline:
            break
        
        rand_points, rand_score = random_configuration(deadline, max_iter=50)
        if rand_points and rand_score > best_score:
            best_points = rand_points
            best_score = rand_score
            # Local search from this point
            best_points, best_score = local_search(best_points, deadline, max_iter=5000)
    
    # Final local search
    if time.time() < deadline:
        best_points, best_score = local_search(best_points, deadline, max_iter=20000)
    
    # Ensure validity
    final_score = fast_count(best_points)
    if final_score < 0:
        # Fallback to grid
        best_points, best_score = grid_configuration()
    
    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Start with 4x4 grid
    best_points, best_score = grid_configuration()
    
    # Local search from grid
    best_points, best_score = local_search(best_points, deadline, max_iter=10000)
    
    # Try random restarts
    for _ in range(50):
        if time.time() > deadline:
            break
        
        rand_points, rand_score = random_configuration(deadline, max_iter=50)
        if rand_points and rand_score > best_score:
            best_points = rand_points
            best_score = rand_score
            # Local search from this point
            best_points, best_score = local_search(best_points, deadline, max_iter=5000)
    
    # Final local search
    if time.time() < deadline:
        best_points, best_score = local_search(best_points, deadline, max_iter=20000)
    
    # Ensure validity
    final_score = fast_count(best_points)
    if final_score < 0:
        # Fallback to grid
        best_points, best_score = grid_configuration()
    
    return best_points
# EVOLVE-BLOCK-END
