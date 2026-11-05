import time
import random
from math import gcd

def _line_key(p1, p2):
    """Calculate the normalized line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    ndx = dx // g
    ndy = dy // g
    # Normalize direction
    if ndx < 0 or (ndx == 0 and ndy < 0):
        ndx = -ndx
        ndy = -ndy
    # c = ndy * x1 - ndx * y1
    c = ndy * x1 - ndx * y1
    return (ndx, ndy, c)

def _build_line_counts(points):
    """Build the line counts dictionary from scratch."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
    return counts

def _score_from_counts(counts):
    """Calculate the score and check validity from line counts."""
    score = 0
    for count in counts.values():
        if count >= 10:  # C(5,2) = 10, so 5 or more points
            return -1, 0  # Invalid
        if count == 6:   # C(4,2) = 6, so exactly 4 points
            score += 1
    return 1, score  # Valid, score

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # Initial configuration: 4x4 grid
    # This has 10 lines of 4 points and is valid.
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Initial line counts
    line_counts = _build_line_counts(points)
    valid, score = _score_from_counts(line_counts)
    if not valid:
        # Fallback, though 4x4 grid is always valid
        points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
        line_counts = _build_line_counts(points)
        valid, score = _score_from_counts(line_counts)
        
    best_score = score
    best_points = [p[:] for p in points]
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    
    # We will use a loop that runs until deadline
    # To make it efficient, we track the current state and update incrementally
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Generate a new position
        # Bias towards small moves to stay in the neighborhood, 
        # but occasionally allow larger jumps.
        if random.random() < 0.8:
            # Small move
            step = random.randint(1, 3)
            dx = random.choice([-step, step])
            dy = random.choice([-step, step])
            new_x = old_x + dx
            new_y = old_y + dy
        else:
            # Medium move
            step = random.randint(4, 10)
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
            new_x = old_x + dx
            new_y = old_y + dy
            
        if new_x == old_x and new_y == old_y:
            continue
            
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check if new point collides with any existing point
        # (Distinctness constraint)
        new_pt = [new_x, new_y]
        collides = False
        for i in range(16):
            if i == idx:
                continue
            if points[i][0] == new_x and points[i][1] == new_y:
                collides = True
                break
        if collides:
            continue
            
        # Update line counts incrementally
        # 1. Remove contributions of old point
        old_pt = [old_x, old_y]
        for i in range(16):
            if i == idx:
                continue
            key = _line_key(old_pt, points[i])
            if key is not None:
                line_counts[key] -= 1
                if line_counts[key] == 0:
                    del line_counts[key]
                    
        # 2. Add contributions of new point
        for i in range(16):
            if i == idx:
                continue
            key = _line_key(new_pt, points[i])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
                
        # 3. Evaluate new state
        valid, new_score = _score_from_counts(line_counts)
        
        if valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                # Accept the move
                points[idx] = new_pt
                T = max(0.1, T * cooling_rate)
            else:
                # Accept or reject based on simulated annealing
                delta = new_score - score
                if delta >= 0:
                    # Accept
                    points[idx] = new_pt
                    score = new_score
                    T = max(0.1, T * cooling_rate)
                else:
                    # Reject with probability 1 - exp(-delta/T)
                    prob = 1.0 - 0.3  # Simple heuristic: 70% chance to accept worse moves early on
                    # Better: use actual temperature
                    # prob = random.exp(-delta / T) if T > 0 else 0
                    # For simplicity and robustness, let's use a fixed probability for worse moves
                    # that decreases over time or stays constant.
                    # Let's stick to a simple 20% acceptance of worse moves to escape local optima.
                    if random.random() < 0.2:
                        points[idx] = new_pt
                        score = new_score
                    else:
                        # Revert
                        # Restore old point
                        points[idx] = old_pt
                        # Restore line counts
                        # Remove new point contributions
                        for i in range(16):
                            if i == idx:
                                continue
                            key = _line_key(new_pt, points[i])
                            if key is not None:
                                line_counts[key] -= 1
                                if line_counts[key] == 0:
                                    del line_counts[key]
                        # Add old point contributions
                        for i in range(16):
                            if i == idx:
                                continue
                            key = _line_key(old_pt, points[i])
                            if key is not None:
                                line_counts[key] = line_counts.get(key, 0) + 1
                        # Score remains unchanged
        else:
            # Invalid state (5+ collinear), revert
            points[idx] = old_pt
            # Restore line counts
            for i in range(16):
                if i == idx:
                    continue
                key = _line_key(new_pt, points[i])
                if key is not None:
                    line_counts[key] -= 1
                    if line_counts[key] == 0:
                        del line_counts[key]
            for i in range(16):
                if i == idx:
                    continue
                key = _line_key(old_pt, points[i])
                if key is not None:
                    line_counts[key] = line_counts.get(key, 0) + 1

    return best_points
