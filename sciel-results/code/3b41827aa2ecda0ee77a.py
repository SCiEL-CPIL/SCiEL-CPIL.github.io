import time
import random
from itertools import combinations
from math import gcd

def normalize_vector(v):
    x, y = v
    if x == 0 and y == 0:
        return (0, 0)
    d = gcd(abs(x), abs(y))
    if d > 0:
        x //= d
        y //= d
    if x < 0 or (x == 0 and y < 0):
        x = -x
        y = -y
    return (x, y)

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a, b = dy, -dx
    if a == 0 and b == 0:
        return None
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
    """
    Counts lines with exactly 4 points.
    Returns -1 if any line has >= 5 points.
    """
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # The 4x4 grid is a strong seed.
    # Points: (0,0) to (3,3)
    seed_points = [[x, y] for x in range(4) for y in range(4)]
    
    # We will perform a more structured local search.
    # Strategy:
    # 1. Start with 4x4 grid.
    # 2. Identify all lines with exactly 3 points.
    # 3. For each such line, try to move a point NOT on the line to a position ON the line 
    #    such that:
    #    a) The point is distinct from existing points.
    #    b) The new position doesn't create a 5-collinear set.
    #    c) The move increases the number of 4-point lines (or keeps it same but improves structure).
    
    best_points = [p[:] for p in seed_points]
    best_score = count_4point_lines(best_points)
    
    current_points = [p[:] for p in seed_points]
    current_score = best_score
    
    def get_lines_with_3_points(points):
        """Returns list of (line_key, [indices_of_points]) for lines with exactly 3 points."""
        n = len(points)
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_points:
                    line_points[key] = []
                line_points[key].extend([i, j])
        
        # Deduplicate indices in line_points
        cleaned_line_points = {}
        for key, idxs in line_points.items():
            unique_idxs = list(set(idxs))
            if len(unique_idxs) == 3:
                cleaned_line_points[key] = unique_idxs
        
        return cleaned_line_points

    def is_valid_configuration(points):
        """Check if all points are distinct, within bounds, and no 5 collinear."""
        if len(set(tuple(p) for p in points)) != len(points):
            return False
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return False
        # Check 5-collinear
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                line_counts[key] = line_counts.get(key, 0) + 1
        for count in line_counts.values():
            if count >= 5:
                return False
        return True

    def try_improve_line(points, line_key, point_indices, score):
        """
        Try to move a point not on the line to a position on the line to make it a 4-point line.
        Returns (new_points, new_score) if improvement found, else (None, score).
        """
        n = len(points)
        on_line = set(point_indices)
        off_line = [i for i in range(n) if i not in on_line]
        
        # Get the direction of the line
        p1 = points[point_indices[0]]
        p2 = points[point_indices[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Normalize direction
        d = gcd(abs(dx), abs(dy))
        if d == 0: return None, score
        dx //= d
        dy //= d
        
        # The line is defined by p1 + t*(dx, dy)
        # We want to find an integer t such that the point (p1[0] + t*dx, p1[1] + t*dy)
        # is not already in the set of points, and moving a point from off_line to this position
        # is valid and improves the score.
        
        # Limit t to a reasonable range around the existing points
        min_t = min([(p[0] - p1[0]) // dx if dx != 0 else 0 for p in points if p[0] == p1[0] + t*dx for t in range(-5, 5)]) # This is messy, let's just iterate t
        # Simpler: iterate t from -10 to 10 relative to the "center" of the 3 points
        
        # Actually, let's just check positions near the 3 existing points.
        # A 4th point on the line must be an integer coordinate.
        # Let's collect all integer points on the line segment extended by 2 units in each direction.
        candidate_positions = set()
        for t in range(-3, 4):
            cx = p1[0] + t * dx
            cy = p1[1] + t * dy
            if abs(cx) <= 1000 and abs(cy) <= 1000:
                candidate_positions.add((cx, cy))
        
        # Remove existing points on the line
        for idx in point_indices:
            candidate_positions.discard(tuple(points[idx]))
        
        if not candidate_positions:
            return None, score
            
        # For each candidate position, try to move each off-line point to it
        for pos in candidate_positions:
            for idx in off_line:
                old_pos = points[idx]
                new_points = [p[:] for p in points]
                new_points[idx] = [pos[0], pos[1]]
                
                # Check validity
                if not is_valid_configuration(new_points):
                    continue
                
                new_score = count_4point_lines(new_points)
                if new_score > score:
                    return new_points, new_score
                # We could also accept equal score if it breaks a potential 5-collinear risk, 
                # but for now, strict improvement is safer to avoid loops.
                
        return None, score

    iterations = 0
    stagnation_counter = 0
    
    while time.time() < deadline:
        # Get lines with exactly 3 points
        lines_3 = get_lines_with_3_points(current_points)
        
        found_improvement = False
        for line_key, idxs in lines_3.items():
            new_points, new_score = try_improve_line(current_points, line_key, idxs, current_score)
            if new_points is not None:
                current_points = new_points
                current_score = new_score
                found_improvement = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                break # Restart the loop with the new configuration
        
        if not found_improvement:
            # No improvement found by line completion.
            # Try a random perturbation to escape local optima.
            # Move a random point to a random nearby position.
            idx = random.randint(0, 15)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = current_points[idx][0] + dx
            new_y = current_points[idx][1] + dy
            
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                new_points = [p[:] for p in current_points]
                new_points[idx] = [new_x, new_y]
                
                if is_valid_configuration(new_points):
                    new_score = count_4point_lines(new_points)
                    if new_score >= current_score:
                        current_points = new_points
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1
                
            # If we've been stuck for a while, do a larger random restart or re-seed
            if stagnation_counter > 50:
                # Random restart in a small box
                random_points = []
                for _ in range(16):
                    x = random.randint(-5, 5)
                    y = random.randint(-5, 5)
                    if [x, y] not in random_points:
                        random_points.append([x, y])
                while len(random_points) < 16:
                    x = random.randint(-10, 10)
                    y = random.randint(-10, 10)
                    if [x, y] not in random_points:
                        random_points.append([x, y])
                
                if is_valid_configuration(random_points):
                    current_points = random_points
                    current_score = count_4point_lines(current_points)
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                stagnation_counter = 0

        iterations += 1
        if iterations % 100 == 0 and time.time() > deadline - 5:
            break

    return best_points
