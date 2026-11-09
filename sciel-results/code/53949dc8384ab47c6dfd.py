import time
import random
from itertools import combinations

def normalize_vector(v):
    x, y = v
    if x == 0 and y == 0:
        return (0, 0)
    g = abs(x) if x != 0 else abs(y)
    # Use gcd to normalize
    from math import gcd
    d = gcd(abs(x), abs(y))
    if d > 0:
        x //= d
        y //= d
    # Ensure consistent sign, e.g., x > 0 or x == 0 and y > 0
    if x < 0 or (x == 0 and y < 0):
        x = -x
        y = -y
    return (x, y)

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    # Normal direction
    # Line equation: ax + by + c = 0
    # Normal is (dy, -dx)
    # Normalize (dy, -dx)
    a, b = dy, -dx
    if a == 0 and b == 0:
        return None
    g = abs(a) if a != 0 else abs(b)
    from math import gcd
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    # Normalize c? 
    # The key (a, b, c) uniquely identifies the line.
    # However, (a, b, c) and (-a, -b, -c) are the same line.
    # We normalized (a, b) so the sign is fixed.
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
        return -1 # Invalid
    return four_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Seed configuration: 4x4 grid
    seed_points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in seed_points]
    best_score = count_4point_lines(best_points)
    
    # Local Search / Hill Climbing
    # We will try to improve the seed and also explore random restarts
    
    def random_perturbation(points):
        new_points = [p[:] for p in points]
        idx = random.randint(0, len(new_points) - 1)
        # Move to a random nearby integer coordinate
        # Range -10 to 10 relative to current
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        new_x = new_points[idx][0] + dx
        new_y = new_points[idx][1] + dy
        # Keep within bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None
        new_points[idx] = [new_x, new_y]
        # Check for duplicates
        for i in range(len(new_points)):
            if i == idx: continue
            if new_points[i] == new_points[idx]:
                return None
        return new_points

    # Local search from seed
    current_points = [p[:] for p in seed_points]
    current_score = best_score
    
    iterations = 0
    while time.time() < deadline:
        # If we have time, do a random restart occasionally
        if iterations % 1000 == 0 and iterations > 0:
            # Generate a random set of 16 points in a small box
            # This helps escape local optima
            if random.random() < 0.1:
                random_points = []
                for _ in range(16):
                    x = random.randint(-5, 5)
                    y = random.randint(-5, 5)
                    if [x, y] not in random_points:
                        random_points.append([x, y])
                # Fill if we have duplicates
                while len(random_points) < 16:
                    x = random.randint(-10, 10)
                    y = random.randint(-10, 10)
                    if [x, y] not in random_points:
                        random_points.append([x, y])
                current_points = random_points
                current_score = count_4point_lines(current_points)
                if current_score < 0:
                    current_score = 0 # Invalid, treat as 0 for search logic, but we need valid points
                    # Actually, if invalid, we should probably revert or penalize
                    # For now, if invalid, we just try to move points to make it valid
                    # Or just skip this restart and go back to last valid
                    # Let's just re-initialize to seed if invalid
                    current_points = [p[:] for p in seed_points]
                    current_score = best_score
        
        new_points = random_perturbation(current_points)
        if new_points is None:
            continue
            
        new_score = count_4point_lines(new_points)
        
        if new_score >= 0:
            # Valid
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            elif new_score == current_score:
                # Accept with some probability to allow exploration
                if random.random() < 0.1:
                    current_points = new_points
        else:
            # Invalid, reject
            pass
            
        iterations += 1
        if iterations % 10000 == 0:
            # Check time
            if time.time() > deadline:
                break

    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Seed configuration: 4x4 grid
    seed_points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in seed_points]
    best_score = count_4point_lines(best_points)
    
    # Local Search / Hill Climbing
    def random_perturbation(points):
        new_points = [p[:] for p in points]
        idx = random.randint(0, len(new_points) - 1)
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        new_x = new_points[idx][0] + dx
        new_y = new_points[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None
        new_points[idx] = [new_x, new_y]
        for i in range(len(new_points)):
            if i == idx: continue
            if new_points[i] == new_points[idx]:
                return None
        return new_points

    current_points = [p[:] for p in seed_points]
    current_score = best_score
    
    iterations = 0
    while time.time() < deadline:
        if iterations % 1000 == 0 and iterations > 0:
            if random.random() < 0.1:
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
                current_points = random_points
                current_score = count_4point_lines(current_points)
                if current_score < 0:
                    current_points = [p[:] for p in seed_points]
                    current_score = best_score
        
        new_points = random_perturbation(current_points)
        if new_points is None:
            continue
            
        new_score = count_4point_lines(new_points)
        
        if new_score >= 0:
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            elif new_score == current_score:
                if random.random() < 0.1:
                    current_points = new_points
        else:
            pass
            
        iterations += 1
        if iterations % 10000 == 0:
            if time.time() > deadline:
                break

    return best_points
# EVOLVE-BLOCK-END
