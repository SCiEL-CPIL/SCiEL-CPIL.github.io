import time
import random
from math import gcd
from collections import defaultdict

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normal vector (a, b) such that a*x + b*y + c = 0
    a = dy
    b = -dx
    # Normalize (a, b)
    g = gcd(abs(a), abs(b))
    if g > 0:
        a //= g
        b //= g
    # Consistent sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def count_4point_lines(points):
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is not None:
                line_counts[key] += 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1000
    return four_count

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Seed configuration: 4x4 grid
    seed_points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in seed_points]
    best_score = count_4point_lines(best_points)
    
    # Initialize line counts for incremental updates
    def build_line_counts(points):
        counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    counts[key] += 1
        return counts

    def calculate_score_from_counts(counts):
        four_count = 0
        valid = True
        for count in counts.values():
            if count >= 5:
                valid = False
                break
            if count == 4:
                four_count += 1
        if not valid:
            return -1000
        return four_count

    def simulate_annealing(initial_points, time_limit):
        points = [p[:] for p in initial_points]
        counts = build_line_counts(points)
        current_score = calculate_score_from_counts(counts)
        
        best_local_points = [p[:] for p in points]
        best_local_score = current_score
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < time_limit and T > min_T:
            # Pick a random point to move
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            # Generate new position
            # Use a random step size that shrinks? Or fixed?
            # Fixed small steps are good for local optima.
            # We can also do larger jumps occasionally.
            if random.random() < 0.1:
                step = random.randint(1, 20)
            else:
                step = random.randint(1, 5)
            
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Check duplicates
            new_point = [new_x, new_y]
            duplicate = False
            for i in range(16):
                if i != idx and points[i] == new_point:
                    duplicate = True
                    break
            if duplicate:
                continue
            
            # Update counts incrementally
            # Remove lines involving old point
            old_lines = []
            for j in range(16):
                if j == idx: continue
                key = get_line_key(points[idx], points[j])
                if key is not None:
                    counts[key] -= 1
                    old_lines.append((j, key))
            
            # Add lines involving new point
            new_lines = []
            for j in range(16):
                if j == idx: continue
                key = get_line_key(new_point, points[j])
                if key is not None:
                    counts[key] += 1
                    new_lines.append((j, key))
            
            # Calculate new score
            new_score = calculate_score_from_counts(counts)
            
            # Acceptance criterion
            if new_score > current_score or (new_score == current_score and random.random() < 0.5):
                # Accept
                points[idx] = new_point
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_points = [p[:] for p in points]
            else:
                # Reject: Revert counts
                for j, key in old_lines:
                    counts[key] += 1
                for j, key in new_lines:
                    counts[key] -= 1
            
            T *= cooling
        
        return best_local_points, best_local_score

    # Run multiple restarts with different initializations
    # 1. 4x4 Grid
    # 2. Random points in small box
    # 3. Random points in medium box
    
    initializations = [
        seed_points,
        [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)],
        [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(16)],
        [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(16)]
    ]
    
    # Ensure distinct points for random initializations
    for i in range(1, len(initializations)):
        pts = initializations[i]
        seen = set()
        unique_pts = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_pts.append(p)
        while len(unique_pts) < 16:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        initializations[i] = unique_pts

    for init in initializations:
        if time.time() > deadline - 5:
            break
            
        # Allocate remaining time to this run, but cap at 30s per run to allow multiple
        time_limit = min(deadline - 5, time.time() + 30)
        
        try:
            res_points, res_score = simulate_annealing(init, time_limit)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points
        except Exception as e:
            pass
            
    return best_points
