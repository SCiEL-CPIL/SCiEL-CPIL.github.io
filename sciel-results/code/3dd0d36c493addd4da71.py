import time
import random
import math
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (or 150s if specified, but 115 is safe)

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def score_points(points):
        """Count lines with exactly 4 points, penalize 5+."""
        line_points = defaultdict(set)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_points[key].add(i)
                line_points[key].add(j)
        
        num_4 = 0
        num_5_plus = 0
        for pts in line_points.values():
            k = len(pts)
            if k == 4:
                num_4 += 1
            elif k >= 5:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate(n4, n5):
        # Heavily penalize 5+ collinear
        return n4 - 10000 * n5

    def initial_grid():
        return [[i, j] for i in range(4) for j in range(4)]

    def random_config(box_size=50):
        pts = []
        used = set()
        for _ in range(16):
            for _attempt in range(500):
                x = random.randint(-box_size, box_size)
                y = random.randint(-box_size, box_size)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
                    break
            else:
                # Fallback: place anywhere unique
                while True:
                    x = random.randint(-1000, 1000)
                    y = random.randint(-1000, 1000)
                    if (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
                        break
        return pts

    def sa_search(initial_pts, time_budget, seed):
        random.seed(seed)
        points = [p[:] for p in initial_pts]
        n4, n5 = score_points(points)
        current_score = evaluate(n4, n5)
        best_score = current_score
        best_points = [p[:] for p in points]
        
        T = 5.0
        min_T = 0.1
        cooling = 0.999
        
        end_time = time.time() + time_budget
        iteration = 0
        stagnation_count = 0
        max_stagnation = 500

        while time.time() < end_time:
            iteration += 1
            if iteration > max_stagnation and stagnation_count > max_stagnation:
                break

            move_type = random.random()
            
            if move_type < 0.5:
                # Local Jitter
                idx = random.randint(0, 15)
                ox, oy = points[idx]
                step = random.randint(1, 5)
                nx = ox + random.randint(-step, step)
                ny = oy + random.randint(-step, step)
                
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                    continue
                    
                old_pos = points[idx][:]
                points[idx] = [nx, ny]
                new_n4, new_n5 = score_points(points)
                new_score = evaluate(new_n4, new_n5)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        current_score = new_score
                        stagnation_count += 1
                    else:
                        points[idx] = old_pos
                        
            elif move_type < 0.8:
                # Line Completion Move
                # Build line structure
                line_points = defaultdict(set)
                for i in range(16):
                    for j in range(i + 1, 16):
                        key = canonical_line(points[i], points[j])
                        line_points[key].add(i)
                        line_points[key].add(j)
                
                # Find lines with exactly 3 points
                three_pt_lines = []
                for key, pts in line_points.items():
                    if len(pts) == 3:
                        three_pt_lines.append((key, pts))
                
                if three_pt_lines:
                    key, existing_pts = random.choice(three_pt_lines)
                    a, b, c = key
                    
                    # Find candidate points NOT on this line
                    candidates = [k for k in range(16) if k not in existing_pts]
                    if candidates:
                        idx = random.choice(candidates)
                        
                        # Find a valid integer position on the line (a*x + b*y = c)
                        # We want to move points[idx] to (nx, ny) on this line
                        found = False
                        # Sample a few x values to find integer y
                        # Range of x: -1000 to 1000
                        # To be efficient, pick random x in a local range or global?
                        # Global random x is better to explore space
                        for _ in range(50):
                            nx = random.randint(-1000, 1000)
                            if b == 0:
                                # Vertical line x = c/a
                                if a != 0 and c % a == 0:
                                    nx = c // a
                                    if abs(nx) > 1000: continue
                                    ny = random.randint(-1000, 1000)
                                else:
                                    continue
                            else:
                                val = c - a * nx
                                if val % b != 0:
                                    continue
                                ny = val // b
                            
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            
                            # Check collision
                            if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                                continue
                            
                            old_pos = points[idx][:]
                            points[idx] = [nx, ny]
                            new_n4, new_n5 = score_points(points)
                            new_score = evaluate(new_n4, new_n5)
                            
                            if new_score > current_score:
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_points = [p[:] for p in points]
                                    stagnation_count = 0
                                else:
                                    stagnation_count += 1
                            else:
                                delta = new_score - current_score
                                if random.random() < math.exp(delta / T):
                                    current_score = new_score
                                    stagnation_count += 1
                                else:
                                    points[idx] = old_pos
                            found = True
                            break

            else:
                # Global Random Move
                idx = random.randint(0, 15)
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                    continue
                old_pos = points[idx][:]
                points[idx] = [nx, ny]
                new_n4, new_n5 = score_points(points)
                new_score = evaluate(new_n4, new_n5)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        current_score = new_score
                        stagnation_count += 1
                    else:
                        points[idx] = old_pos
            
            T = max(min_T, T * cooling)
            
        return best_points, best_score

    # Multi-start SA
    best_overall_score = -float('inf')
    best_overall_points = None
    
    # Initial configurations
    starts = []
    
    # 1. Standard 4x4 Grid
    grid_pts = initial_grid()
    starts.append(grid_pts)
    
    # 2. 4x4 Grid with slight noise
    noisy_grid = [p[:] for p in grid_pts]
    for i in range(16):
        if random.random() < 0.2:
            noisy_grid[i][0] += random.randint(-2, 2)
            noisy_grid[i][1] += random.randint(-2, 2)
    starts.append(noisy_grid)
    
    # 3. Random configurations
    starts.append(random_config(30))
    starts.append(random_config(50))
    
    # 4. Structured: Two 4x4 grids? No, 16 points.
    # Try a "Star" or "Cross" pattern?
    # Just use random for now.
    
    remaining_time = deadline - time.time()
    if remaining_time < 10:
        # Not enough time for SA, just evaluate starts
        for pts in starts:
            n4, n5 = score_points(pts)
            s = evaluate(n4, n5)
            if s > best_overall_score:
                best_overall_score = s
                best_overall_points = pts
        return best_overall_points
    
    time_per_start = remaining_time / len(starts)
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 5:
            break
        # Use different seeds
        seed = 42 + i
        bp, bs = sa_search(start_pts, time_per_start, seed)
        if bs > best_overall_score:
            best_overall_score = bs
            best_overall_points = bp
    
    # Final validation
    if best_overall_points is None:
        best_overall_points = initial_grid()
        
    fn4, fn5 = score_points(best_overall_points)
    if fn5 > 0:
        # If we ended up with a 5-line, try to fix it or fallback
        # Simple fix: move one point from the 5-line
        line_points = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                key = canonical_line(best_overall_points[i], best_overall_points[j])
                line_points[key].add(i)
                line_points[key].add(j)
        
        for key, pts in line_points.items():
            if len(pts) >= 5:
                # Move one point
                idx_to_move = list(pts)[0]
                # Move it far away
                best_overall_points[idx_to_move] = [random.randint(-100, 100), random.randint(-100, 100)]
                break
        
        # Re-evaluate
        fn4, fn5 = score_points(best_overall_points)
        if fn5 > 0:
            # Fallback to grid if still bad
            best_overall_points = initial_grid()
            
    return best_overall_points

if __name__ == "__main__":
    import time
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")
    print(pts)
    print(len(pts))
    
    # Verify
    line_points = defaultdict(set)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            key = canonical_line(pts[i], pts[j])
            line_points[key].add(i)
            line_points[key].add(j)
    
    count_4 = 0
    count_5 = 0
    max_collinear = 0
    for k, v in line_points.items():
        if len(v) == 4:
            count_4 += 1
        elif len(v) >= 5:
            count_5 += 1
        max_collinear = max(max_collinear, len(v))
    
    print("Max collinear:", max_collinear)
    print("4-lines:", count_4)
    print("5+-lines:", count_5)
