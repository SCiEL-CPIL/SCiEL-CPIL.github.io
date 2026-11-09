import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145.0

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

    def build_line_map(points):
        line_map = defaultdict(set)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def score_from_line_map(line_map):
        num_4 = 0
        num_5_plus = 0
        for key, indices in line_map.items():
            k = len(indices)
            if k == 4:
                num_4 += 1
            elif k >= 5:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate(points):
        line_map = build_line_map(points)
        n4, n5 = score_from_line_map(line_map)
        return n4 - 1000 * n5

    def count_lines_through_point(points, p, exclude_idx=None):
        """Count how many 4-lines would be formed if p were added at this position."""
        # For each pair of existing points, check if p + those 2 points form a line with exactly 4 total
        n = len(points)
        count = 0
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                if exclude_idx is not None and i == exclude_idx:
                    continue
                if exclude_idx is not None and j == exclude_idx:
                    continue
                key = canonical_line(points[i], points[j])
                if key in seen_lines:
                    continue
                # Check if p is on this line
                a, b, c = key
                if a * p[0] + b * p[1] == c:
                    # Count all points on this line
                    cnt = 1  # p itself
                    for k in range(n):
                        if k == exclude_idx:
                            continue
                        if a * points[k][0] + b * points[k][1] == c:
                            cnt += 1
                    if cnt == 4:
                        count += 1
                        seen_lines.add(key)
        return count

    def find_best_new_point(points, target_count=16, max_search=2000):
        """Find a point that maximizes new 4-lines while avoiding 5+ collinear."""
        best_pos = None
        best_gain = -1
        best_total_4 = -1
        
        # Try to find points near existing clusters
        existing_xs = [p[0] for p in points]
        existing_ys = [p[1] for p in points]
        
        # Generate candidate positions
        candidates = set()
        
        # Points near existing points
        for p in points:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = p[0] + dx, p[1] + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        # Check distinctness
                        is_distinct = True
                        for q in points:
                            if q[0] == nx and q[1] == ny:
                                is_distinct = False
                                break
                        if is_distinct:
                            candidates.add((nx, ny))
        
        # Also try some random positions
        for _ in range(500):
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
            is_distinct = True
            for q in points:
                if q[0] == nx and q[1] == ny:
                    is_distinct = False
                    break
            if is_distinct:
                candidates.add((nx, ny))
        
        for (nx, ny) in candidates:
            new_point = [nx, ny]
            # Check for 5+ collinear
            line_map = build_line_map(points)
            # Add new point and check
            new_points = points + [new_point]
            new_line_map = build_line_map(new_points)
            n4, n5 = score_from_line_map(new_line_map)
            if n5 > 0:
                continue
            if n4 > best_total_4:
                best_total_4 = n4
                best_gain = n4
                best_pos = (nx, ny)
        
        return best_pos, best_gain

    # Start with 4x4 grid
    best_points = [[i, j] for i in range(4) for j in range(4)]
    best_score = evaluate(best_points)
    
    # Phase 1: Greedy line-based construction
    # Start from empty and build up
    def greedy_construct():
        points = []
        # Start with a few points to bootstrap
        initial = [[0,0], [1,0], [0,1], [1,1]]
        points = initial[:]
        
        for step in range(12):
            if time.time() > deadline - 30:
                break
            pos, gain = find_best_new_point(points)
            if pos is None:
                break
            points.append([pos[0], pos[1]])
        
        return points

    # Try greedy construction
    greedy_pts = greedy_construct()
    greedy_score = evaluate(greedy_pts)
    if greedy_score > best_score:
        best_score = greedy_score
        best_points = greedy_pts

    # Phase 2: Local search with line-aware moves
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    def apply_move(points, i, new_i, new_j):
        new_points = [p[:] for p in points]
        new_points[i] = [new_i, new_j]
        return new_points, evaluate(new_points)

    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9997
    no_improve_count = 0
    max_no_improve = 300

    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Generate diverse candidates
        candidates = []
        for _ in range(30):
            # Mix of local and global moves
            if random.random() < 0.7:
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
            else:
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            is_distinct = True
            for k, p in enumerate(current_points):
                if k != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            candidates.append((new_x, new_y))
        
        best_candidate = None
        best_candidate_score = current_score
        
        for (new_x, new_y) in candidates:
            new_points, new_score = apply_move(current_points, idx, new_x, new_y)
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = (new_points, new_score)
        
        if best_candidate is not None:
            new_points, new_score = best_candidate
            delta = new_score - current_score
            
            if delta >= 0:
                current_points = new_points
                current_score = new_score
                no_improve_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                if random.random() < pow(2.718281828, delta / T):
                    current_points = new_points
                    current_score = new_score
                    no_improve_count += 1
            T *= cooling_rate
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            T = 10.0
            no_improve_count = 0
            current_points = [p[:] for p in best_points]
            current_score = best_score

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(len(pts))
    line_map = build_line_map(pts)
    n4, n5 = score_from_line_map(line_map)
    print("4-lines:", n4, "5+ lines:", n5)
