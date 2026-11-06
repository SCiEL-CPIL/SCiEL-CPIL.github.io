import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        if A == 0 and B == 0:
            return None
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def evaluate_fast(points):
        """Fast evaluation: count points per line using dict."""
        n = len(points)
        line_points = {}  # line_key -> set of point indices
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        is_valid = True
        for key, idx_set in line_points.items():
            count = len(idx_set)
            if count >= 5:
                is_valid = False
                return 0, False
            if count == 4:
                score += 1
        return score, is_valid

    def evaluate_with_lines(points):
        """Return score and the set of 4-point line keys."""
        n = len(points)
        line_points = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        is_valid = True
        four_point_lines = set()
        for key, idx_set in line_points.items():
            count = len(idx_set)
            if count >= 5:
                is_valid = False
                return 0, False, set()
            if count == 4:
                score += 1
                four_point_lines.add(key)
        return score, is_valid, four_point_lines

    # Initial configurations to try
    def initial_4x4_grid():
        return [[x, y] for x in range(4) for y in range(4)]

    def initial_perturbed_grid(seed):
        random.seed(seed)
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x * 3 + random.randint(-1, 1), y * 3 + random.randint(-1, 1)])
        return pts

    def try_configuration(points):
        """Try a configuration, return (score, valid, points) or None if invalid."""
        score, valid = evaluate_fast(points)
        if valid:
            return score, valid, points
        return None

    best_score = -1
    best_points = None

    # Try multiple initial configurations
    initial_configs = [initial_4x4_grid()]
    for seed in range(20):
        initial_configs.append(initial_perturbed_grid(seed))

    for config in initial_configs:
        result = try_configuration(config)
        if result:
            score, valid, pts = result
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in pts]

    if best_points is None:
        best_points = initial_4x4_grid()
        best_score, _ = evaluate_fast(best_points)

    # Local search with targeted moves
    def local_search(points, current_score, max_time):
        """Local search that tries to create new 4-point lines."""
        points = [p[:] for p in points]
        current_score = current_score
        T = 2.0
        cooling = 0.99
        steps = 0
        no_improve = 0
        max_no_improve = 300

        while time.time() < max_time:
            steps += 1
            idx = random.randint(0, 15)
            old_p = points[idx][:]

            # Try different move strategies
            strategy = random.random()
            if strategy < 0.4:
                # Small random move
                move_range = random.choice([3, 5, 8, 12])
                dx = random.randint(-move_range, move_range)
                dy = random.randint(-move_range, move_range)
            elif strategy < 0.7:
                # Move toward another point to potentially create alignment
                other_idx = random.randint(0, 15)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % 16
                ox, oy = points[other_idx]
                px, py = old_p
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            else:
                # Larger jump
                move_range = random.choice([15, 25, 40])
                dx = random.randint(-move_range, move_range)
                dy = random.randint(-move_range, move_range)

            new_p = [old_p[0] + dx, old_p[1] + dy]

            if not (-1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000):
                continue
            if any(p == new_p for p in points if p != old_p):
                continue

            points[idx] = new_p
            new_score, new_valid = evaluate_fast(points)

            if new_valid:
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / max(T, 0.01)):
                        current_score = new_score
                    else:
                        points[idx] = old_p
            else:
                points[idx] = old_p
                no_improve += 1

            T *= cooling
            if T < 0.05:
                T = 0.05

            if no_improve > max_no_improve:
                break

        return points, current_score

    # Run local search from best configuration
    max_search_time = deadline - 5.0
    points, score = local_search(best_points, best_score, max_search_time)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in points]

    # Try a few more restarts if time permits
    restart_count = 0
    while time.time() < deadline - 10.0 and restart_count < 10:
        restart_count += 1
        # Perturb best points
        points = []
        for p in best_points:
            new_p = [p[0] + random.randint(-8, 8), p[1] + random.randint(-8, 8)]
            new_p[0] = max(-1000, min(1000, new_p[0]))
            new_p[1] = max(-1000, min(1000, new_p[1]))
            points.append(new_p)

        # Fix collisions
        for i in range(16):
            for j in range(i + 1, 16):
                if points[i] == points[j]:
                    points[j] = [points[j][0] + random.randint(1, 5), points[j][1]]

        score, valid = evaluate_fast(points)
        if valid:
            # Local search from this restart
            remaining_time = deadline - 10.0 - time.time()
            if remaining_time > 5:
                points, score = local_search(points, score, remaining_time)
                if score > best_score:
                    best_score = score
                    best_points = [p[:] for p in points]

    final_score, final_valid = evaluate_fast(best_points)
    if final_valid:
        return best_points
    else:
        # Fallback to 4x4 grid
        grid = initial_4x4_grid()
        return grid
