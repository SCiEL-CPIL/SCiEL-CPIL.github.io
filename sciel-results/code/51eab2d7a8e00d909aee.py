import time
import random
import sys
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return (0, 0, 0)
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def get_collinear_triples(points):
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.append((pts[i], pts[j], pts[k]))
        return triples

    def compute_exact_score(points):
        """
        Branch and bound to find the maximum size of a subset with no 3 collinear.
        This is equivalent to finding the maximum independent set in the 3-uniform hypergraph of collinear triples.
        """
        triples = get_collinear_triples(points)
        if not triples:
            return N

        # conflict_mask[i] has bits set for all j, k such that (i, j, k) is a triple.
        # If we include i, we must exclude all j, k in conflict_mask[i].
        conflict_mask = [0] * N
        for i, j, k in triples:
            conflict_mask[i] |= (1 << j) | (1 << k)
            conflict_mask[j] |= (1 << i) | (1 << k)
            conflict_mask[k] |= (1 << i) | (1 << j)

        best = [0]

        def backtrack(available_mask, current_size):
            # Pruning: even if we take all remaining, can we beat best?
            if current_size + bin(available_mask).count('1') <= best[0]:
                return

            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return

            # Pick a vertex. Heuristic: pick the one with the most conflicts within the available set.
            # This helps prune the "include" branch faster.
            best_v = -1
            max_conflicts = -1
            temp_mask = available_mask
            while temp_mask:
                v_bit = temp_mask & -temp_mask
                v = v_bit.bit_length() - 1
                # Count conflicts of v within available_mask
                # conflicts = conflict_mask[v] & available_mask
                # Note: v itself is in available_mask, but conflict_mask[v] doesn't contain v.
                # We want to know how many points we would remove if we include v.
                # Removing v and its conflicts reduces the available set.
                # A good heuristic is to pick v that maximizes the number of removed points.
                removed = (conflict_mask[v] & available_mask) | v_bit
                cnt = bin(removed).count('1')
                if cnt > max_conflicts:
                    max_conflicts = cnt
                    best_v = v
                    best_v_bit = v_bit
                temp_mask ^= v_bit

            if best_v == -1:
                return

            v_bit = best_v_bit

            # Option 1: Exclude v
            backtrack(available_mask ^ v_bit, current_size)

            # Option 2: Include v
            # Remove v and all its conflicts
            new_available = available_mask ^ v_bit
            new_available &= ~conflict_mask[best_v]

            backtrack(new_available, current_size + 1)

        backtrack((1 << N) - 1, 0)
        return best[0]

    def generate_random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    def generate_structured_points():
        """
        Generate points with some structure to encourage collinearities.
        e.g., points on a few lines, or near a grid.
        """
        pts = []
        # Try a few different structured generators
        method = random.randint(0, 2)
        if method == 0:
            # Points on 2 or 3 lines + noise
            num_lines = random.randint(2, 3)
            lines = []
            for _ in range(num_lines):
                # Random line y = mx + c
                m = random.randint(-5, 5)
                c = random.randint(-50, 50)
                lines.append((m, c))

            for _ in range(N):
                if random.random() < 0.7:
                    m, c = random.choice(lines)
                    x = random.randint(COORD_MIN, COORD_MAX)
                    y = m * x + c
                    if -100 <= y <= 100:
                        p = [x, y]
                        if p not in pts:
                            pts.append(p)
                # else random
                while len(pts) < N:
                    x = random.randint(COORD_MIN, COORD_MAX)
                    y = random.randint(COORD_MIN, COORD_MAX)
                    p = [x, y]
                    if p not in pts:
                        pts.append(p)
                        break
        elif method == 1:
            # Grid-like with perturbations
            step = random.randint(8, 15)
            start_x = random.randint(-90, 90 - step * 4)
            start_y = random.randint(-90, 90 - step * 4)
            for i in range(5):
                for j in range(4):
                    x = start_x + i * step + random.randint(-2, 2)
                    y = start_y + j * step + random.randint(-2, 2)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        p = [x, y]
                        if p not in pts:
                            pts.append(p)
            while len(pts) < N:
                x = random.randint(COORD_MIN, COORD_MAX)
                y = random.randint(COORD_MIN, COORD_MAX)
                p = [x, y]
                if p not in pts:
                    pts.append(p)
        else:
            # Random
            return generate_random_points()

        return pts[:N]

    best_config = None
    best_score = N + 1

    # Initial search for a valid configuration
    # Try both random and structured
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        if random.random() < 0.5:
            pts = generate_random_points()
        else:
            pts = generate_structured_points()
        if is_valid(pts):
            best_config = pts
            break

    if best_config is None:
        # Fallback: Parabola
        return [[x, x * x] for x in range(-10, 10)]

    current_pts = [p[:] for p in best_config]
    current_score = compute_exact_score(current_pts)
    best_score = current_score
    best_config = [p[:] for p in current_pts]

    # Local Search with Simulated Annealing
    # We want to minimize score.
    temp = 5.0
    cooling = 0.995
    step_count = 0
    no_improve_steps = 0
    max_no_improve = 1000

    while time.time() < deadline - 5:
        step_count += 1
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]

        # New random position
        new_x = random.randint(COORD_MIN, COORD_MAX)
        new_y = random.randint(COORD_MIN, COORD_MAX)

        if new_x == old_x and new_y == old_y:
            continue

        if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
            continue

        current_pts[idx] = [new_x, new_y]

        if not is_valid(current_pts):
            current_pts[idx] = [old_x, old_y]
            continue

        new_score = compute_exact_score(current_pts)

        delta = new_score - current_score

        if new_score < best_score:
            best_score = new_score
            best_config = [p[:] for p in current_pts]
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps > max_no_improve:
                # Restart with new random/structured config to escape global local minima
                for _ in range(10):
                    if time.time() > deadline - 10: break
                    if random.random() < 0.5:
                        new_pts = generate_random_points()
                    else:
                        new_pts = generate_structured_points()
                    if is_valid(new_pts):
                        current_pts = [p[:] for p in new_pts]
                        current_score = compute_exact_score(current_pts)
                        break
                no_improve_steps = 0
                temp = 5.0
                continue

        if delta < 0:
            # Improvement
            current_score = new_score
        else:
            # Deterioration or neutral
            if random.random() < pow(2.718, -delta / temp):
                current_score = new_score
            else:
                current_pts[idx] = [old_x, old_y]
                continue

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_config

if __name__ == "__main__":
    res = construct()
    print(res)
