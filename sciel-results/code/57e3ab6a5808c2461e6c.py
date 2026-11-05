"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _evaluate(points):
    """Return the number of lines containing exactly 4 points, or -1 if invalid.

    Single pass: builds the multiset of lines through every pair, counts them,
    and early-exits as soon as any line reaches 5 points (five-collinear).
    """
    n = len(points)
    if n != 16:
        return -1

    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))

    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            cnt = line_counts[key] + 1
            line_counts[key] = cnt
            if cnt >= 5:
                return -1

    four_count = 0
    for cnt in line_counts.values():
        if cnt == 4:
            four_count += 1
    return four_count

def _perturb(points, rng):
    """Create a neighbor using a mix of move types."""
    new_points = [p[:] for p in points]
    move_type = rng.random()

    if move_type < 0.45:
        # Random jump for 1-2 points
        num_moves = rng.choice([1, 1, 2])
        for _ in range(num_moves):
            idx = rng.randint(0, 15)
            span = rng.choice([10, 30, 100])
            new_points[idx][0] += rng.randint(-span, span)
            new_points[idx][1] += rng.randint(-span, span)
    elif move_type < 0.75:
        # Translate a small spatial cluster
        center_idx = rng.randint(0, 15)
        cx, cy = new_points[center_idx]
        dx = rng.randint(-15, 15)
        dy = rng.randint(-15, 15)
        for idx in range(16):
            if abs(new_points[idx][0] - cx) <= 3 and abs(new_points[idx][1] - cy) <= 3:
                new_points[idx][0] += dx
                new_points[idx][1] += dy
    elif move_type < 0.9:
        # Swap the full coordinates of two points
        i, j = rng.sample(range(16), 2)
        new_points[i][0], new_points[j][0] = new_points[j][0], new_points[i][0]
        new_points[i][1], new_points[j][1] = new_points[j][1], new_points[i][1]
    else:
        # Re-place a single point entirely randomly
        idx = rng.randint(0, 15)
        new_points[idx][0] = rng.randint(-1000, 1000)
        new_points[idx][1] = rng.randint(-1000, 1000)

    return new_points

def construct():
    """Find 16 integer points maximizing 4-point lines with no five collinear."""
    deadline = time.time() + 110  # safety margin before the 150s kill

    init_grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in init_grid]
    best_score = _evaluate(best_points)

    rng = random.Random(42)

    while time.time() < deadline - 5:
        if best_score <= 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        elif rng.random() < 0.75:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            current_points = [[rng.randint(-30, 30), rng.randint(-30, 30)] for _ in range(16)]
            current_score = _evaluate(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                continue

        temperature = 30.0
        cooling = 0.995
        min_temperature = 0.5
        max_iter = 5000

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if temperature < min_temperature:
                break

            new_points = _perturb(current_points, rng)
            new_score = _evaluate(new_points)

            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    temperature = min(temperature, 30.0)
            elif new_score >= 0:
                delta = current_score - new_score
                if rng.random() < math.exp(-delta / max(temperature, 0.01)):
                    current_points = new_points
                    current_score = new_score

            temperature *= cooling

    return best_points

# EVOLVE-BLOCK-END
