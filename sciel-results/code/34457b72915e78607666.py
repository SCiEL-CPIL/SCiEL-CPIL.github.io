import time
import random
import math


def _evaluate(points):
    """Return (score, valid). score = # lines with exactly 4 points; valid = no 5 collinear."""
    n = len(points)
    line_map = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            A = yi - yj
            B = xj - xi
            C = xi * yj - xj * yi
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            if A < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0 and C < 0:
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key in line_map:
                line_map[key].add(i)
                line_map[key].add(j)
            else:
                line_map[key] = {i, j}
    score = 0
    for key, idx_set in line_map.items():
        A, B, C = key
        cnt = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                cnt += 1
        if cnt >= 5:
            return 0, False
        if cnt == 4:
            score += 1
    return score, True


def construct():
    start_time = time.time()
    deadline = start_time + 130

    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _evaluate(best_points)

    random.seed(42)
    current_points = [p[:] for p in best_points]
    current_score = best_score

    T = 3.0
    T_min = 0.05
    cooling = 0.9999

    for _ in range(1000000):
        if time.time() > deadline:
            break

        idx = random.randint(0, 15)
        old = current_points[idx][:]

        r = random.random()
        if r < 0.2:
            nx = old[0] + random.randint(-4, 4)
            ny = old[1] + random.randint(-4, 4)
        elif r < 0.4:
            nx = random.randint(-20, 20)
            ny = random.randint(-20, 20)
        elif r < 0.7:
            nx = random.randint(-80, 80)
            ny = random.randint(-80, 80)
        else:
            nx = random.randint(-400, 400)
            ny = random.randint(-400, 400)

        nx = max(-1000, min(1000, nx))
        ny = max(-1000, min(1000, ny))

        np_ = [nx, ny]
        if np_ in current_points:
            continue

        current_points[idx] = np_
        ns, valid = _evaluate(current_points)

        if not valid:
            current_points[idx] = old
            continue

        if ns > current_score:
            current_score = ns
            if ns > best_score:
                best_score = ns
                best_points = [p[:] for p in current_points]
        else:
            d = ns - current_score
            if d < 0 and T > T_min and random.random() < math.exp(d / T):
                current_score = ns
            else:
                current_points[idx] = old

        T = max(T_min, T * cooling)

        if random.random() < 0.0005:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 3.0

    return best_points
