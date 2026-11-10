import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def collinear3(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if collinear3(points[i], points[j], points[k]):
                            count += 1
                if count >= 4:
                    return True
        return False

    def greedy_score(points):
        n = len(points)
        best_len = 0
        for start in range(n):
            subset = [start]
            for i in range(n):
                if i == start:
                    continue
                valid = True
                for j_idx in range(len(subset)):
                    for k_idx in range(j_idx + 1, len(subset)):
                        if collinear3(points[i], points[subset[j_idx]], points[subset[k_idx]]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    subset.append(i)
            best_len = max(best_len, len(subset))
        return best_len

    def make_parabola(offset_x, offset_y, scale):
        pts = []
        for i in range(20):
            x = offset_x + i * scale
            y = offset_y + (i - 10) ** 2 * scale
            if abs(x) <= 100 and abs(y) <= 100:
                pts.append([x, y])
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in [(p[0], p[1]) for p in pts]:
                pts.append([x, y])
        return pts[:20]

    random.seed(42)
    best_pts = None
    best_score = 20

    # Try multiple initial configurations
    for trial in range(5):
        if time.time() > deadline:
            break
        if trial == 0:
            pts = [
                [10, 0], [11, 0], [12, 0],
                [-12, 2], [-11, 2], [-10, 2],
                [10, 4], [11, 4], [12, 4],
                [-12, 6], [-11, 6], [-10, 6],
                [10, 8], [11, 8], [12, 8],
                [-12, 10], [-11, 10], [-10, 10],
                [50, 50],
                [-50, -50]
            ]
        else:
            ox = random.randint(-5, 5)
            oy = random.randint(-5, 5)
            sc = random.choice([2, 3, 4])
            pts = make_parabola(ox, oy, sc)

        if has_4_collinear(pts):
            continue

        score = greedy_score(pts)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in pts]

        # Local search
        current = [p[:] for p in pts]
        current_score = score

        for _ in range(200):
            if time.time() > deadline:
                break
            idx = random.randint(0, 19)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if (new_x, new_y) in [(p[0], p[1]) for p in current if current.index(p) != idx]:
                continue
            candidate = [p[:] for p in current]
            candidate[idx] = [new_x, new_y]
            if len(set(map(tuple, candidate))) != 20:
                continue
            if has_4_collinear(candidate):
                continue
            new_score = greedy_score(candidate)
            if new_score < current_score:
                current_score = new_score
                current = candidate
                if new_score < best_score:
                    best_score = new_score
                    best_pts = [p[:] for p in current]

    if best_pts is None:
        best_pts = [
            [10, 0], [11, 0], [12, 0],
            [-12, 2], [-11, 2], [-10, 2],
            [10, 4], [11, 4], [12, 4],
            [-12, 6], [-11, 6], [-10, 6],
            [10, 8], [11, 8], [12, 8],
            [-12, 10], [-11, 10], [-10, 10],
            [50, 50],
            [-50, -50]
        ]

    return best_pts
