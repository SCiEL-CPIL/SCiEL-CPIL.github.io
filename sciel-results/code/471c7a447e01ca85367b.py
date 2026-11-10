import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def better_initial_config():
        """Generate a configuration using a grid with perturbations to break collinearities."""
        # Use a 5x4 grid pattern with strategic perturbations
        base_points = []
        for i in range(5):
            for j in range(4):
                x = -40 + i * 20 + (j % 2) * 3
                y = -30 + j * 20 + (i % 2) * 2
                base_points.append([x, y])
        return base_points

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

    def improved_local_search(current, current_score, deadline):
        """More effective local search with multiple swap attempts."""
        best_current = [p[:] for p in current]
        best_score = current_score
        no_improve_count = 0
        max_no_improve = 50

        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            for idx in range(20):
                if time.time() > deadline:
                    break
                for _ in range(30):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    if (new_x, new_y) in [(p[0], p[1]) for p in best_current if best_current.index(p) != idx]:
                        continue
                    candidate = [p[:] for p in best_current]
                    candidate[idx] = [new_x, new_y]
                    if len(set(map(tuple, candidate))) != 20:
                        continue
                    if has_4_collinear(candidate):
                        continue
                    new_score = greedy_score(candidate)
                    if new_score < best_score:
                        best_score = new_score
                        best_current = candidate
                        improved = True
                        no_improve_count = 0
                        break
            if not improved:
                no_improve_count += 1

        return best_current, best_score

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
    initial_configs = [
        better_initial_config(),
        make_parabola(0, 0, 3),
        make_parabola(-5, -5, 4),
        make_parabola(5, 5, 2),
        make_parabola(-3, 3, 3)
    ]

    for trial, pts in enumerate(initial_configs):
        if time.time() > deadline:
            break

        if has_4_collinear(pts):
            continue

        score = greedy_score(pts)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in pts]

        # Improved local search
        current, current_score = improved_local_search(pts, score, deadline)
        if current_score < best_score:
            best_score = current_score
            best_pts = current

    if best_pts is None:
        best_pts = better_initial_config()

    return best_pts
