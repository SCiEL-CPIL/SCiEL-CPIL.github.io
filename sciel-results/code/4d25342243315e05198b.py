"""Combined program: fast numpy scoring + simulated annealing local search + structured candidates."""
import time
import random
import numpy as np
from collections import Counter

def _fast_score(points):
    """Calculate max multiplicity of a positive doubled triangle area using numpy."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)

    for i in range(n):
        V = P[i + 1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)

    return int(np.max(counts))

def _score_and_best_area(points):
    """Return (max_multiplicity, best_area_value)."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)

    for i in range(n):
        V = P[i + 1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)

    best_area = int(np.argmax(counts))
    return int(counts[best_area]), best_area

def _generate_candidates():
    """Generate diverse structured candidates."""
    candidates = []

    # Standard grids
    grid_configs = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]
    for w, h in grid_configs:
        for ox in range(-30, 31 - w + 1, max(1, (60 - w) // 4)):
            for oy in range(-30, 31 - h + 1, max(1, (60 - h) // 4)):
                points = [(ox + x, oy + y) for x in range(w) for y in range(h)]
                if len(points) == 30:
                    candidates.append(points)

    # Sheared grids
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        for shear in range(-4, 5):
            for ox in range(-30, 31, 5):
                for oy in range(-30, 31, 5):
                    points = []
                    for i in range(w):
                        for j in range(h):
                            x = ox + i + shear * j
                            y = oy + j
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                if (x, y) not in points:
                                    points.append((x, y))
                    if len(points) == 30:
                        candidates.append(points)

    # Lattice patterns
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                for d in range(-3, 4):
                    if (a == 0 and b == 0) or (c == 0 and d == 0):
                        continue
                    det = a * d - b * c
                    if det == 0:
                        continue
                    for ox in range(-20, 21, 5):
                        for oy in range(-20, 21, 5):
                            points = []
                            for i in range(10):
                                for j in range(6):
                                    x = ox + i * a + j * c
                                    y = oy + i * b + j * d
                                    if -30 <= x <= 30 and -30 <= y <= 30:
                                        if (x, y) not in points:
                                            points.append((x, y))
                                    if len(points) >= 30:
                                        break
                                if len(points) >= 30:
                                    break
                            if len(points) == 30:
                                candidates.append(points)

    # Parabolic curves
    for a in [1, 2, 3]:
        for sx in range(-5, 6):
            for sy in range(-15, 16):
                pts = []
                for x in range(-10, 11):
                    y = a * x * x + sy
                    px = x + sx
                    py = y
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if len(set(pts)) == 30:
                        candidates.append(pts)

    # Random samples with fixed seeds
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for seed in [42, 123, 777, 2024, 314, 271]:
        random.seed(seed)
        pts = random.sample(all_coords, 30)
        candidates.append(pts)

    return candidates

def _local_search(points, time_limit, target_area=None):
    """Local search with simulated annealing, optimizing for max area multiplicity."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, current_best_area = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score

    T = 5.0
    cooling = 0.998
    min_T = 0.05
    occupied = set(points)
    target_a = current_best_area

    while time.time() - start_time < time_limit:
        idx = random.randint(0, 29)
        old_p = points[idx]

        # Try several candidate moves
        found_move = False
        for _ in range(10):
            # Mix of local and global moves
            if random.random() < 0.6:
                ref_idx = random.randint(0, 29)
                rx, ry = points[ref_idx]
                new_x = rx + random.randint(-5, 5)
                new_y = ry + random.randint(-5, 5)
            else:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            new_p = (new_x, new_y)
            if new_p == old_p or new_p in occupied:
                continue
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                continue
            found_move = True
            break

        if not found_move:
            continue

        points[idx] = new_p
        occupied.remove(old_p)
        occupied.add(new_p)

        new_score, new_best_area = _score_and_best_area(points)
        delta = new_score - current_score

        if new_best_area != target_a and new_score >= current_score:
            target_a = new_best_area

        # Acceptance criterion
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            current_best_area = new_best_area
            if new_score > best_score:
                best_score = new_score
                best_points = list(points)
        else:
            points[idx] = old_p
            occupied.remove(new_p)
            occupied.add(old_p)

        T *= cooling
        if T < min_T:
            T = min_T

    return best_points, best_score

def _polish(points, time_limit):
    """Focused polish: try all single-point moves to nearby positions."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, _ = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    occupied = set(points)

    for radius in [2, 3, 4, 5]:
        if time.time() - start_time > time_limit:
            break
        for idx in range(30):
            if time.time() - start_time > time_limit:
                break
            old_p = points[idx]
            ox, oy = old_p
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_p = (nx, ny)
                    if new_p in occupied:
                        continue

                    points[idx] = new_p
                    occupied.remove(old_p)
                    occupied.add(new_p)

                    new_score, _ = _score_and_best_area(points)

                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_points = list(points)
                    else:
                        points[idx] = old_p
                        occupied.remove(new_p)
                        occupied.add(old_p)

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin

    def time_ok(margin=0):
        return time.time() < deadline - margin

    best_points = None
    best_score = 0

    # Generate candidates and find best initial
    candidates = _generate_candidates()
    for points in candidates:
        if not time_ok(30):
            break
        if len(points) != 30:
            continue
        score = _fast_score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)

    # Local search from best candidate
    if best_points is not None and time_ok(30):
        improved_points, improved_score = _local_search(best_points, time_limit=25)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points

    # Polish
    if best_points is not None and time_ok(20):
        polished_points, polished_score = _polish(best_points, time_limit=15)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points

    # Additional random restarts with local search
    for _ in range(5):
        if not time_ok(15):
            break
        points = []
        used = set()
        while len(points) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                points.append((x, y))
                used.add((x, y))
        score = _fast_score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)
        if time_ok(10):
            improved_points, improved_score = _local_search(points, time_limit=8)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points

    # Final polish
    if best_points is not None and time_ok(10):
        polished_points, polished_score = _polish(best_points, time_limit=8)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points

    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]

    return best_points
