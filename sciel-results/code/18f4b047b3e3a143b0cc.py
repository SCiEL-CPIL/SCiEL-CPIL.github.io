import time
import random
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 30
    bounds = 30

    def compute_all_areas(pts):
        """Compute area counter using NumPy vectorization."""
        pts_arr = np.array(pts, dtype=np.float64)
        n = len(pts_arr)
        areas = []
        # Vectorized: for each pair (i,j), compute cross product with all k
        for i in range(n):
            for j in range(i+1, n):
                # Vector over k
                diffs = pts_arr[j] - pts_arr[i]
                # Cross product with all other points
                vecs = pts_arr - pts_arr[i]
                cross = diffs[0] * vecs[:, 1] - diffs[1] * vecs[:, 0]
                # Only consider k > j to avoid double counting and self
                cross[j+1:] = cross[j+1:]  # keep
                cross[:j+1] = 0  # mask out i, j and earlier
                valid = cross > 0
                if valid.any():
                    areas.extend(cross[valid].tolist())
        if not areas:
            return Counter()
        return Counter(areas)

    def compute_all_areas_fast(pts):
        """Fast area computation using structured loops but with numpy for inner."""
        pts_arr = np.array(pts, dtype=np.float64)
        n = len(pts_arr)
        area_counts = Counter()
        for i in range(n - 2):
            xi, yi = pts_arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = pts_arr[j]
                # Vectorized over k > j
                xk = pts_arr[j+1:, 0]
                yk = pts_arr[j+1:, 1]
                # 2A = |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
                vals = np.abs(xi*(yj - yk) + xj*(yk - yi) + xk*(yi - yj))
                valid = vals > 0
                if valid.any():
                    for v in vals[valid].astype(int):
                        area_counts[v] += 1
        return area_counts

    def get_max_count(counts):
        if not counts:
            return 0
        return max(counts.values())

    def get_best_area(counts):
        if not counts:
            return 0
        return counts.most_common(1)[0][0]

    def compute_delta(old_pt, new_pt, pts, idx):
        """Compute removed and added area counts when replacing pts[idx]."""
        removed = Counter()
        added = Counter()
        x_old, y_old = old_pt
        x_new, y_new = new_pt
        others = [i for i in range(N) if i != idx]
        n_others = len(others)
        for a in range(n_others):
            i = others[a]
            x1, y1 = pts[i]
            for b in range(a + 1, n_others):
                j = others[b]
                x2, y2 = pts[j]
                d_old = abs(x_old*(y1-y2) + x1*(y2-y_old) + x2*(y_old-y1))
                if d_old > 0:
                    removed[d_old] += 1
                d_new = abs(x_new*(y1-y2) + x1*(y2-y_new) + x2*(y_new-y1))
                if d_new > 0:
                    added[d_new] += 1
        return removed, added

    def local_search(pts, counts, current_max_count, deadline, max_stagnation=2000):
        """Improve pts via local search."""
        best_max_count = current_max_count
        best_pts = pts[:]
        stagnation_count = 0

        while time.time() < deadline:
            if stagnation_count > max_stagnation:
                # Restart from best with random perturbation
                pts = best_pts[:]
                for _ in range(random.randint(3, 8)):
                    idx = random.randint(0, N - 1)
                    x, y = pts[idx]
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    nx, ny = x + dx, y + dy
                    if -bounds <= nx <= bounds and -bounds <= ny <= bounds:
                        new_pt = (nx, ny)
                        if new_pt not in pts:
                            pts[idx] = new_pt
                counts = compute_all_areas_fast(pts)
                current_max_count = get_max_count(counts)
                if current_max_count > best_max_count:
                    best_max_count = current_max_count
                    best_pts = pts[:]
                stagnation_count = 0
                continue

            idx = random.randint(0, N - 1)
            x, y = pts[idx]
            # Biased neighborhood: smaller moves more likely
            r = random.random()
            if r < 0.5:
                dx = random.choice([-1, 1])
                dy = random.choice([-1, 1])
            elif r < 0.8:
                dx = random.choice([-2, -1, 1, 2])
                dy = random.choice([-2, -1, 1, 2])
            else:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
                stagnation_count += 1
                continue
            new_pt = (nx, ny)
            if new_pt in pts:
                stagnation_count += 1
                continue

            old_pt = pts[idx]
            removed, added = compute_delta(old_pt, new_pt, pts, idx)

            for area, cnt in removed.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in added.items():
                counts[area] += cnt

            new_max_count = get_max_count(counts)

            if new_max_count >= current_max_count:
                pts[idx] = new_pt
                current_max_count = new_max_count
                stagnation_count = 0
                if current_max_count > best_max_count:
                    best_max_count = current_max_count
                    best_pts = pts[:]
            else:
                for area, cnt in added.items():
                    counts[area] -= cnt
                    if counts[area] <= 0:
                        del counts[area]
                for area, cnt in removed.items():
                    counts[area] += cnt
                stagnation_count += 1

        return best_pts, best_max_count

    def try_arithmetic_grid(sx, sy, ox, oy, n_rows, n_cols, total):
        """Generate points on an arithmetic grid."""
        pts = []
        for i in range(n_rows):
            for j in range(n_cols):
                x = ox + j * sx
                y = oy + i * sy
                if -bounds <= x <= bounds and -bounds <= y <= bounds:
                    pts.append((x, y))
                    if len(pts) == total:
                        break
            if len(pts) == total:
                break
        return pts if len(pts) == total else None

    def try_lattice_config(base_x, base_y, v1x, v1y, v2x, v2y):
        """Generate points on a 2D lattice."""
        pts = []
        seen = set()
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = base_x + i * v1x + j * v2x
                y = base_y + i * v1y + j * v2y
                if -bounds <= x <= bounds and -bounds <= y <= bounds:
                    p = (x, y)
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
                if len(pts) >= total:
                    break
            if len(pts) >= total:
                break
        return pts[:total] if len(pts) >= total else None

    total = N
    best_overall = None
    best_score_overall = 0

    # Generate diverse initial configurations
    initial_configs = []

    # 1. Simple grid: 6x5
    cfg = try_arithmetic_grid(1, 1, -2, -2, 5, 6, total)
    if cfg:
        initial_configs.append(cfg)

    # 2. Grid with spacing 2
    cfg = try_arithmetic_grid(2, 2, -5, -4, 5, 6, total)
    if cfg:
        initial_configs.append(cfg)

    # 3. Hexagonal-like lattice
    cfg = try_lattice_config(0, 0, 2, 0, 1, 2)
    if cfg:
        initial_configs.append(cfg)

    # 4. Another lattice
    cfg = try_lattice_config(0, 0, 3, 0, 1, 2)
    if cfg:
        initial_configs.append(cfg)

    # 5. Diagonal lattice
    cfg = try_lattice_config(-5, -5, 2, 2, 3, -1)
    if cfg:
        initial_configs.append(cfg)

    # 6. Arithmetic progression in x, varying y
    pts = []
    for i in range(total):
        x = -14 + i
        y = (i * 7) % 29 - 14
        if -bounds <= x <= bounds and -bounds <= y <= bounds:
            pts.append((x, y))
    if len(pts) == total:
        initial_configs.append(pts)

    # 7. Random structured
    random.seed(42)
    for _ in range(5):
        sx = random.randint(1, 4)
        sy = random.randint(1, 4)
        ox = random.randint(-15, 15 - sx * 5)
        oy = random.randint(-15, 15 - sy * 4)
        cfg = try_arithmetic_grid(sx, sy, ox, oy, 5, 6, total)
        if cfg:
            initial_configs.append(cfg)

    # 8. Pure arithmetic progression
    pts = [(i, 0) for i in range(total)]
    initial_configs.append(pts)

    # 9. Two-row configuration
    pts = []
    for i in range(15):
        pts.append((i - 7, 0))
        pts.append((i - 7, 3))
    initial_configs.append(pts[:total])

    # 10. Random
    for _ in range(3):
        random.seed(random.randint(0, 100000))
        pts = set()
        while len(pts) < total:
            x = random.randint(-bounds, bounds)
            y = random.randint(-bounds, bounds)
            pts.add((x, y))
        initial_configs.append(list(pts))

    # Evaluate each initial config and run local search
    for cfg in initial_configs:
        if time.time() > deadline - 10:
            break
        if len(cfg) != total:
            continue
        # Check distinctness
        if len(set(cfg)) != total:
            continue
        counts = compute_all_areas_fast(cfg)
        current_max = get_max_count(counts)
        if current_max > best_score_overall:
            best_score_overall = current_max
            best_overall = cfg[:]

        # Run local search with remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        search_deadline = time.time() + min(remaining - 2, 25)
        improved_pts, improved_score = local_search(cfg, counts, current_max, search_deadline)
        if improved_score > best_score_overall:
            best_score_overall = improved_score
            best_overall = improved_pts[:]

    # Final: try a few more random restarts if time permits
    if best_overall is None:
        best_overall = [(i, j) for i in range(6) for j in range(5)]

    return best_overall


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} points")
    counts = compute_all_areas_fast(result)
    print(f"Best area multiplicity: {get_max_count(counts)}")
    print(f"Best area value: {get_best_area(counts)}")
