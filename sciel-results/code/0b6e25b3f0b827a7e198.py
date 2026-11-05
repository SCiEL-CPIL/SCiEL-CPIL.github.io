import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    n = 30

    # ---------- Helper: full area computation ----------
    def _calculate_all_areas(points):
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, {}
        return max(areas.values()), areas

    # ---------- Helper: fast vectorized delta for moving one point ----------
    def calc_move_delta(points, idx, old_pos, new_pos, target_areas):
        """Return dict {area: delta_count} for moving points[idx] from old_pos to new_pos."""
        x1, y1 = new_pos
        ox, oy = old_pos
        others = [points[j] for j in range(n) if j != idx]
        if len(others) < 2:
            return {a: 0 for a in target_areas}
        
        other_arr = np.array(others, dtype=np.float64)
        m = len(others)
        # All pairs
        ii, jj = np.triu_indices(m, 1)
        p1 = other_arr[ii]
        p2 = other_arr[jj]
        
        dx1n = p1[:, 0] - x1
        dy1n = p1[:, 1] - y1
        dx2n = p2[:, 0] - x1
        dy2n = p2[:, 1] - y1
        det_new = np.abs(dx1n * dy2n - dy1n * dx2n)
        
        dx1o = p1[:, 0] - ox
        dy1o = p1[:, 1] - oy
        dx2o = p2[:, 0] - ox
        dy2o = p2[:, 1] - oy
        det_old = np.abs(dx1o * dy2o - dy1o * dx2o)
        
        results = {}
        for a in target_areas:
            results[a] = int(np.sum(det_new == a)) - int(np.sum(det_old == a))
        return results

    # ---------- Generate diverse initial candidates ----------
    candidates = []
    
    # Grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)

    # Sheared grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3)]:
        for k in [1, -1, 2, -2]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid:
                    break
            if valid and len(pts) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # Random
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        pts = []
        attempts = 0
        while len(pts) < 30 and attempts < 500:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
            attempts += 1
        if len(pts) == 30:
            candidates.append(pts)

    if not candidates:
        return [(0, 0)] * 30

    # ---------- Evaluate candidates ----------
    best_score = 0
    best_points = candidates[0]
    best_areas = {}
    for pts in candidates:
        score, areas = _calculate_all_areas(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            best_areas = areas

    current_points = list(best_points)
    current_score = best_score
    current_areas = dict(best_areas)

    # ---------- Simulated Annealing with targeted moves ----------
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 2:
        no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart with random perturbation
            for _ in range(5):
                idx = random.randint(0, 29)
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                if (new_x, new_y) not in current_points:
                    current_points[idx] = (new_x, new_y)
            no_improve = 0
            T = max(T, 20.0)
            continue

        # Choose strategy: 60% targeted (use existing high-area triples), 40% random
        if current_areas and random.random() < 0.6:
            # Find top area
            top_area = max(current_areas, key=lambda a: current_areas[a])
            target_areas = list(current_areas.keys())[:10]  # top 10 areas
            target_areas = sorted(target_areas, key=lambda a: -current_areas.get(a, 0))[:10]
            
            # Pick a random existing triple with the top area to "guide" the move
            # We'll just use target_areas for delta calculation
            # Pick a point to move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            
            # Try moves in a neighborhood, biased toward creating target areas
            best_move = None
            best_delta_score = 0
            
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx = old_x + dx
                    ny = old_y + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    if (nx, ny) in current_points:
                        continue
                    
                    deltas = calc_move_delta(current_points, idx, (old_x, old_y), (nx, ny), target_areas)
                    # Score: sum of positive deltas weighted by current count
                    s = 0
                    for a in target_areas:
                        d = deltas[a]
                        w = current_areas.get(a, 0)
                        if d > 0:
                            s += d * w
                        else:
                            s += d * w * 1.2
                    if s > best_delta_score:
                        best_delta_score = s
                        best_move = (nx, ny)
            
            if best_move is None:
                # Random small move
                for _ in range(10):
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    nx = old_x + dx
                    ny = old_y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx, ny) not in current_points:
                        best_move = (nx, ny)
                        break
                if best_move is None:
                    continue
        else:
            # Random move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            target_areas = list(current_areas.keys())[:10] if current_areas else []
            for _ in range(20):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx = old_x + dx
                ny = old_y + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx, ny) not in current_points:
                    best_move = (nx, ny)
                    break
            if best_move is None:
                continue

        # Apply move and evaluate
        old_pos = current_points[idx]
        current_points[idx] = best_move
        
        new_score, new_areas = _calculate_all_areas(current_points)
        
        # Acceptance criterion (simulated annealing)
        if new_score > current_score:
            current_score = new_score
            current_areas = new_areas
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
            no_improve = 0
        else:
            # Accept with probability exp((new - old) / T)
            delta = new_score - current_score
            if delta >= 0 or random.random() < np.exp(delta / max(T, 0.01)):
                current_areas = new_areas
            else:
                # Reject
                current_points[idx] = old_pos

        T *= cooling
        if T < T_min:
            T = T_min

    # Final evaluation
    final_score, final_areas = _calculate_all_areas(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = list(current_points)

    return best_points

def _calculate_max_multiplicity(points):
    """Compatibility wrapper."""
    n = len(points)
    if n < 3:
        return 0, 0
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
    if not areas:
        return 0, 0
    max_mult = max(areas.values())
    max_area = max(areas, key=lambda a: areas[a])
    return max_mult, max_area
