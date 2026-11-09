import time
import numpy as np
import random
from itertools import combinations
from collections import Counter, defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def score_of(pts):
        """Compute max multiplicity of any non-zero doubled area."""
        P = np.array(pts, dtype=np.int64)
        n = len(P)
        if n < 3:
            return 0
        idx = np.array(list(combinations(range(n), 3)), dtype=np.int64)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        xk, yk = P[k, 0], P[k, 1]
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas != 0]
        if len(areas) == 0:
            return 0
        _, cnts = np.unique(areas, return_counts=True)
        return int(cnts.max())

    def get_area_counts(pts):
        """Get Counter of all non-zero doubled areas."""
        P = np.array(pts, dtype=np.int64)
        n = len(P)
        if n < 3:
            return Counter()
        idx = np.array(list(combinations(range(n), 3)), dtype=np.int64)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        xk, yk = P[k, 0], P[k, 1]
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas != 0]
        if len(areas) == 0:
            return Counter()
        vals, cnts = np.unique(areas, return_counts=True)
        return dict(zip(vals.tolist(), cnts.tolist()))

    def fast_score(pts):
        """Fast score computation for local search."""
        P = np.array(pts, dtype=np.int64)
        n = len(P)
        if n < 3:
            return 0
        idx = np.array(list(combinations(range(n), 3)), dtype=np.int64)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        xk, yk = P[k, 0], P[k, 1]
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        areas = areas[areas != 0]
        if len(areas) == 0:
            return 0
        _, cnts = np.unique(areas, return_counts=True)
        return int(cnts.max())

    # Generate diverse initial candidates
    def gen_candidates():
        cands = []
        
        # Two parallel lines
        for y1, y2, n1, n2 in [(0, 1, 15, 15), (0, 2, 15, 15), (-1, 1, 15, 15)]:
            cand = [(x, y1) for x in range(n1)] + [(x, y2) for x in range(n2)]
            if len(cand) == 30:
                cands.append(cand)
        
        # Three parallel lines
        for y1, y2, y3, n1, n2, n3 in [(0, 1, 2, 10, 10, 10), (-1, 0, 1, 10, 10, 10)]:
            cand = [(x, y1) for x in range(n1)] + [(x, y2) for x in range(n2)] + [(x, y3) for x in range(n3)]
            if len(cand) == 30:
                cands.append(cand)
        
        # Grid structures
        for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
            cand = [(x, y) for x in range(w) for y in range(h)]
            if len(cand) == 30:
                cands.append(cand)
        
        # Diagonal structures
        for slope, intercept in [(1, 0), (1, 1), (2, 0), (-1, 0)]:
            cand = []
            seen = set()
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30 and (x, y) not in seen:
                    cand.append((x, y))
                    seen.add((x, y))
                    if len(cand) >= 30:
                        break
            if len(cand) == 30:
                cands.append(cand)
        
        # Random candidates
        random.seed(42)
        for seed_val in [42, 123, 456]:
            random.seed(seed_val)
            cand = []
            seen = set()
            while len(cand) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in seen:
                    cand.append((x, y))
                    seen.add((x, y))
            cands.append(cand)
        
        return cands

    # Evaluate all candidates and pick best
    best_points = None
    best_score = 0
    cands = gen_candidates()
    for cand in cands:
        if len(cand) != 30:
            continue
        s = fast_score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)

    # Local search with target-area concentration
    if best_points is None:
        # Fallback: return a valid 30-point set
        return [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]

    current_points = list(best_points)
    current_score = best_score
    used_set = set(current_points)
    
    random.seed(42)
    step_size = 2
    no_improve = 0
    max_no_improve = 800
    
    # Get target area
    area_counts = get_area_counts(current_points)
    target_area = max(area_counts.keys(), key=lambda x: area_counts[x]) if area_counts else 1

    while elapsed() < DEADLINE - 5:
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Adaptive step size
        r = random.random()
        if r < 0.01:
            sx = random.randint(-10, 10)
            sy = random.randint(-10, 10)
        elif r < 0.1:
            sx = random.randint(-5, 5)
            sy = random.randint(-5, 5)
        else:
            sx = random.randint(-step_size, step_size)
            sy = random.randint(-step_size, step_size)
        
        new_x, new_y = old_x + sx, old_y + sy
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y) or new_pt in used_set:
            continue
        
        # Try move
        used_set.discard((old_x, old_y))
        used_set.add(new_pt)
        current_points[idx] = new_pt
        new_score = fast_score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                # Refresh target area occasionally
                if random.random() < 0.1:
                    area_counts = get_area_counts(current_points)
                    if area_counts:
                        target_area = max(area_counts.keys(), key=lambda x: area_counts[x])
                # Decay step size
                if step_size > 1 and random.random() < 0.2:
                    step_size -= 1
        else:
            # Revert
            used_set.discard(new_pt)
            used_set.add((old_x, old_y))
            current_points[idx] = (old_x, old_y)
            no_improve += 1
        
        # Restart if stuck
        if no_improve > max_no_improve:
            current_points = list(best_points)
            current_score = best_score
            no_improve = 0
            used_set = set(current_points)
            step_size = random.randint(3, 8)
            area_counts = get_area_counts(current_points)
            if area_counts:
                target_area = max(area_counts.keys(), key=lambda x: area_counts[x])
        
        # Occasionally randomize step size
        if random.random() < 0.005:
            step_size = random.randint(1, 6)

    # Final polish: try small moves systematically
    if elapsed() < DEADLINE - 1:
        polish_points = list(best_points)
        polish_score = best_score
        polish_used = set(polish_points)
        
        for _ in range(500):
            if elapsed() > DEADLINE - 0.5:
                break
            idx = random.randint(0, 29)
            old_x, old_y = polish_points[idx]
            best_new = None
            best_new_score = polish_score
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    npt = (nx, ny)
                    if npt in polish_used:
                        continue
                    
                    polish_used.discard((old_x, old_y))
                    polish_used.add(npt)
                    polish_points[idx] = npt
                    s = fast_score(polish_points)
                    
                    if s > best_new_score:
                        best_new_score = s
                        best_new = npt
                    else:
                        polish_used.discard(npt)
                        polish_used.add((old_x, old_y))
                        polish_points[idx] = (old_x, old_y)
            
            if best_new is not None:
                polish_used.discard((old_x, old_y))
                polish_used.add(best_new)
                polish_points[idx] = best_new
                polish_score = best_new_score
                if polish_score > best_score:
                    best_score = polish_score
                    best_points = list(polish_points)
            else:
                # Ensure clean state
                polish_used.discard((old_x, old_y))
                polish_used.add((old_x, old_y))
                polish_points[idx] = (old_x, old_y)

    return best_points
