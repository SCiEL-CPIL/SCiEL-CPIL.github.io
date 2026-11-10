import time
import random
import itertools
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0

    # 1. Fast scoring using NumPy
    def score(pts):
        n = len(pts)
        if n < 3:
            return 0
        arr = np.array(pts, dtype=np.int64)
        counts = {}
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                xk, yk = rest[:, 0], rest[:, 1]
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        v_int = int(v)
                        counts[v_int] = counts.get(v_int, 0) + int(c)
        if not counts:
            return 0
        return max(counts.values())

    # 2. Candidate Generation - diverse patterns
    candidates = []

    # Regular grids
    for w in [2, 3, 5, 6, 10, 15]:
        h = 30 // w
        if h < 2:
            continue
        for dx in [1, 2, 3, 4]:
            for dy in [1, 2, 3, 4]:
                if dx * (w - 1) > 60 or dy * (h - 1) > 60:
                    continue
                for ox in range(-5, 6, 5):
                    for oy in range(-5, 6, 5):
                        pts = []
                        valid = True
                        for i in range(w):
                            for j in range(h):
                                x = ox + i * dx
                                y = oy + j * dy
                                if -30 <= x <= 30 and -30 <= y <= 30:
                                    pts.append((x, y))
                                else:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(pts) == 30:
                            candidates.append(pts)

    # Diagonal line patterns
    for slope in [1, 2, 3]:
        for intercept in range(-30, 31, 5):
            pts = []
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)

    # Parabolic patterns
    for a in [1, 2, 3]:
        for c in range(-20, 21, 5):
            pts = []
            for x in range(-30, 31):
                y = a * x * x // 10 + c
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)

    # Random structured patterns
    random.seed(42)
    for _ in range(20):
        base_w = random.choice([3, 5, 6, 10])
        base_h = 30 // base_w
        dx = random.choice([1, 2, 3])
        dy = random.choice([1, 2, 3])
        ox = random.randint(-10, 10)
        oy = random.randint(-10, 10)
        
        pts = []
        for i in range(base_w):
            for j in range(base_h):
                x = ox + i * dx + random.randint(-1, 1)
                y = oy + j * dy + random.randint(-1, 1)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        
        pts = list(set(pts))
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        
        if len(pts) == 30:
            candidates.append(pts)

    # Pure random candidates
    for _ in range(10):
        random_pts = set()
        while len(random_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            random_pts.add((x, y))
        candidates.append(list(random_pts))

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # 3. Initial Evaluation
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > total_deadline - 30:
            break
        s = score(c)
        scored_candidates.append((s, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    best_overall_score = scored_candidates[0][0] if scored_candidates else 0
    best_overall_pts = scored_candidates[0][1] if scored_candidates else []

    # 4. Local Search with simulated annealing
    def local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration
        
        T = 5.0
        T_min = 0.5
        cooling_rate = 0.9995
        
        while time.time() < end_time:
            i = random.randint(0, 29)
            old_p = current_pts[i]
            
            r = random.random()
            if r < 0.7:
                nx = old_p[0] + random.randint(-2, 2)
                ny = old_p[1] + random.randint(-2, 2)
            elif r < 0.9:
                nx = old_p[0] + random.randint(-5, 5)
                ny = old_p[1] + random.randint(-5, 5)
            else:
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
            
            new_p = (nx, ny)
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                if new_p != old_p and new_p not in current_set:
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    
                    new_score = score(current_pts)
                    delta = new_score - current_score
                    
                    if delta > 0:
                        current_score = new_score
                    else:
                        if random.random() < np.exp(delta / T):
                            current_score = new_score
                        else:
                            current_pts[i] = old_p
                            current_set.remove(new_p)
                            current_set.add(old_p)
                    
                    T = max(T_min, T * cooling_rate)

        return current_score, current_pts

    # Run searches
    search_times = [20, 15, 10]
    for idx in range(min(3, len(scored_candidates))):
        if time.time() > total_deadline - 5:
            break
        
        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 2
        if rem_time <= 0:
            break
        
        t = min(search_times[idx], rem_time)
        new_s, new_p = local_search(p, t)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p

    return best_overall_pts
