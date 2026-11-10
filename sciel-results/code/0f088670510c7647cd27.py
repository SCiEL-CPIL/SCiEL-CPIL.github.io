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

    # 2. Incremental delta scoring: O(N^2) instead of O(N^3)
    def compute_all_areas(arr):
        """Compute all doubled areas for all triples, return dict area -> count"""
        n = len(arr)
        counts = Counter()
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
                        counts[int(v)] += int(c)
        return counts

    def delta_score(pts, old_pt, new_pt, idx):
        """Compute change in best multiplicity when replacing pts[idx] from old_pt to new_pt.
        Returns (delta, new_counts) where delta = new_best - old_best.
        Uses O(N^2) computation."""
        n = len(pts)
        arr = np.array(pts, dtype=np.int64)
        
        # Compute all areas involving the old point at index idx
        old_areas = []
        new_areas = []
        
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if i == idx or j == idx:
                    continue
                # Triangle (i, idx, j)
                xi, yi = arr[i]
                xj, yj = arr[j]
                old_x, old_y = old_pt
                new_x, new_y = new_pt
                old_area = abs(int(xi) * (old_y - yj) + int(old_x) * (yj - yi) + int(xj) * (yi - old_y))
                new_area = abs(int(xi) * (new_y - yj) + int(new_x) * (yj - yi) + int(xj) * (yi - new_y))
                old_areas.append(old_area)
                new_areas.append(new_area)
        
        # Also need triangles where idx is the first or second point
        # Actually the above loop covers all pairs (i,j) with i<j, i!=idx, j!=idx
        # But we also need triangles where one of i or j IS idx... wait no.
        # The triple is (i, idx, j) for all pairs (i,j) where i,j != idx and i < j.
        # That's correct.
        
        # Build old counts and new counts
        old_counts = Counter()
        for a in old_areas:
            if a > 0:
                old_counts[a] += 1
        
        new_counts = Counter()
        for a in new_areas:
            if a > 0:
                new_counts[a] += 1
        
        # Compute full counts for old configuration
        full_old_counts = compute_all_areas(arr)
        
        # For new configuration, we need to compute all areas
        # But we can be smart: only areas involving idx change
        # Areas NOT involving idx are the same
        
        # Let's just compute the full new score efficiently
        # The areas not involving idx are unchanged
        # The areas involving idx are in old_areas/new_areas
        
        # Full old score
        old_best = max(full_old_counts.values()) if full_old_counts else 0
        
        # For new score, we need counts for all triples
        # Triples not involving idx: same as old
        # Triples involving idx: use new_areas
        
        # Build new full counts
        new_full_counts = Counter()
        # Add areas not involving idx
        for i in range(n - 2):
            if i == idx:
                continue
            for j in range(i + 1, n - 1):
                if j == idx:
                    continue
                xk_idx = j if j > idx else j  # need to handle indices carefully
                # Actually let me just recompute properly
                pass
        
        # Simpler approach: compute new full counts from scratch but only for the changed part
        # Actually, let me just compute the delta directly
        
        # Old best multiplicity
        old_best = max(full_old_counts.values()) if full_old_counts else 0
        
        # New best multiplicity: for each area value, new_count = old_count_not_involving_idx + new_areas_count
        # But we need to know old_count_not_involving_idx for each area
        
        # Let me take a different approach: compute old full counts, then adjust
        # For each area a:
        #   old_total[a] = old_not_idx[a] + old_idx[a]
        #   new_total[a] = old_not_idx[a] + new_idx[a]
        # where old_idx[a] = count of a in old_areas, new_idx[a] = count of a in new_areas
        
        # old_not_idx[a] = old_total[a] - old_idx[a]
        
        new_full = Counter()
        for a, cnt in full_old_counts.items():
            new_full[a] = cnt - old_counts.get(a, 0) + new_counts.get(a, 0)
        
        # Add any areas that only appear in new_areas but not in old
        for a, cnt in new_counts.items():
            if a not in full_old_counts:
                new_full[a] = cnt
            # If a is in full_old_counts, it was already handled above
        
        # Clean up zeros
        new_full = Counter({a: c for a, c in new_full.items() if c > 0})
        
        new_best = max(new_full.values()) if new_full else 0
        delta = new_best - old_best
        return delta, new_best

    # 3. Candidate Generation
    candidates = []

    # Regular grids with various parameters
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

    # Hexagonal lattice patterns
    for dx in [2, 3, 4]:
        for dy in [2, 3, 4]:
            for ox in range(-10, 11, 5):
                for oy in range(-10, 11, 5):
                    pts = []
                    for i in range(6):
                        for j in range(5):
                            x = ox + i * dx + (j % 2) * (dx // 2)
                            y = oy + j * dy
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                pts.append((x, y))
                    if len(pts) == 30:
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

    # 4. Initial Evaluation
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > total_deadline - 30:
            break
        s = score(c)
        scored_candidates.append((s, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    best_overall_score = scored_candidates[0][0] if scored_candidates else 0
    best_overall_pts = scored_candidates[0][1] if scored_candidates else []

    # 5. Fast local search with incremental delta scoring
    def fast_local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_arr = np.array(current_pts, dtype=np.int64)
        current_counts = compute_all_areas(current_arr)
        current_best = max(current_counts.values()) if current_counts else 0
        current_set = set(current_pts)
        end_time = time.time() + duration
        
        T = 3.0
        T_min = 0.1
        cooling_rate = 0.9998
        iterations = 0
        
        while time.time() < end_time:
            iterations += 1
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
                    # Apply move tentatively
                    current_pts[i] = new_p
                    current_arr[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    
                    # Compute delta using incremental method
                    # Areas involving index i change
                    delta, new_best = delta_score(current_pts, old_p, new_p, i)
                    
                    if delta > 0:
                        current_best = new_best
                    else:
                        if random.random() < np.exp(delta / T):
                            current_best = new_best
                        else:
                            # Revert
                            current_pts[i] = old_p
                            current_arr[i] = old_p
                            current_set.remove(new_p)
                            current_set.add(old_p)
                    
                    T = max(T_min, T * cooling_rate)
                    
                    if current_best > best_overall_score:
                        best_overall_score = current_best
                        best_overall_pts = list(current_pts)
        
        return current_best, current_pts, iterations

    # Run multiple searches
    search_configs = [
        (0, 20),
        (1, 15),
        (2, 10),
    ]

    for idx, t in search_configs:
        if time.time() > total_deadline - 3:
            break
        if idx >= len(scored_candidates):
            break
        
        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 2
        if rem_time <= 0:
            break
        
        t = min(t, rem_time)
        new_s, new_p, iters = fast_local_search(p, t)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p

    return best_overall_pts
