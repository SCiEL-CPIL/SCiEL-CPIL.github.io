import time
import math
import random
import sys
from collections import defaultdict

def construct():
    random.seed(42)
    start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A; B = -B; C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                cnt = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        cnt += 1
                if cnt >= 4:
                    return True
        return False

    def build_conflict_bitmasks(pts):
        """For each point i, conflict_mask[i] is a bitmask of points that form a triple with i.
        Actually, we need: for point i, the set of (j,k) pairs where i,j,k are collinear.
        For B&B: when adding point i, check if any two already-selected points form a triple with i.
        We'll precompute: for each i, a list of (j,k) pairs.
        But for speed, we use: conflict[i] = bitmask of all points that are in some triple with i.
        Then when adding i: if (selected & conflict[i]) has >= 2 bits that are paired... 
        That's not quite right. Let's do it properly:
        
        For each point i, store a list of frozensets {j,k} such that i,j,k collinear.
        When adding i to selected set S: for each {j,k} in triples[i], if j in S and k in S, conflict.
        
        For bitset optimization: 
        conflict_mask[i] = OR of (1<<j | 1<<k) for all {j,k} in triples_with_i
        But that's not sufficient for the check. We need to verify that no pair {j,k} is BOTH in S.
        
        Alternative: For each i, precompute a list of masks: mask_pair = (1<<j)|(1<<k)
        When adding i: for each mask_pair in pair_masks[i], if (S & mask_pair) == mask_pair, conflict.
        """
        n = len(pts)
        pair_masks = [[] for _ in range(n)]  # pair_masks[i] = list of (1<<j)|(1<<k)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        m = (1 << i) | (1 << j) | (1 << k)
                        pair_masks[i].append((1 << j) | (1 << k))
                        pair_masks[j].append((1 << i) | (1 << k))
                        pair_masks[k].append((1 << i) | (1 << j))
        return pair_masks

    def exact_score_fast(pts, time_budget=5.0):
        """Bitset-optimized B&B for max no-three-collinear subset."""
        n = len(pts)
        if n < 3:
            return n
        t0 = time.time()
        pair_masks = build_conflict_bitmasks(pts)
        best = [0]
        
        # Order: points in more triples first
        order = sorted(range(n), key=lambda i: len(pair_masks[i]), reverse=True)
        
        sys.setrecursionlimit(10000)
        
        def bbb(step, cur_mask, cur_size):
            nonlocal best
            if time.time() - t0 > time_budget:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            # prune
            if cur_size + (n - step) <= best[0]:
                return
            idx = order[step]
            # Option exclude
            bbb(step + 1, cur_mask, cur_size)
            # Option include: check conflict
            ok = True
            for pm in pair_masks[idx]:
                if (cur_mask & pm) == pm:
                    ok = False
                    break
            if ok:
                bbb(step + 1, cur_mask | (1 << idx), cur_size + 1)

        bbb(0, 0, 0)
        return best[0]

    def count_triples(pts):
        """Count number of collinear triples."""
        n = len(pts)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        cnt += 1
        return cnt

    def build_dense_config(seed=0):
        """Build 20 points with many 3-collinear triples, no 4 collinear."""
        rng = random.Random(seed)
        pts = []
        used = set()

        directions = [
            (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2),
            (3, 1), (1, 3), (3, -1), (-3, 1), (3, 2), (2, 3), (3, -2), (2, -3),
            (4, 1), (1, 4), (4, -1), (-4, 1), (4, 2), (2, 4), (4, -2), (2, -4),
            (5, 1), (1, 5), (5, -1), (-5, 1), (5, 2), (2, 5), (5, -2), (2, -5),
        ]
        rng.shuffle(directions)

        attempts = 0
        while len(pts) < 20 and attempts < 3000:
            attempts += 1
            d = rng.choice(directions)
            for _ in range(50):
                sx = rng.randint(-95, 95)
                sy = rng.randint(-95, 95)
                cand = [(sx, sy), (sx + d[0], sy + d[1]), (sx + 2*d[0], sy + 2*d[1])]
                if any(not (-100 <= x <= 100 and -100 <= y <= 100) for x, y in cand):
                    continue
                if len(set(cand)) != 3:
                    continue
                new_pts = [p for p in cand if p not in used]
                if not new_pts:
                    continue
                trial = pts + new_pts
                if has_4_collinear(trial):
                    continue
                pts = trial
                for p in new_pts:
                    used.add(p)
                if len(pts) >= 20:
                    break

        if len(pts) < 20:
            while len(pts) < 20:
                p = (rng.randint(-100, 100), rng.randint(-100, 100))
                if p in used:
                    continue
                trial = pts + [p]
                if has_4_collinear(trial):
                    continue
                pts.append(p)
                used.add(p)
        return pts[:20]

    # Build initial candidates
    candidates = []
    for seed in range(15):
        if elapsed() > DEADLINE - 60:
            break
        cfg = build_dense_config(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)

    if not candidates:
        pts = []
        used = set()
        rng = random.Random(7)
        while len(pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p in used:
                continue
            if has_4_collinear(pts + [p]):
                continue
            pts.append(p)
            used.add(p)
        candidates = [pts]

    # Evaluate and pick best
    best_pts = None
    best_val = float('inf')
    for cfg in candidates:
        if elapsed() > DEADLINE - 20:
            break
        s = exact_score_fast(cfg, time_budget=3.0)
        if s < best_val:
            best_val = s
            best_pts = cfg

    if best_pts is None:
        best_pts = candidates[0]
        best_val = exact_score_fast(best_pts, time_budget=3.0)

    # Simulated Annealing to minimize score
    cur_pts = [tuple(p) for p in best_pts]
    cur_val = best_val
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    rng_sa = random.Random(123)
    
    while elapsed() < DEADLINE - 15 and T > T_min:
        T *= cooling
        idx = rng_sa.randint(0, 19)
        old = cur_pts[idx]
        if rng_sa.random() < 0.5:
            np = (old[0] + rng_sa.randint(-10, 10), old[1] + rng_sa.randint(-10, 10))
        else:
            np = (rng_sa.randint(-100, 100), rng_sa.randint(-100, 100))
        if not (-100 <= np[0] <= 100 and -100 <= np[1] <= 100):
            continue
        if np in cur_pts:
            continue
        trial = cur_pts[:]
        trial[idx] = np
        if has_4_collinear(trial):
            continue
        s = exact_score_fast(trial, time_budget=2.0)
        delta = s - cur_val
        if delta <= 0 or rng_sa.random() < math.exp(-delta / T):
            cur_pts = trial
            if s < cur_val:
                cur_val = s
                best_val = s
                best_pts = cur_pts[:]

    # Final exact evaluation
    final_score = exact_score_fast([tuple(p) for p in best_pts], time_budget=10.0)
    if final_score < best_val:
        best_val = final_score

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
