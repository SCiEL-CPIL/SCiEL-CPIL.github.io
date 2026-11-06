import time
import math
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 135.0

    def elapsed():
        return time.time() - start

    sys.setrecursionlimit(100000)

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

    # ---- Fast exact scorer: largest subset with no 3 collinear ----
    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        # Precompute: for each point, list of (other1, other2) pairs forming a triple
        triples_of = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]; x2, y2 = pts[j]
                dx = x2 - x1; dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples_of[i].append((j, k))
                        triples_of[j].append((i, k))
                        triples_of[k].append((i, j))
        # Precompute bitmask for each point's triples (for fast conflict check)
        # conflict_mask[i] has bits j,k set for each triple (i,j,k)
        # Actually we need: when adding i, check if any pair (j,k) in triples_of[i] both in mask
        # We'll store triples_of[i] as list of (j,k)
        # Order: most constrained first
        order = sorted(range(n), key=lambda i: len(triples_of[i]), reverse=True)
        # Precompute greedy upper bound for remaining points in order
        # greedy_ub[step] = max size of 3-collinear-free subset of points order[step:]
        # Compute greedily: iterate remaining, add point if it doesn't conflict with already-chosen
        # This gives a valid (lower) bound, but we need UPPER bound.
        # Simple upper bound: count of remaining points. Tighter: use the fact that each triple
        # can have at most 2 points. But that's complex. Use: remaining count minus forced exclusions.
        # For speed, use: remaining count. The pruning will be weaker but faster.
        # Actually, let's use a better bound: for each remaining point, if it's in a triple with
        # two already-chosen points, it can't be added. But that depends on current mask.
        # So we compute bound dynamically.

        best = [0]
        deadline = DEADLINE - 8

        def bbb(step, cur_mask, cur_size):
            if elapsed() > deadline:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            # Upper bound: cur_size + (n - step)
            if cur_size + (n - step) <= best[0]:
                return
            idx = order[step]
            # Try include first (often better for finding large sets quickly)
            ok = True
            for a, b in triples_of[idx]:
                if (cur_mask >> a) & 1 and (cur_mask >> b) & 1:
                    ok = False
                    break
            if ok:
                bbb(step + 1, cur_mask | (1 << idx), cur_size + 1)
            # Try exclude
            if cur_size + (n - step - 1) > best[0]:
                bbb(step + 1, cur_mask, cur_size)

        bbb(0, 0, 0)
        return best[0]

    # ---- Build a hypergraph-like configuration: many 3-collinear triples, no 4 collinear ----
    def build_config(seed=0):
        rng = random.Random(seed)
        pts = []
        used = set()
        # We'll add points in groups of 3 along a line, ensuring no 4 collinear.
        # Strategy: pick random directions and offsets.
        for _ in range(200):
            if len(pts) >= 20:
                break
            # Pick a random direction (dx, dy) with small components
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            # Pick a starting point
            sx = rng.randint(-95, 95)
            sy = rng.randint(-95, 95)
            cand = [(sx, sy), (sx + dx, sy + dy), (sx + 2*dx, sy + 2*dy)]
            # Check bounds
            if any(not (-100 <= x <= 100 and -100 <= y <= 100) for x, y in cand):
                continue
            if len(set(cand)) != 3:
                continue
            # Check that adding these doesn't create 4 collinear
            # A point p is "new" if not in used
            new_pts = [p for p in cand if p not in used]
            if not new_pts:
                continue
            trial = pts + new_pts
            if has_4_collinear(trial):
                continue
            pts = trial
            for p in new_pts:
                used.add(p)
        # Fill remaining randomly
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

    # ---- 2-swap local search ----
    def local_search(pts, max_iters=20000):
        cur = [tuple(p) for p in pts]
        cur_score = exact_score(cur)
        best = cur[:]
        best_score = cur_score
        iters = 0
        while iters < max_iters and elapsed() < DEADLINE - 10:
            iters += 1
            # Pick two random indices to swap
            i = rng2.randint(0, 19)
            j = rng2.randint(0, 19)
            if i == j:
                continue
            # Try a small perturbation: move point i to a random nearby point
            old_i = cur[i]
            new_i = (old_i[0] + rng2.randint(-10, 10), old_i[1] + rng2.randint(-10, 10))
            if not (-100 <= new_i[0] <= 100 and -100 <= new_i[1] <= 100):
                continue
            if new_i in cur:
                continue
            trial = cur[:]
            trial[i] = new_i
            if has_4_collinear(trial):
                continue
            s = exact_score(trial)
            if s < cur_score:
                cur = trial
                cur_score = s
                if s < best_score:
                    best_score = s
                    best = cur[:]
            iters += 1
        return best, best_score

    rng2 = random.Random(123)

    # Build several initial candidates
    candidates = []
    for seed in range(30):
        if elapsed() > DEADLINE - 40:
            break
        cfg = build_config(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)

    if not candidates:
        # Fallback: grid-like
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

    # Evaluate all candidates and pick the best
    best_pts = None
    best_val = float('inf')
    for cfg in candidates:
        if elapsed() > DEADLINE - 15:
            break
        s = exact_score(cfg)
        if s < best_val:
            best_val = s
            best_pts = cfg

    if best_pts is None:
        best_pts = candidates[0]
        best_val = exact_score(best_pts)

    # Local search
    if elapsed() < DEADLINE - 20:
        best_pts, best_val = local_search(best_pts, max_iters=50000)

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
