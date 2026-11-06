import time
import math
import random
import sys

def construct():
    random.seed(12345)
    start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
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

    def precompute_triples(pts):
        n = len(pts)
        triples_with_idx = [[] for _ in range(n)]
        triple_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
                        triple_set.add((i, j, k))
        return triples_with_idx, triple_set

    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        triples_with_idx, _ = precompute_triples(pts)
        best = [0]

        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)

        def bbb(step, cur_mask, cur_size):
            if elapsed() > DEADLINE - 3:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            if cur_size + (n - step) <= best[0]:
                return
            idx = order[step]
            # Option exclude
            bbb(step + 1, cur_mask, cur_size)
            # Option include: check conflict
            ok = True
            for a, b in triples_with_idx[idx]:
                if (cur_mask & (1 << a)) and (cur_mask & (1 << b)):
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

    def build_structured_config():
        """
        Build 20 points using a structured approach:
        Use a small grid with perturbation to create many 3-collinear lines.
        
        Strategy: Start with points on a 5x4 grid, then perturb slightly
        to break 4-collinear while preserving 3-collinear patterns.
        """
        # Base grid: 5 columns x 4 rows
        base_pts = []
        for i in range(5):
            for j in range(4):
                base_pts.append((i * 20, j * 20))
        
        # This gives 20 points. Check for 4-collinear.
        # In a regular grid, there will be many 4-collinear lines.
        # We need to perturb to break 4-collinear but keep 3-collinear.
        
        # Perturb each point by small random amounts
        rng = random.Random(42)
        pts = []
        used = set()
        
        for i in range(5):
            for j in range(4):
                base_x = i * 20
                base_y = j * 20
                # Small perturbation
                px = base_x + rng.randint(-3, 3)
                py = base_y + rng.randint(-3, 3)
                # Ensure within bounds and distinct
                if not (-100 <= px <= 100 and -100 <= py <= 100):
                    continue
                if (px, py) in used:
                    continue
                trial = pts + [(px, py)]
                if has_4_collinear(trial):
                    # Try different perturbation
                    for _ in range(10):
                        px = base_x + rng.randint(-5, 5)
                        py = base_y + rng.randint(-5, 5)
                        if not (-100 <= px <= 100 and -100 <= py <= 100):
                            continue
                        if (px, py) in used:
                            continue
                        trial = pts + [(px, py)]
                        if not has_4_collinear(trial):
                            break
                    else:
                        continue
                pts.append((px, py))
                used.add((px, py))
        
        if len(pts) < 20:
            # Fill remaining
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

    def build_dense_lines_config(seed=0):
        """
        Build 20 points by explicitly creating many 3-point lines.
        Use a set of lines, each with exactly 3 points, sharing endpoints.
        """
        rng = random.Random(seed)
        pts = []
        used = set()
        
        # Define several line segments (directions)
        directions = [
            (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2),
            (2, -1), (1, -2), (3, 1), (1, 3), (3, -1), (-3, 1),
            (3, 2), (2, 3), (3, -2), (2, -3), (4, 1), (1, 4),
            (4, -1), (-4, 1)
        ]
        rng.shuffle(directions)
        
        attempts = 0
        while len(pts) < 20 and attempts < 5000:
            attempts += 1
            d = rng.choice(directions)
            for _ in range(50):
                sx = rng.randint(-80, 80)
                sy = rng.randint(-80, 80)
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

    # Build multiple candidate configurations
    candidates = []
    
    # Structured grid-based
    if elapsed() < DEADLINE - 60:
        cfg = build_structured_config()
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)
    
    # Dense lines with multiple seeds
    for seed in range(8):
        if elapsed() > DEADLINE - 50:
            break
        cfg = build_dense_lines_config(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)
    
    if not candidates:
        # Fallback: random
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

    # Evaluate candidates and pick best
    best_pts = None
    best_val = float('inf')
    for cfg in candidates:
        if elapsed() > DEADLINE - 10:
            break
        s = exact_score(cfg)
        if s < best_val:
            best_val = s
            best_pts = cfg

    if best_pts is None:
        best_pts = candidates[0]
        best_val = exact_score(best_pts)

    # Local search to improve
    cur_pts = [tuple(p) for p in best_pts]
    cur_val = best_val
    iters = 0
    while elapsed() < DEADLINE - 5 and iters < 300:
        iters += 1
        idx = random.randint(0, 19)
        old = cur_pts[idx]
        if random.random() < 0.5:
            np = (old[0] + random.randint(-5, 5), old[1] + random.randint(-5, 5))
        else:
            np = (random.randint(-100, 100), random.randint(-100, 100))
        if not (-100 <= np[0] <= 100 and -100 <= np[1] <= 100):
            continue
        if np in cur_pts:
            continue
        trial = cur_pts[:]
        trial[idx] = np
        if has_4_collinear(trial):
            continue
        s = exact_score(trial)
        if s < cur_val:
            cur_val = s
            cur_pts = trial
            best_val = s
            best_pts = cur_pts
        elif s == cur_val and random.random() < 0.1 and elapsed() < DEADLINE - 15:
            cur_pts = trial
            best_pts = cur_pts

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
