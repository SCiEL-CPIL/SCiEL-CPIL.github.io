import time
import random
from math import gcd
from itertools import combinations

def construct():
    """
    Returns 12 integer points minimizing the number of ordinary lines.
    Uses simulated annealing with structured seeds and line-extension moves.
    """
    _START = time.time()
    _DEADLINE = _START + 115.0  # Safety margin
    
    def _count_ordinary(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
                if g:
                    A //= g
                    B //= g
                    C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A, B, C = -A, -B, -C
                key = (A, B, C)
                line_counts[key] = line_counts.get(key, 0) + 1
        return sum(1 for c in line_counts.values() if c == 1)

    def _valid(points):
        if len(points) != 12:
            return False
        seen = set()
        for x, y in points:
            if abs(x) > 10**9 or abs(y) > 10**9:
                return False
            t = (x, y)
            if t in seen:
                return False
            seen.add(t)
        # Check not all collinear
        if len(points) >= 3:
            x0, y0 = points[0]
            x1, y1 = points[1]
            x2, y2 = points[2]
            if (x1-x0)*(y2-y0) == (y1-y0)*(x2-x0):
                all_col = True
                for i in range(3, 12):
                    xi, yi = points[i]
                    if (x1-x0)*(yi-y0) != (y1-y0)*(xi-x0):
                        all_col = False
                        break
                if all_col:
                    return False
        return True

    def _make_seed_hesse_like():
        # Seed based on the 8 lines of 4, 4 lines of 3 structure
        # Try to construct points on 4 parallel lines and 4 parallel lines
        # Or use the known integer coordinates for the (8_4, 4_3) configuration if known
        # For now, use a dense grid perturbation
        pts = []
        for x in range(-2, 3):
            for y in range(-2, 3):
                if x*x + y*y <= 4: # 13 points, remove one
                    pts.append([x, y])
        if len(pts) > 12:
            pts.pop()
        return pts

    def _make_seed_boroczky():
        # 6 points on y=0, 6 points on y=1
        # This often yields few ordinary lines
        pts = []
        for x in range(-3, 3):
            pts.append([x, 0])
            pts.append([x, 1])
        return pts

    def _simulated_annealing(points, max_iters=100000):
        cur = [p[:] for p in points]
        cur_score = _count_ordinary(cur)
        best = [p[:] for p in cur]
        best_score = cur_score
        rng = random.Random(42)
        T = 5.0
        T_min = 0.1
        cooling = 0.9999
        iters = 0
        while iters < max_iters and time.time() < _DEADLINE - 5:
            iters += 1
            if T < T_min:
                T = T_min
            idx = rng.randrange(12)
            old = cur[idx]
            cur_set = set(tuple(p) for p in cur)
            
            # Generate candidate
            if rng.random() < 0.7:
                # Small perturbation
                nx = old[0] + rng.randint(-5, 5)
                ny = old[1] + rng.randint(-5, 5)
                cand = (nx, ny)
            else:
                # Line extension
                i = rng.randrange(12)
                j = rng.randrange(12)
                if i == j:
                    i = (j + 1) % 12
                x1, y1 = cur[i]
                x2, y2 = cur[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
                if g:
                    dx //= g
                    dy //= g
                k = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                base = cur[i] if rng.random() < 0.5 else cur[j]
                cand = (base[0] + k * dx, base[1] + k * dy)
            
            if cand in cur_set:
                continue
            if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
                continue
            
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            
            new_score = _count_ordinary(new_pts)
            delta = new_score - cur_score
            
            if delta < 0 or (rng.random() < _exp_neg(delta / T)):
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                if cur_score < best_score:
                    best_score = cur_score
                    best = [p[:] for p in cur]
                    if best_score <= 6:
                        return best, best_score
                
            T *= cooling
        
        return best, best_score

    def _exp_neg(x):
        if x < -50:
            return 0.0
        import math
        return math.exp(-x)

    best_points = None
    best_score = float('inf')
    
    seeds = [
        _make_seed_hesse_like(),
        _make_seed_boroczky(),
        [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], [0,2], [1,2], [2,2], [3,2]],
        [[1,1], [1,-1], [-1,1], [-1,-1], [2,0], [0,2], [-2,0], [0,-2], [3,3], [3,-3], [-3,3], [-3,-3]]
    ]
    
    for seed in seeds:
        if not _valid(seed):
            continue
        pts, score = _simulated_annealing(seed)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            break
        if time.time() > _DEADLINE - 10:
            break
            
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Points: {pts}")
    print(f"Count: {len(pts)}")
    seen = set()
    for p in pts:
        assert tuple(p) not in seen, f"Duplicate: {p}"
        seen.add(tuple(p))
    print("All distinct")
