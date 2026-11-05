import time
import random
from math import gcd
from itertools import combinations
import math

def construct():
    """
    Returns 12 integer points minimizing the number of ordinary lines.
    Uses structured line-based construction with targeted local search.
    """
    _START = time.time()
    _DEADLINE = _START + 115.0  # Safety margin before 120s limit
    
    def _count_ordinary(points):
        """Count lines containing exactly 2 points."""
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
    
    def _line_through(p1, p2):
        """Return normalized line (A, B, C) for Ax + By + C = 0 through p1, p2."""
        x1, y1 = p1
        x2, y2 = p2
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
        return (A, B, C)
    
    def _points_on_line(line, points, exclude_idx=None):
        """Return indices of points on the given line."""
        A, B, C = line
        result = []
        for i, (x, y) in enumerate(points):
            if exclude_idx is not None and i == exclude_idx:
                continue
            if A * x + B * y + C == 0:
                result.append(i)
        return result
    
    def _candidate_positions(points, move_idx, rng, limit=30):
        """
        Generate candidate positions for point at move_idx.
        Strategy: 
        1. Points on lines through pairs of other points (extend lines)
        2. Integer points near intersections of such lines
        3. Small random perturbations
        """
        cands = set()
        other_indices = [i for i in range(12) if i != move_idx]
        
        # 1. For each pair of other points, find the line and generate integer points on it
        #    that are close to the current point
        x_cur, y_cur = points[move_idx]
        lines_seen = set()
        
        for j in range(len(other_indices)):
            for k in range(j+1, len(other_indices)):
                i1 = other_indices[j]
                i2 = other_indices[k]
                line = _line_through(points[i1], points[i2])
                if line in lines_seen:
                    continue
                lines_seen.add(line)
                
                A, B, C = line
                # Find integer points on this line near (x_cur, y_cur)
                # Ax + By + C = 0
                if B != 0:
                    # y = -(A*x + C) / B
                    for dx in range(-15, 16):
                        x = x_cur + dx
                        val = -A * x - C
                        if val % B == 0:
                            y = val // B
                            if abs(y - y_cur) <= 15 and abs(x) <= 10**9 and abs(y) <= 10**9:
                                cands.add((x, y))
                elif A != 0:
                    # x = -(B*y + C) / A
                    for dy in range(-15, 16):
                        y = y_cur + dy
                        val = -B * y - C
                        if val % A == 0:
                            x = val // A
                            if abs(x - x_cur) <= 15 and abs(x) <= 10**9 and abs(y) <= 10**9:
                                cands.add((x, y))
                
                if len(cands) > limit * 3:
                    break
            if len(cands) > limit * 3:
                break
        
        # 2. Also try points that complete 3-point lines with two other points
        #    (already covered above, but let's add some random nearby points too)
        for _ in range(10):
            dx = rng.randint(-8, 8)
            dy = rng.randint(-8, 8)
            cands.add((x_cur + dx, y_cur + dy))
        
        return list(cands)
    
    def _local_search(points, max_iters=50000):
        """
        Local search that tries to move each point to reduce ordinary lines.
        Uses line-based candidate generation.
        """
        cur = [tuple(p) for p in points]
        cur_score = _count_ordinary(cur)
        best = tuple(cur)
        best_score = cur_score
        rng = random.Random(12345)
        
        iters = 0
        no_improve = 0
        max_no_improve = 2000
        
        while iters < max_iters and time.time() < _DEADLINE - 5:
            iters += 1
            
            idx = rng.randrange(12)
            old = cur[idx]
            cur_set = set(cur)
            
            cands = _candidate_positions(cur, idx, rng)
            
            # Shuffle candidates for diversity
            rng.shuffle(cands)
            
            found_better = False
            for cand in cands:
                if cand in cur_set:
                    continue
                if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
                    continue
                
                new_pts = list(cur)
                new_pts[idx] = cand
                
                if not _valid(new_pts):
                    continue
                
                new_score = _count_ordinary(new_pts)
                
                if new_score < cur_score:
                    cur = new_pts
                    cur_score = new_score
                    cur_set = set(cur)
                    if cur_score < best_score:
                        best_score = cur_score
                        best = tuple(cur)
                        if best_score <= 6:
                            return [list(p) for p in best], best_score
                    no_improve = 0
                    found_better = True
                    break
                elif new_score == cur_score:
                    # Accept equal moves with some probability to escape local minima
                    if rng.random() < 0.3:
                        cur = new_pts
                        cur_set = set(cur)
                        no_improve += 1
                        break
            
            if not found_better:
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Reset to best and try different random seed
                cur = list(best)
                cur_score = best_score
                no_improve = 0
                rng = random.Random(rng.randrange(100000))
        
        return [list(p) for p in best], best_score
    
    def _make_seed_4x3():
        """4 lines of 3 points: a 4x3 grid."""
        pts = []
        for x in range(4):
            for y in range(3):
                pts.append([x, y])
        return pts
    
    def _make_seed_3x4():
        """3 lines of 4 points: a 3x4 grid."""
        pts = []
        for x in range(3):
            for y in range(4):
                pts.append([x, y])
        return pts
    
    def _make_seed_hexagonal():
        """Hexagonal-like arrangement."""
        pts = [
            [0, 0], [1, 0], [2, 0],
            [-1, 1], [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2],
            [-1, -1], [0, -1]
        ]
        return pts
    
    def _make_seed_parallel():
        """6 points on y=0, 6 points on y=2 with staggered x."""
        pts = []
        for x in range(-3, 3):
            pts.append([x, 0])
            pts.append([x, 2])
        return pts
    
    def _make_seed_diagonal():
        """Points on diagonal lines."""
        pts = []
        # Line y=x: 4 points
        for i in range(4):
            pts.append([i, i])
        # Line y=x+2: 4 points
        for i in range(4):
            pts.append([i, i+2])
        # Line y=x-2: 4 points
        for i in range(4):
            pts.append([i, i-2])
        return pts
    
    def _make_seed_grid_perturbed():
        """Perturbed grid to break symmetry."""
        rng = random.Random(777)
        pts = []
        for x in range(4):
            for y in range(3):
                px = x + rng.randint(-1, 1)
                py = y + rng.randint(-1, 1)
                pts.append([px, py])
        return pts
    
    seeds = [
        _make_seed_4x3(),
        _make_seed_3x4(),
        _make_seed_hexagonal(),
        _make_seed_parallel(),
        _make_seed_diagonal(),
        _make_seed_grid_perturbed(),
        # Additional structured seeds
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
        [[0,0],[2,0],[4,0],[6,0],[0,1],[2,1],[4,1],[6,1],[0,2],[2,2],[4,2],[6,2]],
        [[0,0],[1,1],[2,2],[3,3],[0,1],[1,2],[2,3],[3,4],[0,2],[1,3],[2,4],[3,5]],
    ]
    
    best_points = None
    best_score = float('inf')
    
    for seed in seeds:
        if not _valid(seed):
            continue
        pts, score = _local_search(seed)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            break
        if time.time() > _DEADLINE - 15:
            break
    
    if best_points is None:
        # Fallback
        best_points = _make_seed_4x3()
    
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
    print(f"Ordinary lines: {_count_ordinary(pts)}")
