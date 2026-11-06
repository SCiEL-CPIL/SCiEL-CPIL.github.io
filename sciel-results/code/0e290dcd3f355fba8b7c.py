"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations, permutations
from fractions import Fraction
import random

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 145
    best_pts = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Try known optimal configurations for 12 points.
    # The theoretical minimum is 6.
    # Known configurations with 6 ordinary lines for 12 points:
    # - A 2x6 grid? Let's check.
    #   Points: (0,0)..(0,5) and (1,0)..(1,5).
    #   Lines:
    #   - Vertical: 2 lines with 6 pts.
    #   - Horizontal: 6 lines with 2 pts. (These are ordinary!)
    #   - Diagonals:
    #     Slope 1: (0,0)-(1,1), (0,1)-(1,2), etc. 
    #     In a 2xN grid, diagonals of slope 1 have length 2.
    #     So all "diagonal" lines are ordinary.
    #     There are many.
    #     Let's count for 2x6:
    #     Pairs: 66.
    #     Vertical lines: 2 (size 6). Pairs: 2 * 15 = 30.
    #     Horizontal lines: 6 (size 2). Pairs: 6 * 1 = 6.
    #     Diagonals (slope 1): (0,0)-(1,1), (0,1)-(1,2), (0,2)-(1,3), (0,3)-(1,4), (0,4)-(1,5). 5 lines (size 2). Pairs: 5.
    #     Diagonals (slope -1): (0,1)-(1,0), (0,2)-(1,1), (0,3)-(1,2), (0,4)-(1,3), (0,5)-(1,4). 5 lines (size 2). Pairs: 5.
    #     Other slopes?
    #     (0,0) to (1,2) slope 2. (0,1) to (1,3) slope 2.
    #     (0,0)-(1,2) is a line. Does it contain other points? No, only 2 cols.
    #     So any line connecting (0,i) and (1,j) with i!=j is a line of size 2, UNLESS it aligns with others.
    #     Wait, in a 2xN grid, any line passing through a point in col 0 and a point in col 1 is uniquely determined by those two points.
    #     So ALL cross-column pairs are ordinary lines?
    #     Total cross-column pairs: 6*6 = 36.
    #     We identified some as "diagonals" (slope +/- 1).
    #     But (0,0) and (1,2) define a line. Are there other points on it?
    #     No, because the grid is only 2 wide.
    #     So ALL 36 cross-column pairs are ordinary lines?
    #     Plus the 6 horizontal lines? No, horizontal lines are within columns? No, horizontal lines connect (0,i) and (1,i).
    #     Wait, (0,i) and (1,i) are the same row.
    #     So horizontal lines are a subset of cross-column pairs.
    #     So total ordinary lines = 36 (all cross-column pairs are size 2 lines).
    #     Score 36. Very bad.
    
    # Let's try a 3x4 grid.
    # Points: 12.
    # Lines:
    # - Vertical: 3 lines of size 4. Pairs: 3 * 6 = 18.
    # - Horizontal: 4 lines of size 3. Pairs: 4 * 3 = 12.
    # - Diagonals:
    #   Slope 1:
    #   (0,0)-(1,1)-(2,2) -> size 3. Pairs 3.
    #   (0,1)-(1,2) -> size 2.
    #   (1,0)-(2,1) -> size 2.
    #   (0,2)-(1,3) -> size 2.
    #   (1,1)-(2,2) is part of the first one.
    #   Let's list all lines in 3x4 grid carefully.
    #   This is a known configuration.
    #   Let's just calculate it.
    
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    score_3x4 = _count_ordinary(grid_3x4)
    if score_3x4 < best_score:
        best_score = score_3x4
        best_pts = grid_3x4
    
    # Try 4x3 grid (same set of points, just rotated, score should be same)
    # Try 2x6 grid
    grid_2x6 = [[x, y] for x in range(2) for y in range(6)]
    score_2x6 = _count_ordinary(grid_2x6)
    if score_2x6 < best_score:
        best_score = score_2x6
        best_pts = grid_2x6
        
    # Try 6x2 grid
    grid_6x2 = [[x, y] for x in range(6) for y in range(2)]
    score_6x2 = _count_ordinary(grid_6x2)
    if score_6x2 < best_score:
        best_score = score_6x2
        best_pts = grid_6x2
        
    # Try 1x12 (invalid, all collinear)
    
    # Is there a better configuration?
    # The "Hessian" configuration is 9 points, 4 lines of 3, 12 lines of 2.
    # For 12 points, the minimum is 6.
    # One known configuration with 6 ordinary lines is derived from the affine plane of order 3?
    # Affine plane of order 3 has 9 points.
    # Maybe add 3 points?
    
    # Let's try a structured search for a 12-point set with 6 ordinary lines.
    # We can use the fact that the lines must cover all 66 pairs.
    # Let k be the number of lines with size >= 3.
    # Let o be the number of ordinary lines (size 2).
    # Total pairs = 66.
    # Sum over lines L of C(|L|, 2) = 66.
    # We want to minimize o.
    # If o=6, then the other lines must cover 66 - 6 = 60 pairs.
    # Possible line sizes: 3, 4, 5, ...
    # C(3,2)=3, C(4,2)=6, C(5,2)=10, C(6,2)=15, C(7,2)=21...
    
    # A common optimal structure for n=12 is:
    # 4 lines of size 4? 4*6 = 24 pairs.
    # 2 lines of size 5? 2*10 = 20 pairs.
    # 2 lines of size 4? 2*6 = 12 pairs.
    # 2 lines of size 3? 2*3 = 6 pairs.
    # Total pairs from non-ordinary lines: 24+20+12+6 = 62.
    # Remaining pairs: 66 - 62 = 4.
    # So 4 ordinary lines?
    # But the minimum is 6. So this combination doesn't work or I miscalculated.
    
    # Let's check the known result: "The minimum number of ordinary lines for 12 points is 6."
    # Example configuration:
    # Take a 3x3 grid (9 points).
    # Add 3 points at infinity? No, integer coords.
    
    # Let's try a local search starting from the best grid found so far.
    # But the previous local search was weak.
    # Let's try a "Simulated Annealing" or "Random Restart" local search.
    
    if best_score > 6:
        # Local search with random restarts
        random.seed(42)
        
        def fast_count(pts):
            n = len(pts)
            line_map = {}
            for i in range(n):
                x1, y1 = pts[i]
                for j in range(i + 1, n):
                    x2, y2 = pts[j]
                    a = y2 - y1
                    b = x1 - x2
                    c = x2 * y1 - x1 * y2
                    g = gcd(gcd(abs(a), abs(b)), abs(c))
                    if g > 0:
                        a //= g
                        b //= g
                        c //= g
                    if a < 0 or (a == 0 and b < 0):
                        a, b, c = -a, -b, -c
                    key = (a, b, c)
                    if key not in line_map:
                        line_map[key] = 0
                    line_map[key] += 1
            return sum(1 for v in line_map.values() if v == 1)
            
        def local_search(pts, max_iters=1000, time_limit=10):
            current_pts = [list(p) for p in pts]
            current_score = fast_count(current_pts)
            best_local = [list(p) for p in current_pts]
            best_local_score = current_score
            start_time = time.time()
            
            for _ in range(max_iters):
                if time.time() > time_limit + start_time:
                    break
                # Pick a random point to move
                i = random.randint(0, 11)
                orig_x, orig_y = current_pts[i]
                
                # Move to a random nearby point
                # Or try to move to a point that creates a new collinearity?
                # Heuristic: Pick a random other point j, and a random third point k.
                # Try to place i such that i, j, k are collinear?
                # Or just random small move.
                
                # Let's try random small moves first, then targeted moves.
                if random.random() < 0.5:
                    dx = random.randint(-20, 20)
                    dy = random.randint(-20, 20)
                else:
                    # Targeted: pick two other points, try to align
                    j, k = random.sample(range(12), 2)
                    if j == i or k == i:
                        continue
                    xj, yj = current_pts[j]
                    xk, yk = current_pts[k]
                    # Line through j and k
                    # We want i to be on this line.
                    # i = j + t * (k - j)
                    # Try integer t
                    dx = xk - xj
                    dy = yk - yj
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    dx //= g
                    dy //= g
                    # Try t in range
                    for t in range(-10, 11):
                        if t == 0:
                            continue
                        new_x = xj + t * dx
                        new_y = yj + t * dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check distinctness
                        valid = True
                        for m in range(12):
                            if m == i:
                                continue
                            if current_pts[m][0] == new_x and current_pts[m][1] == new_y:
                                valid = False
                                break
                        if not valid:
                            continue
                        current_pts[i] = [new_x, new_y]
                        new_score = fast_count(current_pts)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_local_score:
                                best_local_score = new_score
                                best_local = [list(p) for p in current_pts]
                        else:
                            current_pts[i] = [orig_x, orig_y]
                        break # Only try one t per iteration for speed? Or try a few.
                    
                # If random move
                if random.random() < 0.5:
                    dx = random.randint(-20, 20)
                    dy = random.randint(-20, 20)
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    valid = True
                    for m in range(12):
                        if m == i:
                            continue
                        if current_pts[m][0] == new_x and current_pts[m][1] == new_y:
                            valid = False
                            break
                    if not valid:
                        continue
                    current_pts[i] = [new_x, new_y]
                    new_score = fast_count(current_pts)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local = [list(p) for p in current_pts]
                    else:
                        current_pts[i] = [orig_x, orig_y]
                        
            return best_local, best_local_score
        
        # Run several restarts from different grids
        grids = [
            [[x, y] for x in range(3) for y in range(4)],
            [[x, y] for x in range(4) for y in range(3)],
            [[x, y] for x in range(2) for y in range(6)],
            [[x, y] for x in range(6) for y in range(2)],
            [[x, y] for x in range(5) for y in range(3)][:12], # 5x3 has 15, take 12
            [[x, y] for x in range(3) for y in range(5)][:12],
        ]
        
        for g in grids:
            if time.time() > deadline - 10:
                break
            # Randomly permute points to break symmetry?
            # No, the set is the same.
            # But local search depends on the order? No, it's a set.
            # So running local search on the same set is redundant unless we add noise.
            
            # Add noise
            noisy_pts = [list(p) for p in g]
            for i in range(12):
                noisy_pts[i][0] += random.randint(-5, 5)
                noisy_pts[i][1] += random.randint(-5, 5)
            
            res_pts, res_score = local_search(noisy_pts, max_iters=500, time_limit=5)
            if res_score < best_score:
                best_score = res_score
                best_pts = res_pts
                if best_score == 6:
                    return [list(p) for p in best_pts]
                    
    # Final check
    if best_pts is None:
        best_pts = [[0,0], [1,0], [2,0], [3,0], [4,0], [5,0], [0,1], [1,1], [2,1], [3,1], [4,1], [5,1]]
        best_score = _count_ordinary(best_pts)
        
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
