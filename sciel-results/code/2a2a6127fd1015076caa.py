"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _canonical_line_key(x1, y1, x2, y2):
    """Canonical (a,b,c) for line through two points: a*x + b*y = c, gcd=1, a>0 or (a==0,b>0)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    a = dy // g
    b = -dx // g
    c = a * x1 + b * y1
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _build_line_map(points):
    """Build a map from line_key to set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            key = _canonical_line_key(x1, y1, x2, y2)
            if key:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Calculate score and check validity from line map."""
    score = 0
    for key, pts in line_map.items():
        cnt = len(pts)
        if cnt >= 5:
            return -1, False
        if cnt == 4:
            score += 1
    return score, True

def construct():
    deadline = time.time() + 145
    rng = random.Random(99)
    
    # Initial configuration: 4x4 grid
    # The 4x4 grid has 4 rows, 4 cols, 2 diagonals = 10 lines of 4 points.
    # Wait, let's check 4x4 grid [0..3]x[0..3].
    # Rows: y=0,1,2,3 (4 lines)
    # Cols: x=0,1,2,3 (4 lines)
    # Diagonals: x-y=0, x+y=3, x-y=-1, x-y=1, x-y=-2, x-y=2, x+y=1, x+y=2, x+y=4, x+y=5, x+y=6, x+y=7?
    # Let's just trust the seed score of 1.2 which corresponds to 12 lines?
    # Seed score 1.2 means value/seed = 1.2. If seed is 10, value is 12.
    # Let's start with the 4x4 grid and improve.
    
    base = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in base]
    best_score, _ = _score_from_line_map(_build_line_map(base))
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        elif rng.random() < 0.3:
            current = [p[:] for p in best_points]
        else:
            # Perturb best points
            current = [p[:] for p in best_points]
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, 15)
                new_x = current[idx][0] + rng.randint(-3, 3)
                new_y = current[idx][1] + rng.randint(-3, 3)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                    if not any([new_x, new_y] == p for p in current if p is not current[idx]):
                        current[idx] = [new_x, new_y]
        
        line_map = _build_line_map(current)
        current_score, valid = _score_from_line_map(line_map)
        if not valid:
            continue
        
        T = 0.5
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            idx = rng.randint(0, 15)
            old_pt = current[idx][:]
            
            # Heuristic: Try to complete a line
            # Find lines with 3 points that do NOT include idx
            # If we move idx to a point on such a line, it becomes 4.
            # But we must ensure it doesn't create a 5-point line elsewhere.
            
            new_pt = None
            
            # 1. Try line completion
            # Identify candidate lines (3 points)
            # This is expensive to do from scratch every time. 
            # Approximation: Check lines formed by pairs of other points?
            # Or just use random moves but with a bias?
            
            # Let's stick to random moves but with a larger step size sometimes
            # and rely on incremental scoring to be fast.
            
            step_size = rng.randint(-4, 4)
            if rng.random() < 0.1:
                step_size = rng.randint(-10, 10)
                
            new_x = old_pt[0] + step_size
            new_y = old_pt[1] + rng.randint(-4, 4)
            
            # If we are stuck, try a larger jump
            if no_improve > 500:
                new_x = old_pt[0] + rng.randint(-10, 10)
                new_y = old_pt[1] + rng.randint(-10, 10)

            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if [new_x, new_y] == old_pt:
                continue
            if any([new_x, new_y] == p for p in current if p is not current[idx]):
                continue
            
            # Incremental Scoring
            # 1. Remove old point from lines
            old_lines = []
            for key, pts in line_map.items():
                if idx in pts:
                    old_lines.append((key, len(pts)))
                    pts.remove(idx)
                    if len(pts) == 0:
                        del line_map[key]
            
            # 2. Add new point to lines
            current[idx] = [new_x, new_y]
            new_lines = []
            for j in range(16):
                if j == idx:
                    continue
                key = _canonical_line_key(new_x, new_y, current[j][0], current[j][1])
                if key:
                    if key not in line_map:
                        line_map[key] = set()
                    line_map[key].add(idx)
                    line_map[key].add(j)
                    new_lines.append((key, len(line_map[key])))
            
            # 3. Check validity and calculate delta
            valid = True
            delta_score = 0
            
            # Check for 5+ collinear in new lines
            for key, cnt in new_lines:
                if cnt >= 5:
                    valid = False
                    break
            if not valid:
                # Rollback
                for key, cnt in new_lines:
                    line_map[key].remove(idx)
                    if len(line_map[key]) == 0:
                        del line_map[key]
                for key, cnt in old_lines:
                    if key not in line_map:
                        line_map[key] = set()
                    line_map[key].add(idx)
                current[idx] = old_pt
                no_improve += 1
                continue
            
            # Calculate score change
            # Lines that lost the point (old_lines)
            for key, old_cnt in old_lines:
                new_cnt = len(line_map.get(key, set()))
                if old_cnt == 4 and new_cnt == 3:
                    delta_score -= 1
                elif old_cnt == 3 and new_cnt == 4:
                    # This shouldn't happen for old lines unless the point was removed and re-added?
                    # No, old lines are lines that contained idx.
                    # If idx is removed, count drops.
                    pass
            
            # Lines that gained the point (new_lines)
            for key, new_cnt in new_lines:
                # We need to know the count BEFORE adding idx
                # The count is currently len(line_map[key]) which includes idx.
                # So count_before = new_cnt - 1
                count_before = new_cnt - 1
                if count_before == 3 and new_cnt == 4:
                    delta_score += 1
                elif count_before == 4 and new_cnt == 5:
                    # Already handled by valid check, but just in case
                    valid = False
                    break
            
            if not valid:
                # Rollback
                for key, cnt in new_lines:
                    line_map[key].remove(idx)
                    if len(line_map[key]) == 0:
                        del line_map[key]
                for key, cnt in old_lines:
                    if key not in line_map:
                        line_map[key] = set()
                    line_map[key].add(idx)
                current[idx] = old_pt
                no_improve += 1
                continue

            new_score = current_score + delta_score
            
            if new_score > current_score or (T > 0.05 and rng.random() < math.exp((new_score - current_score) / T)):
                if new_score > current_score:
                    no_improve = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = new_score
            else:
                # Rollback
                for key, cnt in new_lines:
                    line_map[key].remove(idx)
                    if len(line_map[key]) == 0:
                        del line_map[key]
                for key, cnt in old_lines:
                    if key not in line_map:
                        line_map[key] = set()
                    line_map[key].add(idx)
                current[idx] = old_pt
                no_improve += 1
            
            T = max(0.05, T * 0.999)

    return best_points
# EVOLVE-BLOCK-END
