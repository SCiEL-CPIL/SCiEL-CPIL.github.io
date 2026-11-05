"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            count = 2
            for k in range(n):
                if k == i or k == j: continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j: continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
    return triples

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + (n - idx) <= best[0]:
            return
        
        dfs(idx + 1, current, current_size)
        
        can_add = True
        for i in current:
            for j in current:
                if i < j and collinear[idx][i][j]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
            
    dfs(0, [], 0)
    return best[0]

def _construct_with_triples(deadline):
    """
    Deterministic construction: explicitly build 3-point lines (triples) to maximize
    the number of collinear triples while ensuring no 4 are collinear.
    
    Strategy: 
    - Start with a grid-like base
    - For each potential line (defined by direction and offset), try to place exactly 3 points
    - Use a greedy approach: pick lines that maximize new triples without creating 4-collinear
    """
    start_time = time.time()
    rng = random.Random(12345)
    
    # Generate candidate lines: all possible lines with 3+ integer points in [-100,100]^2
    # We'll focus on lines with small slopes to keep coordinates manageable
    candidate_lines = []
    
    # Horizontal lines: y = c, need at least 3 points
    for y in range(-100, 101):
        xs = list(range(-100, 101))
        candidate_lines.append(('h', y, xs))
    
    # Vertical lines: x = c
    for x in range(-100, 101):
        ys = list(range(-100, 101))
        candidate_lines.append(('v', x, ys))
    
    # Diagonal lines with slope 1: y = x + c
    for c in range(-200, 201):
        pts = []
        for x in range(-100, 101):
            y = x + c
            if -100 <= y <= 100:
                pts.append((x, y))
        if len(pts) >= 3:
            candidate_lines.append(('d1', c, pts))
    
    # Diagonal lines with slope -1: y = -x + c
    for c in range(-200, 201):
        pts = []
        for x in range(-100, 101):
            y = -x + c
            if -100 <= y <= 100:
                pts.append((x, y))
        if len(pts) >= 3:
            candidate_lines.append(('d-1', c, pts))
    
    # Other rational slopes: y = (a/b)*x + c where gcd(a,b)=1, |a|,|b| <= 3
    for a in range(-3, 4):
        for b in range(1, 4):
            if math.gcd(abs(a), b) != 1:
                continue
            if a == 0 and b != 1:  # already covered horizontal
                continue
            if a == 1 and b == 1:  # already covered
                continue
            if a == -1 and b == 1:  # already covered
                continue
            
            for c in range(-300, 301):
                pts = []
                for x in range(-100, 101):
                    # y = (a/b)*x + c => b*y = a*x + b*c
                    num = a * x + b * c
                    if num % b == 0:
                        y = num // b
                        if -100 <= y <= 100:
                            pts.append((x, y))
                if len(pts) >= 3:
                    candidate_lines.append((f'{a}/{b}', c, pts))
    
    if time.time() > deadline - 60:
        return None
    
    # Shuffle candidates for diversity
    rng.shuffle(candidate_lines)
    
    best_config = None
    best_triple_count = -1
    
    # Try to build configurations by selecting lines and placing points
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        
        points = {}  # (x,y) -> index
        point_list = []
        line_usage = {}  # line_key -> set of point indices on this line
        point_lines = {}  # point index -> set of line keys containing this point
        
        # Try to assign points to lines, ensuring each line has exactly 3 points
        # and no point is on more than 2 lines (to avoid 4-collinear issues)
        
        # Greedy: pick lines that can still accommodate 3 points
        available_lines = [l for l in candidate_lines if len(l[2]) >= 3]
        rng.shuffle(available_lines)
        
        success = True
        for line_key, line_offset, line_pts in available_lines:
            if time.time() > deadline - 30:
                success = False
                break
            
            lk = (line_key, line_offset)
            if lk in line_usage:
                continue
            
            # Find 3 points on this line that are not already heavily used
            candidates = []
            for pt in line_pts:
                if pt in points:
                    idx = points[pt]
                    # Check if this point is already on 2 lines
                    if len(point_lines.get(idx, set())) >= 2:
                        continue
                    candidates.append((pt, idx))
                else:
                    candidates.append((pt, None))
            
            if len(candidates) < 3:
                continue
            
            # Pick 3 candidates, preferring new points
            new_cands = [c for c in candidates if c[1] is None]
            old_cands = [c for c in candidates if c[1] is not None]
            
            selected = []
            # Prefer new points first
            for pt, idx in new_cands[:3]:
                selected.append((pt, idx))
            # Then add old points if needed
            if len(selected) < 3:
                for pt, idx in old_cands:
                    if len(selected) >= 3:
                        break
                    selected.append((pt, idx))
            
            if len(selected) < 3:
                continue
            
            # Verify no 4-collinear would be created
            valid = True
            for pt, idx in selected:
                if idx is not None:
                    # Check if adding this point to the line creates 4-collinear
                    # The line already has some points, we're adding one more
                    existing_on_line = line_usage.get(lk, set())
                    if len(existing_on_line) >= 2:
                        # Would make 3 or more, but we need to check if it makes 4
                        # Actually, we're selecting exactly 3 for this line, so it should be fine
                        # But if the point is already on another line with 2+ points, that's a problem
                        pass
            
            # Assign points
            line_pts_used = set()
            for pt, idx in selected:
                if idx is None:
                    idx = len(point_list)
                    point_list.append(list(pt))
                    points[pt] = idx
                    point_lines[idx] = set()
                point_lines[idx].add(lk)
                line_pts_used.add(idx)
            
            line_usage[lk] = line_pts_used
            
            if len(point_list) >= 20:
                break
        
        if len(point_list) != 20 or not success:
            continue
        
        # Check validity
        if _has_four_collinear(point_list):
            continue
        
        triples = _count_collinear_triples(point_list)
        if triples > best_triple_count:
            best_triple_count = triples
            best_config = [p[:] for p in point_list]
    
    return best_config

def _local_search(points, deadline):
    """
    Local search to maximize the number of collinear triples.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        i = rng.randint(0, 19)
        orig = points[i][:]
        
        # Generate candidates: points on lines defined by pairs of other points
        candidates = set()
        for j in range(20):
            if j == i: continue
            for k in range(j+1, 20):
                if k == i: continue
                x1, y1 = points[j]
                x2, y2 = points[k]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx //= g
                dy //= g
                
                for t in range(-200, 201):
                    x = x1 + t * dx
                    y = y1 + t * dy
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.add((x, y))
        
        for _ in range(10):
            nx = orig[0] + rng.randint(-5, 5)
            ny = orig[1] + rng.randint(-5, 5)
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                candidates.add((nx, ny))
        
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        for cand in cand_list:
            if any(points[m] == list(cand) for m in range(20) if m != i):
                continue
            
            points[i] = list(cand)
            
            if _has_four_collinear(points):
                points[i] = orig
                continue
            
            new_triples = _count_collinear_triples(points)
            if new_triples > current_triples:
                current_triples = new_triples
                improved = True
                break
            else:
                points[i] = orig
        
        if not improved:
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First try the constructive triple-based approach
    config = _construct_with_triples(deadline - 20)
    
    if config is not None:
        # Refine with local search
        refined = _local_search(config, deadline - 10)
        return refined
    
    # Fallback: random initialization with local search
    best_points = None
    best_triples = -1
    
    for seed in range(50):
        if time.time() > deadline - 30:
            break
            
        rng = random.Random(seed)
        points = []
        used = set()
        
        while len(points) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) in used:
                continue
            points.append([x, y])
            used.add((x, y))
            
            if _has_four_collinear(points):
                points.pop()
                used.remove((x, y))
        
        if len(points) != 20:
            continue
            
        improved_points = _local_search(points, deadline - 10)
        improved_triples = _count_collinear_triples(improved_points)
        
        if improved_triples > best_triples:
            best_triples = improved_triples
            best_points = [p[:] for p in improved_points]
            
    if best_points is not None:
        return best_points
    else:
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
