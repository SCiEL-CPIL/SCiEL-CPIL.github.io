"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _line_key(a, b, c):
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(pts):
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
            key = _line_key(a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
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

def _build_line_membership(pts):
    n = len(pts)
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            key = _line_key(a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def _delta_score(pts, idx, new_x, new_y):
    n = len(pts)
    line_map = _build_line_membership(pts)
    old_ordinary = sum(1 for v in line_map.values() if len(v) == 2)
    
    new_line_map = {}
    for key, members in line_map.items():
        if idx in members:
            new_members = members - {idx}
            if len(new_members) >= 2:
                new_line_map[key] = new_members
        else:
            new_line_map[key] = members
    
    for j in range(n):
        if j == idx:
            continue
        x1, y1 = new_x, new_y
        x2, y2 = pts[j]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        key = _line_key(a, b, c)
        if key not in new_line_map:
            new_line_map[key] = set()
        new_line_map[key].add(idx)
        new_line_map[key].add(j)
    
    new_ordinary = sum(1 for v in new_line_map.values() if len(v) == 2)
    return new_ordinary - old_ordinary

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    # Strategy: Start with structured geometric configurations that naturally have
    # many collinearities, then repair/optimize using line-aware moves.
    
    initial_configs = []
    
    # Config A: Regular hexagon vertices (6) + center + 3 axis midpoints
    # Hexagon vertices at angles 0,60,120,180,240,300 degrees
    # Using integer approximations: (R,0), (R/2, R*sqrt(3)/2) ~ (5, 9) for R=10
    # Let's use a simpler integer-based hexagon-like structure
    # Actually, let's use points that form 3 lines of 4 each through a center
    # Line 1 (horizontal): (-3,0), (-1,0), (1,0), (3,0)
    # Line 2 (60 deg approx): use slope 1 for simplicity -> (-3,-3), (-1,-1), (1,1), (3,3)
    # Line 3 (120 deg approx): use slope -1 -> (-3,3), (-1,1), (1,-1), (3,-3)
    # But center (0,0) is shared. We need 12 distinct points.
    # Let's place 4 points on each of 3 lines through origin, excluding origin, and add origin? No, origin would be 4th on each.
    # If we have 3 lines through origin with 4 points each (including origin), that's 1 + 3*3 = 10 points.
    # Add 2 more points to create more collinearities.
    
    # Better: Use a 3x4 grid but perturb to create more collinearities?
    # The 3x4 grid has 10 ordinary lines. We want 6.
    
    # Let's try: 4 points on y=0, 4 points on y=1, 4 points on y=2, but shifted.
    # If we shift the middle row, we can create diagonals.
    
    # Known optimal for 12 points is 6 ordinary lines.
    # One construction: 3 lines of 4 points each, no two lines parallel, no three concurrent.
    # This creates 3 lines with 4 points. The other lines will be ordinary or have 3.
    # Let's try to construct 3 lines of 4 integer points.
    
    # Line 1: y = 0 -> (0,0), (1,0), (2,0), (3,0)
    # Line 2: y = x -> (0,0) is shared. Let's use (4,4), (5,5), (6,6), (7,7)
    # Line 3: y = 2x -> (0,0) is shared. Let's use (4,8), (5,10), (6,12), (7,14)
    # This gives 12 distinct points if we avoid overlap.
    # P1: (0,0), (1,0), (2,0), (3,0)
    # P2: (4,4), (5,5), (6,6), (7,7)
    # P3: (4,8), (5,10), (6,12), (7,14)
    # Check for accidental collinearities.
    initial_configs.append(
        [[0,0], [1,0], [2,0], [3,0],
         [4,4], [5,5], [6,6], [7,7],
         [4,8], [5,10], [6,12], [7,14]]
    )
    
    # Config B: 3 lines of 4, not concurrent.
    # L1: y=0: (0,0), (1,0), (2,0), (3,0)
    # L2: y=1: (0,1), (1,1), (2,1), (3,1) -> parallel, not good for minimizing ordinary lines?
    # Parallel lines don't intersect, so they don't create extra collinearities between them.
    # We want non-parallel.
    # L1: y=0: (0,0), (1,0), (2,0), (3,0)
    # L2: y=x: (4,4), (5,5), (6,6), (7,7)
    # L3: y=-x+10: (0,10), (1,9), (2,8), (3,7)
    # Check intersections: L1 & L2 intersect at (0,0) which is in L1. L2 point (4,4) is not on L1.
    # L1 & L3 intersect at (0,0) and (10,0). (0,0) is in L1.
    # L2 & L3: x = -x+10 -> 2x=10 -> x=5, y=5. (5,5) is in L2! And (5,5) is not in L3?
    # L3 is y=-x+10. 5 = -5+10 = 5. Yes, (5,5) is on L3.
    # So (5,5) is on L2 and L3. This creates a concurrency at (5,5).
    # Points:
    # L1: (0,0), (1,0), (2,0), (3,0)
    # L2: (4,4), (5,5), (6,6), (7,7)
    # L3: (0,10), (1,9), (2,8), (3,7)
    # (5,5) is shared by L2 and L3.
    # This is a valid set of 12 points.
    initial_configs.append(
        [[0,0], [1,0], [2,0], [3,0],
         [4,4], [5,5], [6,6], [7,7],
         [0,10], [1,9], [2,8], [3,7]]
    )
    
    # Config C: 4x3 grid (standard)
    initial_configs.append(
        [[x, y] for x in range(4) for y in range(3)]
    )
    
    # Config D: 2x6 grid
    initial_configs.append(
        [[x, y] for x in range(6) for y in range(2)]
    )
    
    best_pts = None
    best_score = float('inf')
    
    for config in initial_configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        best_pts = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
        best_score = _count_ordinary(best_pts)
    
    if best_score <= 6:
        return [list(p) for p in best_pts]
    
    # Line-aware repair:
    # 1. Identify all lines with >= 3 points.
    # 2. Try to move points from ordinary lines to join existing lines or create new high-multiplicity lines.
    
    # Phase 1: Random restarts with line-aware moves
    for restart in range(5):
        if time.time() > deadline:
            break
        
        current_pts = [list(p) for p in best_pts]
        current_score = best_score
        
        # Try to improve current_pts
        improved = True
        while improved and time.time() < deadline:
            improved = False
            line_map = _build_line_membership(current_pts)
            
            # Find points that are on ordinary lines (lines with exactly 2 points)
            # and try to move them to join a line with >= 3 points or create a new one.
            
            # Strategy: For each point, try moving it to a position that maximizes collinearity.
            # Specifically, try to place it on an existing line with >= 3 points.
            
            ordinary_lines = [key for key, members in line_map.items() if len(members) == 2]
            
            # Try moving each point
            for idx in range(12):
                if time.time() > deadline:
                    break
                orig_x, orig_y = current_pts[idx]
                
                # Candidate moves:
                # 1. Move to a point on an existing high-multiplicity line (if not already there)
                # 2. Move to create a new line with 3 points
                
                # Try small random perturbations first (hill climbing)
                for _ in range(10):
                    step = random.randint(1, 10)
                    new_x = orig_x + random.randint(-step, step)
                    new_y = orig_y + random.randint(-step, step)
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if new_x == orig_x and new_y == orig_y:
                        continue
                    dup = False
                    for i, p in enumerate(current_pts):
                        if i != idx and p[0] == new_x and p[1] == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    delta = _delta_score(current_pts, idx, new_x, new_y)
                    if delta < 0:
                        current_pts[idx] = [new_x, new_y]
                        current_score += delta
                        improved = True
                        break
                
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_pts = [list(p) for p in current_pts]
                
                if best_score <= 6:
                    return [list(p) for p in best_pts]
    
    # Phase 2: Systematic local search with larger steps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(12):
            if time.time() > deadline:
                break
            orig_x, orig_y = best_pts[idx]
            
            # Try a grid of nearby points
            for dx in range(-10, 11, 2):
                if time.time() > deadline:
                    break
                for dy in range(-10, 11, 2):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    dup = False
                    for i, p in enumerate(best_pts):
                        if i != idx and p[0] == new_x and p[1] == new_y:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    delta = _delta_score(best_pts, idx, new_x, new_y)
                    if delta < 0:
                        best_pts[idx] = [new_x, new_y]
                        best_score += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if best_score <= 6:
            break
    
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
