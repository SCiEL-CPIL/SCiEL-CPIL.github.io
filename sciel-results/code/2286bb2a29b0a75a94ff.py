import time
import random
import math
import numpy as np
from itertools import combinations

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if len(set(tuple(p) for p in points)) != n:
        return -1
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    
    lines_found = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            count = 0
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    count += 1
                    indices_on_line.add(k)
            if count >= 5:
                return -1
            if count == 4:
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
    return len(lines_found)

def _get_line_id(p1, p2, pts, n):
    """
    Returns a canonical key for the line passing through p1 and p2,
    based on the sorted tuple of indices of all points on that line.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    indices = []
    for k in range(n):
        pk = pts[k]
        if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
            indices.append(k)
    return tuple(sorted(indices))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    N = 16
    # We will use a 1D array of x and y coordinates for speed
    # Initial configuration: a 4x4 grid, slightly perturbed to break symmetries if needed, 
    # but let's start with a known good structure or random dense grid.
    # Actually, starting from a 4x4 grid gives 6 lines (4 rows, 4 cols... wait, 4x4 has 4 rows of 4, 4 cols of 4. Total 8 lines? 
    # Rows: y=0,1,2,3. Cols: x=0,1,2,3. 
    # Diagonals? No, 4x4 grid doesn't have 4-point diagonals except main ones? 
    # Main diagonal has 4 points. Anti-diagonal has 4 points. 
    # So 4x4 grid has 4+4+2 = 10 lines? 
    # Let's check: 
    # (0,0)(1,1)(2,2)(3,3) - yes
    # (0,3)(1,2)(2,1)(3,0) - yes
    # Any others? (0,0)(1,2)... no.
    # So 4x4 grid score is likely 10.
    # The current program scores 1.0, which implies it found 10 lines (since seed score is 1.0 and 10 is the value).
    # Wait, the task says "real value 15 for n=16 is an upper reference". 
    # If seed is 1.0, and seed is 4x4 grid, then 4x4 grid score must be 10.
    # Let's verify 4x4 grid score.
    # Points: (0,0)..(3,3).
    # Lines with 4 points:
    # Rows: y=0,1,2,3 (4 lines)
    # Cols: x=0,1,2,3 (4 lines)
    # Diagonals: slope 1, -1 (2 lines)
    # Total 10.
    # So seed score = 10.
    # We want to beat 10.
    
    # Strategy: Simulated Annealing
    # State: list of 16 points.
    # Move: Pick a random point, move it to a random nearby integer coordinate (within bounds).
    # Or: Pick a random point, swap with another? No, coordinates are continuous-ish integers.
    # Better: Pick a random point, change its x or y by a small random amount.
    # To ensure we explore, we should sometimes jump.
    
    # Data structures for incremental scoring:
    # line_counts: dict mapping line_id (tuple of sorted indices) -> count of points on that line?
    # No, line_id is defined by the set of points ON the line.
    # If we move point i, the line_id changes for all lines passing through i.
    # This is complex to update incrementally because the "identity" of the line changes if the set of points changes.
    
    # Alternative Incremental Scoring:
    # Maintain a dictionary: `line_to_points` mapping a canonical geometric line representation (e.g., normalized (A,B,C) for Ax+By+C=0) to a set of point indices.
    # When a point moves:
    # 1. Remove point i from all geometric lines it was on.
    # 2. Move point i to new position.
    # 3. Add point i to all geometric lines it is now on.
    # 4. Recalculate the score based on the counts in `line_to_points`.
    
    # Geometric Line Key:
    # For a line passing through (x1,y1) and (x2,y2):
    # A = y1 - y2
    # B = x2 - x1
    # C = x1*y2 - x2*y1
    # Normalize (A, B, C) by dividing by GCD and fixing sign (e.g., A>0 or A==0,B>0).
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1*y2 - x2*y1
        if A == 0 and B == 0:
            return None
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        # Normalize sign
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    # Initialize state
    # Start with 4x4 grid
    xs = [0, 1, 2, 3] * 4
    ys = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    # Wait, 4x4 grid is:
    # (0,0) (1,0) (2,0) (3,0)
    # (0,1) (1,1) (2,1) (3,1)
    # ...
    # Let's generate properly
    init_pts = []
    for x in range(4):
        for y in range(4):
            init_pts.append([x, y])
    
    xs = np.array([p[0] for p in init_pts])
    ys = np.array([p[1] for p in init_pts])
    
    # Build initial line map
    line_map = {} # key -> set of indices
    for i in range(N):
        for j in range(i+1, N):
            key = get_line_key(init_pts[i], init_pts[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    def get_score(line_map):
        score = 0
        for key, pts_in_line in line_map.items():
            cnt = len(pts_in_line)
            if cnt == 4:
                score += 1
            elif cnt >= 5:
                return -1
        return score
    
    current_score = get_score(line_map)
    best_score = current_score
    best_xs = xs.copy()
    best_ys = ys.copy()
    
    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    # We need to be careful with time. 
    # Each move: 
    # 1. Remove point i from old lines. O(degree(i)). Degree is small, maybe 10-20.
    # 2. Update coords.
    # 3. Add point i to new lines. O(degree(i)).
    # 4. Check if any line has >= 5 points. We can track this incrementally or just check affected lines.
    # 5. Calculate score change.
    
    # To speed up, we only recalculate score for affected lines.
    # But `get_score` iterates over all lines. 
    # Better: Maintain `current_score` and `invalid_count` (number of lines with >= 5).
    # If `invalid_count > 0`, score is -1.
    
    invalid_count = 0
    # Initialize invalid_count
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) >= 5:
            invalid_count += 1
            
    def update_line_count(key, add_index, remove_index, line_map, invalid_count):
        # This is tricky because we are modifying the sets.
        # Let's do a simpler approach: 
        # Since N=16 is small, we can just recalculate the score from scratch if the move is promising?
        # No, that's O(N^2 * N) = O(N^3) = 4096 ops. 
        # 100,000 moves * 4096 = 400M ops. Might be too slow in Python.
        
        # Let's try the incremental update on the line_map sets and track invalid_count.
        pass
    
    # Actually, let's stick to a simpler, robust incremental update.
    # When point i moves from (x_old, y_old) to (x_new, y_new):
    # 1. Find all lines containing i. Let this set be L_old.
    # 2. For each line in L_old:
    #    - Get the other points on the line.
    #    - The line key might change? No, the geometric line is defined by the other points if i is removed?
    #    - Actually, if we remove i, the line still exists geometrically if there are other points.
    #    - We need to update the set of points for that geometric line.
    #    - If the set becomes empty, remove the key from line_map.
    # 3. Find all lines containing the new position of i. Let this set be L_new.
    # 4. For each line in L_new:
    #    - Add i to the set of points for that geometric line.
    #    - If the key didn't exist, create it.
    
    # The challenge: "Find all lines containing i".
    # We can precompute for each point i, the list of line keys it belongs to.
    # point_lines: list of lists. point_lines[i] = list of keys.
    
    point_lines = [[] for _ in range(N)]
    for key, pts_in_line in line_map.items():
        for idx in pts_in_line:
            point_lines[idx].append(key)
    
    # To find new lines for a point at (x, y):
    # We need to find all other points j such that (x,y) and pts[j] define a line.
    # Then get the key for that line.
    # This is O(N) per move.
    
    def find_lines_for_point(x, y, idx, xs, ys, line_map):
        keys = set()
        for j in range(N):
            if j == idx:
                continue
            key = get_line_key((x, y), (xs[j], ys[j]))
            if key:
                keys.add(key)
        return keys
    
    # Main Loop
    steps = 0
    while T > T_min and time.time() < deadline - 10:
        steps += 1
        # Pick random point
        i = random.randint(0, N-1)
        old_x, old_y = xs[i], ys[i]
        
        # Generate new position
        # Random walk with occasional jumps
        if random.random() < 0.1:
            new_x = random.randint(-10, 10)
            new_y = random.random() < 0.5 and random.randint(-10, 10) or ys[i] + random.randint(-5, 5)
            new_y = random.randint(-10, 10)
        else:
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            continue
            
        # Check if point already exists at new position
        # We need to ensure distinct points.
        # Check if (new_x, new_y) is in current points (excluding i).
        # O(N) check.
        is_dup = False
        for k in range(N):
            if k != i and xs[k] == new_x and ys[k] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Perform move
        # 1. Remove i from old lines
        old_keys = point_lines[i]
        for key in old_keys:
            pts_in_line = line_map.get(key)
            if pts_in_line is None:
                continue
            pts_in_line.discard(i)
            if len(pts_in_line) == 0:
                del line_map[key]
            # Update invalid_count
            old_cnt = len(pts_in_line) + 1 # before removal
            new_cnt = len(pts_in_line)
            if old_cnt >= 5 and new_cnt < 5:
                invalid_count -= 1
            elif old_cnt < 5 and new_cnt >= 5:
                invalid_count += 1
                
        # 2. Update coordinates
        xs[i] = new_x
        ys[i] = new_y
        
        # 3. Add i to new lines
        new_keys = find_lines_for_point(new_x, new_y, i, xs, ys, line_map)
        point_lines[i] = []
        for key in new_keys:
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            point_lines[i].append(key)
            
            old_cnt = len(line_map[key]) - 1
            new_cnt = len(line_map[key])
            if old_cnt >= 5 and new_cnt < 5:
                invalid_count -= 1
            elif old_cnt < 5 and new_cnt >= 5:
                invalid_count += 1
                
        # 4. Calculate new score
        if invalid_count > 0:
            new_score = -1
        else:
            # Count lines with exactly 4 points
            # We can maintain a count of 4-point lines incrementally too, but let's just count.
            # To be fast, we only count if invalid_count == 0.
            # Iterating over line_map is O(L). L is small.
            new_score = 0
            for key, pts_in_line in line_map.items():
                if len(pts_in_line) == 4:
                    new_score += 1
            # If new_score is -1, it means invalid. But we already checked invalid_count.
            # So new_score is valid.
            
        # Accept/Reject
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_xs = xs.copy()
                best_ys = ys.copy()
        else:
            # Revert
            # 1. Remove i from new lines
            for key in point_lines[i]:
                pts_in_line = line_map.get(key)
                if pts_in_line:
                    pts_in_line.discard(i)
                    if len(pts_in_line) == 0:
                        del line_map[key]
                    old_cnt = len(pts_in_line) + 1
                    new_cnt = len(pts_in_line)
                    if old_cnt >= 5 and new_cnt < 5:
                        invalid_count -= 1
                    elif old_cnt < 5 and new_cnt >= 5:
                        invalid_count += 1
                        
            # 2. Revert coordinates
            xs[i] = old_x
            ys[i] = old_y
            
            # 3. Add i to old lines
            for key in old_keys:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                
                old_cnt = len(line_map[key]) - 1
                new_cnt = len(line_map[key])
                if old_cnt >= 5 and new_cnt < 5:
                    invalid_count -= 1
                elif old_cnt < 5 and new_cnt >= 5:
                    invalid_count += 1
                    
            # Revert point_lines
            point_lines[i] = old_keys
            
            # Revert current_score
            # current_score remains the old score
            
        T *= cooling
        
    # Final Result
    result = [[int(best_xs[i]), int(best_ys[i])] for i in range(N)]
    
    # Final Validation
    score = _count_lines_exactly_4(result)
    if score < 0:
        # Fallback to 4x4 grid
        result = []
        for x in range(4):
            for y in range(4):
                result.append([x, y])
                
    return result

if __name__ == "__main__":
    pts = construct()
    print(_count_lines_exactly_4(pts))
