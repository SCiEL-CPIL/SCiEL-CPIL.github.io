import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def grid_configuration():
    points = [[x, y] for x in range(4) for y in range(4)]
    return points, fast_count(points)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Strategy: Combinatorial Design / MaxSAT-like search
    # We view the configuration as a set of 16 points.
    # The "variables" are the positions of these points.
    # The "constraints" are:
    #   1. All points distinct.
    #   2. No line has > 4 points.
    # The "objective" is to maximize the number of lines with exactly 4 points.
    
    # Instead of moving points randomly, we use a "Slot Filling" approach.
    # We pre-define a set of "Candidate Lines" (potential 4-point lines).
    # We try to select a subset of these lines and place points such that
    # the intersection of these lines creates a valid configuration.
    
    # A simpler, robust combinatorial approach:
    # 1. Generate a pool of "good" 4-point lines (e.g., from a grid or structured patterns).
    # 2. Use a greedy/set-cover-like algorithm to select lines that share points efficiently.
    # 3. Refine with local search on the "line selection" rather than point coordinates.
    
    # However, since we need integer coordinates, we can use a "Pattern Matching" strategy.
    # We know that optimal configurations often resemble affine grids or specific lattices.
    
    # Let's try a hybrid:
    # 1. Start with a 4x4 grid (known to have many 4-lines).
    # 2. Use "Line Swap" moves: Identify two lines with 4 points. Try to swap a point from one to another to maintain validity and potentially create more 4-lines or fix 5-lines.
    # 3. Crucially, use a "Constraint Propagation" check before accepting a move.
    
    points = [[x, y] for x in range(4) for y in range(4)]
    current_score = fast_count(points)
    best_points = [p[:] for p in points]
    best_score = current_score

    if current_score < 0:
        return best_points

    # Helper to get lines for a point set
    def get_lines(pts):
        lines = {}
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        return lines

    lines = get_lines(points)
    
    # Objective: Maximize 4-lines, strictly avoid 5+ lines.
    # Score function for evaluation:
    # +1 for each 4-line
    # -100 for each 5+ line (hard constraint penalty)
    
    def calc_score(lines):
        s = 0
        for k, v in lines.items():
            c = len(v)
            if c == 4:
                s += 1
            elif c >= 5:
                s -= 100
        return s

    current_eff_score = calc_score(lines)

    # Precompute potential "moves" based on line geometry.
    # A move is defined by:
    #   - Source point index
    #   - Target coordinate
    # We generate target coordinates that are "aligned" with existing lines.
    
    # To make this efficient, we will use a "Line-Driven" local search.
    # 1. Pick a line with 3 points (a "near-miss").
    # 2. Find the 4th point that would complete it.
    # 3. If that position is empty, move a point there (if moving it doesn't break a 5-line).
    # 4. If that position is occupied, try to "swap" the occupying point with one from a 3-line.
    
    # Also, to escape local minima, we allow random "jumps" to structured positions.
    
    # Generate a pool of "Structured Candidates"
    # These are points that lie on many grid lines.
    structured_pool = set()
    for x in range(-2, 8):
        for y in range(-2, 8):
            structured_pool.add((x, y))
    
    # Add some diagonal/other structured points
    for x in range(-2, 8):
        for y in range(-2, 8):
            if (x + y) % 2 == 0:
                structured_pool.add((x, y))
            if (x - y) % 2 == 0:
                structured_pool.add((x, y))
                
    # Expand pool slightly for diversity
    for _ in range(100):
        structured_pool.add((random.randint(-5, 15), random.randint(-5, 15)))
        
    structured_pool = list(structured_pool)

    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9999

    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.4:
            # Move Type A: Complete a 3-line
            three_lines = [(k, v) for k, v in lines.items() if len(v) == 3]
            if three_lines:
                line_key, line_pts = random.choice(three_lines)
                p1_idx, p2_idx, p3_idx = list(line_pts)
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                
                # Direction vector
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Find integer points on this line
                # p = p1 + t * (dx, dy)
                # We want points that are not already in the set.
                candidates = []
                for t in range(-5, 6):
                    if t == 0 or t == 1:
                        continue
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        candidates.append((nx, ny))
                
                if candidates:
                    # Shuffle candidates to avoid bias
                    random.shuffle(candidates)
                    
                    for target in candidates:
                        # Check if target is occupied
                        occupied_idx = None
                        for i in range(16):
                            if points[i][0] == target[0] and points[i][1] == target[1]:
                                occupied_idx = i
                                break
                        
                        if occupied_idx is None:
                            # Target is empty. We need to move a point to it.
                            # Which point? Prefer one that is not on a critical 4-line or 5-line.
                            # Or just pick a random point not on the current 3-line.
                            movable = [i for i in range(16) if i not in line_pts]
                            if not movable:
                                continue
                            
                            # Score movable points: prefer those on fewer 4/5 lines
                            def point_line_weight(idx):
                                w = 0
                                for k, v in lines.items():
                                    if idx in v:
                                        c = len(v)
                                        if c == 4:
                                            w += 1
                                        elif c >= 5:
                                            w -= 10
                                return w
                            
                            movable.sort(key=point_line_weight)
                            idx = movable[0]
                            
                            # Perform move
                            old_pos = points[idx][:]
                            points[idx] = list(target)
                            
                            # Update lines
                            # Remove idx from all lines it was on
                            for k in list(lines.keys()):
                                if idx in lines[k]:
                                    lines[k].remove(idx)
                                    if not lines[k]:
                                        del lines[k]
                            
                            # Add idx to all lines it is now on
                            for j in range(16):
                                if j == idx:
                                    continue
                                key = get_line_key(points[idx], points[j])
                                if key is None:
                                    continue
                                if key not in lines:
                                    lines[key] = set()
                                lines[key].add(idx)
                            
                            new_score = calc_score(lines)
                            delta = new_score - current_eff_score
                            
                            if delta >= 0 or random.random() < pow(2.71828, delta / T):
                                current_eff_score = new_score
                                if fast_count(points) > best_score:
                                    best_score = fast_count(points)
                                    best_points = [p[:] for p in points]
                                break
                            else:
                                # Revert
                                points[idx] = old_pos
                                for k in list(lines.keys()):
                                    if idx in lines[k]:
                                        lines[k].remove(idx)
                                        if not lines[k]:
                                            del lines[k]
                                for j in range(16):
                                    if j == idx:
                                        continue
                                    key = get_line_key(points[idx], points[j])
                                    if key is None:
                                        continue
                                    if key not in lines:
                                        lines[key] = set()
                                    lines[key].add(idx)
                                break # Try next candidate or next iteration

        elif move_type < 0.7:
            # Move Type B: Break a 5-line
            bad_lines = [(k, v) for k, v in lines.items() if len(v) >= 5]
            if bad_lines:
                line_key, line_pts = random.choice(bad_lines)
                idx = random.choice(list(line_pts))
                old_pos = points[idx][:]
                
                # Move to a random structured position
                for _ in range(20):
                    target = random.choice(structured_pool)
                    if target[0] == old_pos[0] and target[1] == old_pos[1]:
                        continue
                    if abs(target[0]) > 1000 or abs(target[1]) > 1000:
                        continue
                    
                    # Check collision
                    collision = False
                    for i in range(16):
                        if i != idx and points[i][0] == target[0] and points[i][1] == target[1]:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    points[idx] = list(target)
                    
                    # Update lines
                    for k in list(lines.keys()):
                        if idx in lines[k]:
                            lines[k].remove(idx)
                            if not lines[k]:
                                del lines[k]
                    for j in range(16):
                        if j == idx:
                            continue
                        key = get_line_key(points[idx], points[j])
                        if key is None:
                            continue
                        if key not in lines:
                            lines[key] = set()
                        lines[key].add(idx)
                    
                    new_score = calc_score(lines)
                    delta = new_score - current_eff_score
                    
                    if delta >= 0 or random.random() < pow(2.71828, delta / T):
                        current_eff_score = new_score
                        if fast_count(points) > best_score:
                            best_score = fast_count(points)
                            best_points = [p[:] for p in points]
                        break
                    else:
                        # Revert
                        points[idx] = old_pos
                        for k in list(lines.keys()):
                            if idx in lines[k]:
                                lines[k].remove(idx)
                                if not lines[k]:
                                    del lines[k]
                        for j in range(16):
                            if j == idx:
                                continue
                            key = get_line_key(points[idx], points[j])
                            if key is None:
                                continue
                            if key not in lines:
                                lines[key] = set()
                            lines[key].add(idx)

        else:
            # Move Type C: Random structured jump
            idx = random.randint(0, 15)
            old_pos = points[idx][:]
            target = random.choice(structured_pool)
            if target[0] == old_pos[0] and target[1] == old_pos[1]:
                continue
            if abs(target[0]) > 1000 or abs(target[1]) > 1000:
                continue
            
            collision = False
            for i in range(16):
                if i != idx and points[i][0] == target[0] and points[i][1] == target[1]:
                    collision = True
                    break
            if collision:
                continue
            
            points[idx] = list(target)
            
            # Update lines
            for k in list(lines.keys()):
                if idx in lines[k]:
                    lines[k].remove(idx)
                    if not lines[k]:
                        del lines[k]
            for j in range(16):
                if j == idx:
                    continue
                key = get_line_key(points[idx], points[j])
                if key is None:
                    continue
                if key not in lines:
                    lines[key] = set()
                lines[key].add(idx)
            
            new_score = calc_score(lines)
            delta = new_score - current_eff_score
            
            if delta >= 0 or random.random() < pow(2.71828, delta / T):
                current_eff_score = new_score
                if fast_count(points) > best_score:
                    best_score = fast_count(points)
                    best_points = [p[:] for p in points]
            else:
                # Revert
                points[idx] = old_pos
                for k in list(lines.keys()):
                    if idx in lines[k]:
                        lines[k].remove(idx)
                        if not lines[k]:
                            del lines[k]
                for j in range(16):
                    if j == idx:
                        continue
                    key = get_line_key(points[idx], points[j])
                    if key is None:
                        continue
                    if key not in lines:
                        lines[key] = set()
                    lines[key].add(idx)

        T *= cooling_rate
        if T < T_min:
            T = T_min
            if random.random() < 0.05:
                T = 3.0

    # Final verification
    final_score = fast_count(best_points)
    if final_score > best_score:
        best_score = final_score
    elif final_score < 0:
        grid_pts, grid_score = grid_configuration()
        return grid_pts

    return best_points
# EVOLVE-BLOCK-END
