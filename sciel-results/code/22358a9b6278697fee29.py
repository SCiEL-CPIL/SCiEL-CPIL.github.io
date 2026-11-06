import math
import random
import time
from collections import defaultdict
from itertools import combinations

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    dy = y2 - y1
    dx = x2 - x1
    g = math.gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    elif dx_n == 0 and dy_n < 0:
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g = math.gcd(abs(a), abs(b), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def evaluate(points):
    """Full evaluation: count lines with exactly 4 points, reject if any line has >=5."""
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_points = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_points[key].append(i)
    
    score = 0
    for key, pts_on_line in line_points.items():
        k = len(pts_on_line)
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def compute_line_info(points):
    """Compute line structure: returns (line_map, pair_to_line, four_line_count).
    line_map: line_key -> set of point indices
    pair_to_line: (i,j) -> line_key
    four_line_count: number of lines with exactly 4 points
    """
    n = len(points)
    line_map = defaultdict(set)
    pair_to_line = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
            pair_to_line[(i, j)] = key
    four_count = sum(1 for key, s in line_map.items() if len(s) == 4)
    return line_map, pair_to_line, four_count

def incremental_eval(points, line_map, pair_to_line, four_count, moved_idx, old_pt, new_pt):
    """
    Fast incremental evaluation after moving point moved_idx from old_pt to new_pt.
    Returns (score, valid, updated_line_map, updated_pair_to_line, updated_four_count).
    score is -1 if invalid (5+ collinear or duplicate).
    """
    n = len(points)
    
    # Check distinctness
    for i in range(n):
        if i != moved_idx and points[i] == new_pt:
            return -1, False, line_map, pair_to_line, four_count
    
    # Collect all lines affected by this move
    # Lines that contained old_pt (pairs involving moved_idx)
    affected_lines = set()
    for j in range(n):
        if j == moved_idx:
            continue
        pair_key = (moved_idx, j) if moved_idx < j else (j, moved_idx)
        if pair_key in pair_to_line:
            affected_lines.add(pair_to_line[pair_key])
    
    # For each affected line, determine old count and whether it was a 4-line
    old_line_sizes = {}
    for lk in affected_lines:
        old_line_sizes[lk] = len(line_map[lk])
    
    # Remove old_pt from all affected lines
    for lk in affected_lines:
        line_map[lk].discard(moved_idx)
        # Update pair_to_line: remove pairs involving moved_idx for this line
        for j in range(n):
            if j == moved_idx:
                continue
            pair_key = (moved_idx, j) if moved_idx < j else (j, moved_idx)
            if pair_key in pair_to_line and pair_to_line[pair_key] == lk:
                del pair_to_line[pair_key]
    
    # Now compute new lines for new_pt
    new_affected_lines = set()
    for j in range(n):
        if j == moved_idx:
            continue
        new_key = line_hash(new_pt, points[j])
        new_affected_lines.add(new_key)
        # Add to line_map
        line_map[new_key].add(moved_idx)
        line_map[new_key].add(j)
        pair_to_line[(moved_idx, j) if moved_idx < j else (j, moved_idx)] = new_key
    
    # Recompute four_count: subtract old 4-lines that are no longer size 4, add new ones
    # First, check for any line with >= 5 points (invalid)
    for lk in new_affected_lines:
        if len(line_map[lk]) >= 5:
            # Invalid - need to revert
            # This is complex; let's just do a full re-eval
            return -1, False, line_map, pair_to_line, four_count
    
    # Recalculate four_count from scratch for affected lines only
    # Actually, let's just recount all - it's only ~100 lines max
    new_four_count = 0
    for lk, s in line_map.items():
        if len(s) == 4:
            new_four_count += 1
        elif len(s) >= 5:
            return -1, False, line_map, pair_to_line, four_count
    
    return new_four_count, True, line_map, pair_to_line, new_four_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Start with 4x4 grid which has 10 four-point lines
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    def local_search_line_completion(points, time_limit):
        """
        Local search using line-completion heuristic.
        30% of moves: pick a line with 3 points, try to place a 4th point on it.
        70% of moves: random perturbation.
        Uses incremental evaluation for speed.
        """
        current_points = [p[:] for p in points]
        line_map, pair_to_line, current_score = compute_line_info(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 1.5
        cooling = 0.998
        no_improve = 0
        max_no_improve = 3000
        iter_count = 0
        
        while iter_count < max_no_improve * 3:
            if time.time() > time_limit:
                break
            iter_count += 1
            
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            strategy = random.random()
            
            if strategy < 0.30:
                # Line completion: find a line with exactly 3 points, try to add a 4th
                # Pick a random line with 3 points
                three_lines = [lk for lk, s in line_map.items() if len(s) == 3]
                if three_lines:
                    target_line = random.choice(three_lines)
                    # Get the 3 points on this line
                    pts_on_line = list(line_map[target_line])
                    # Try to place moved_idx on this line at a position that keeps it on the line
                    # The line is defined by the 3 points; find a 4th integer point on it
                    # Extract line equation from the 3 points
                    p_a = current_points[pts_on_line[0]]
                    p_b = current_points[pts_on_line[1]]
                    x1, y1 = p_a
                    x2, y2 = p_b
                    dx = x2 - x1
                    dy = y2 - y1
                    g = math.gcd(abs(dx), abs(dy))
                    if g > 0:
                        dx_n = dx // g
                        dy_n = dy // g
                    else:
                        continue
                    
                    # Parametric: (x1 + t*dx_n, y1 + t*dy_n) for integer t
                    # Find t values that give integer points in range
                    # Current t values for the 3 points
                    t_vals = set()
                    for pi in pts_on_line:
                        px, py = current_points[pi]
                        if dx_n != 0:
                            t = (px - x1) // dx_n
                        else:
                            t = (py - y1) // dy_n
                        t_vals.add(t)
                    
                    # Try nearby t values not already used
                    candidates = []
                    for t in range(min(t_vals) - 3, max(t_vals) + 4):
                        if t not in t_vals:
                            nx = x1 + t * dx_n
                            ny = y1 + t * dy_n
                            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                                candidates.append([nx, ny])
                    
                    if candidates:
                        new_pt = random.choice(candidates)
                    else:
                        new_pt = [old_pt[0] + random.randint(-3, 3), old_pt[1] + random.randint(-3, 3)]
                else:
                    # No 3-lines, do a small random move
                    new_pt = [old_pt[0] + random.randint(-3, 3), old_pt[1] + random.randint(-3, 3)]
            elif strategy < 0.50:
                # Small perturbation
                step = random.randint(1, 3)
                new_pt = [old_pt[0] + random.randint(-step, step), old_pt[1] + random.randint(-step, step)]
            elif strategy < 0.65:
                # Medium perturbation
                step = random.randint(3, 8)
                new_pt = [old_pt[0] + random.randint(-step, step), old_pt[1] + random.randint(-step, step)]
            elif strategy < 0.80:
                # Large jump
                new_pt = [random.randint(-30, 30), random.randint(-30, 30)]
            elif strategy < 0.92:
                # Move towards another point
                other_idx = random.randint(0, 15)
                if other_idx != idx:
                    other = current_points[other_idx]
                    t = random.randint(1, 10) / 10.0
                    new_pt = [int(old_pt[0] + t * (other[0] - old_pt[0])),
                              int(old_pt[1] + t * (other[1] - old_pt[1]))]
                else:
                    new_pt = [old_pt[0] + random.randint(-5, 5), old_pt[1] + random.randint(-5, 5)]
            else:
                # Grid-aligned
                gx = round(old_pt[0] / 3) * 3 + random.randint(-2, 2)
                gy = round(old_pt[1] / 3) * 3 + random.randint(-2, 2)
                new_pt = [gx, gy]
            
            if not (-1000 <= new_pt[0] <= 1000 and -1000 <= new_pt[1] <= 1000):
                no_improve += 1
                continue
            if new_pt == old_pt:
                no_improve += 1
                continue
            
            # Apply move
            current_points[idx] = new_pt
            
            # Incremental evaluation
            new_score, valid, line_map, pair_to_line, current_score = incremental_eval(
                current_points, line_map, pair_to_line, current_score, idx, old_pt, new_pt
            )
            
            if not valid or new_score == -1:
                current_points[idx] = old_pt
                # Rebuild line info since we modified it
                line_map, pair_to_line, current_score = compute_line_info(current_points)
                no_improve += 1
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = [p[:] for p in current_points]
                    temp = 1.5
            else:
                if delta == 0:
                    current_score = new_score
                    no_improve += 1
                else:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_points[idx] = old_pt
                        line_map, pair_to_line, current_score = compute_line_info(current_points)
                    no_improve += 1
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
            
            if no_improve > max_no_improve:
                break
        
        return best_local, best_local_score
    
    # Phase 1: Optimize the initial grid with line-completion search
    pts, score = local_search_line_completion(initial_points, deadline - 80.0)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
    
    if best_score >= 15:
        return best_points
    
    # Phase 2: Try structured configurations
    for offset in range(-2, 3):
        if time.time() > deadline - 15.0:
            break
        pts = []
        for x in range(4):
            for y in range(4):
                px = x + (offset if x == 3 else 0)
                py = y + (offset if y == 3 else 0)
                pts.append([px, py])
        s = evaluate(pts)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    # Phase 3: Random restarts with diverse configurations
    restart_attempts = 0
    max_restarts = 80
    
    while time.time() < deadline - 10.0 and restart_attempts < max_restarts:
        restart_attempts += 1
        
        mode = random.randint(0, 5)
        
        if mode == 0:
            # Perturbed 4x4 grid
            pts = [p[:] for p in initial_points]
            for i in range(16):
                if random.random() < 0.4:
                    pts[i] = [pts[i][0] + random.randint(-4, 4), pts[i][1] + random.randint(-4, 4)]
        elif mode == 1:
            # Random in small box
            pts = []
            while len(pts) < 16:
                x = random.randint(-8, 8)
                y = random.randint(-8, 8)
                if [x, y] not in pts:
                    pts.append([x, y])
        elif mode == 2:
            # Random in medium box
            pts = []
            while len(pts) < 16:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if [x, y] not in pts:
                    pts.append([x, y])
        elif mode == 3:
            # Structured: two overlapping grids
            pts = []
            base = random.randint(-5, 5)
            for x in range(4):
                for y in range(4):
                    if x < 2 and y < 2:
                        pts.append([base + x, base + y])
                    elif x < 2 and y >= 2:
                        pts.append([base + x + 1, base + y])
                    elif x >= 2 and y < 2:
                        pts.append([base + x, base + y + 1])
                    else:
                        pts.append([base + x + 1, base + y + 1])
        elif mode == 4:
            # Grid with one row/column shifted
            pts = []
            shift = random.randint(1, 4)
            for x in range(4):
                for y in range(4):
                    if x == 2:
                        pts.append([x + shift, y])
                    else:
                        pts.append([x, y])
        else:
            # Diagonal-biased random
            pts = []
            while len(pts) < 16:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if [x, y] not in pts:
                    pts.append([x, y])
        
        s = evaluate(pts)
        if s < 0:
            continue
        
        # Run line-completion local search
        remaining = deadline - 10.0 - time.time()
        if remaining < 2.0:
            break
        pts, score = local_search_line_completion(pts, deadline - 10.0)
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
        if best_score >= 15:
            break
    
    # Phase 4: Final polish - intensive search on best found
    if time.time() < deadline - 5.0:
        pts, score = local_search_line_completion(best_points, deadline - 3.0)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
    
    return best_points
