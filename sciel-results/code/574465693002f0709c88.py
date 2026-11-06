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
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g2 = math.gcd(abs(a), abs(b), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def compute_line_counts(points):
    """Compute line pair counts for all pairs."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_counts[key] += 1
    return line_counts

def score_from_line_counts(line_counts):
    """Compute score and validity from line pair counts."""
    score = 0
    for key, count in line_counts.items():
        disc = 1 + 8 * count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc != disc:
            return -1
        k = (1 + sqrt_disc) // 2
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def evaluate(points):
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_counts = compute_line_counts(points)
    return score_from_line_counts(line_counts)

def incremental_score(points, idx, line_counts):
    """
    Compute new score after moving point idx.
    line_counts is the OLD line counts.
    Returns (new_score, new_line_counts) or (-1, None) if invalid.
    """
    n = len(points)
    old_pt = points[idx]
    
    # Check distinctness
    for i, p in enumerate(points):
        if i != idx and p == old_pt:
            return -1, None
    
    # Remove old lines involving idx
    # We need to compute what the new line counts would be
    # Lines involving idx: 15 lines
    # For each other point j, the line (idx, j) has some count c_old
    # After removing idx, that line loses 1 pair (the pair idx-j)
    # But wait, we're moving idx to a new position, so:
    # - Remove all pairs (idx, j) for j != idx from line_counts
    # - Add all new pairs (idx_new, j) for j != idx to line_counts
    
    new_line_counts = dict(line_counts)
    
    # Remove old contributions
    for j in range(n):
        if j == idx:
            continue
        key = line_hash(old_pt, points[j])
        new_line_counts[key] -= 1
        if new_line_counts[key] == 0:
            del new_line_counts[key]
    
    # Add new contributions
    for j in range(n):
        if j == idx:
            continue
        key = line_hash(old_pt, points[j])  # old_pt is now the new position
        if key in new_line_counts:
            new_line_counts[key] += 1
        else:
            new_line_counts[key] = 1
    
    score = score_from_line_counts(new_line_counts)
    return score, new_line_counts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Start with 4x4 grid which has 10 four-point lines
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    best_line_counts = compute_line_counts(best_points)
    
    def local_search(points, max_iters, time_limit):
        """Enhanced local search with incremental evaluation."""
        current_points = [p[:] for p in points]
        current_line_counts = compute_line_counts(current_points)
        current_score = score_from_line_counts(current_line_counts)
        if current_score < 0:
            return current_points, current_score, current_line_counts
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        best_local_line_counts = dict(current_line_counts)
        
        temp = 1.5
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 8000
        iter_count = 0
        
        while iter_count < max_iters:
            if time.time() > time_limit:
                break
            iter_count += 1
            
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            
            # Multiple move strategies
            strategy = random.random()
            
            if strategy < 0.30:
                # Small perturbation
                step = random.randint(1, 2)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            elif strategy < 0.50:
                # Medium perturbation
                step = random.randint(2, 6)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            elif strategy < 0.65:
                # Large jump
                new_x = random.randint(-20, 20)
                new_y = random.randint(-20, 20)
            elif strategy < 0.80:
                # Move towards another point
                other_idx = random.randint(0, 15)
                if other_idx != idx:
                    other = current_points[other_idx]
                    t = random.randint(1, 10) / 10.0
                    new_x = int(old_pt[0] + t * (other[0] - old_pt[0]))
                    new_y = int(old_pt[1] + t * (other[1] - old_pt[1]))
                else:
                    new_x = old_pt[0] + random.randint(-5, 5)
                    new_y = old_pt[1] + random.randint(-5, 5)
            else:
                # Grid-aligned move
                gx = round(old_pt[0] / 3) * 3 + random.randint(-2, 2)
                gy = round(old_pt[1] / 3) * 3 + random.randint(-2, 2)
                new_x = gx
                new_y = gy
            
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
            
            new_pt = [new_x, new_y]
            if new_pt == old_pt:
                continue
            
            # Check distinctness quickly
            is_distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p == new_pt:
                    is_distinct = False
                    break
            if not is_distinct:
                no_improve += 1
                continue
            
            # Apply move temporarily
            current_points[idx] = new_pt
            
            # Incremental evaluation
            new_score, new_line_counts = incremental_score(current_points, idx, current_line_counts)
            
            if new_score == -1:
                current_points[idx] = old_pt
                no_improve += 1
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                current_line_counts = new_line_counts
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = [p[:] for p in current_points]
                    best_local_line_counts = dict(current_line_counts)
                    temp = 1.5
            else:
                if delta == 0:
                    current_score = new_score
                    current_line_counts = new_line_counts
                    no_improve += 1
                else:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                        current_line_counts = new_line_counts
                    else:
                        current_points[idx] = old_pt
                    no_improve += 1
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
            
            if no_improve > max_no_improve:
                break
        
        return best_local, best_local_score, best_local_line_counts
    
    def pairwise_swap_search(points, line_counts, max_iters, time_limit):
        """Try swapping pairs of points."""
        current_points = [p[:] for p in points]
        current_line_counts = dict(line_counts)
        current_score = score_from_line_counts(current_line_counts)
        if current_score < 0:
            return current_points, current_score, current_line_counts
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        best_local_line_counts = dict(current_line_counts)
        
        iter_count = 0
        no_improve = 0
        
        while iter_count < max_iters:
            if time.time() > time_limit:
                break
            iter_count += 1
            
            i, j = random.sample(range(16), 2)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            
            # Swap
            current_points[i] = old_j
            current_points[j] = old_i
            
            # Check distinctness
            is_distinct = True
            seen = set()
            for p in current_points:
                if tuple(p) in seen:
                    is_distinct = False
                    break
                seen.add(tuple(p))
            if not is_distinct:
                current_points[i] = old_i
                current_points[j] = old_j
                no_improve += 1
                continue
            
            # Recompute line counts (swap affects lines involving i or j)
            # For efficiency, recompute all lines involving i or j
            new_line_counts = dict(current_line_counts)
            
            # Remove old lines involving i or j
            for k in range(16):
                if k == i or k == j:
                    continue
                # Line (i, k) and (j, k)
                key_ik = line_hash(old_i, current_points[k])
                key_jk = line_hash(old_j, current_points[k])
                new_line_counts[key_ik] -= 1
                if new_line_counts[key_ik] == 0:
                    del new_line_counts[key_ik]
                new_line_counts[key_jk] -= 1
                if new_line_counts[key_jk] == 0:
                    del new_line_counts[key_jk]
            
            # Remove line (i, j)
            key_ij = line_hash(old_i, old_j)
            new_line_counts[key_ij] -= 1
            if new_line_counts[key_ij] == 0:
                del new_line_counts[key_ij]
            
            # Add new lines
            for k in range(16):
                if k == i or k == j:
                    continue
                key_ik_new = line_hash(old_j, current_points[k])  # i now has old_j
                key_jk_new = line_hash(old_i, current_points[k])  # j now has old_i
                if key_ik_new in new_line_counts:
                    new_line_counts[key_ik_new] += 1
                else:
                    new_line_counts[key_ik_new] = 1
                if key_jk_new in new_line_counts:
                    new_line_counts[key_jk_new] += 1
                else:
                    new_line_counts[key_jk_new] = 1
            
            # Add new line (i, j)
            key_ij_new = line_hash(old_j, old_i)
            if key_ij_new in new_line_counts:
                new_line_counts[key_ij_new] += 1
            else:
                new_line_counts[key_ij_new] = 1
            
            new_score = score_from_line_counts(new_line_counts)
            
            if new_score == -1:
                current_points[i] = old_i
                current_points[j] = old_j
                no_improve += 1
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                current_line_counts = new_line_counts
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = [p[:] for p in current_points]
                    best_local_line_counts = dict(current_line_counts)
            else:
                current_points[i] = old_i
                current_points[j] = old_j
                no_improve += 1
            
            if no_improve > 5000:
                break
        
        return best_local, best_local_score, best_local_line_counts
    
    # Phase 1: Optimize the initial grid with single-point moves
    pts, score, lc = local_search(initial_points, max_iters=10000, time_limit=deadline - 80.0)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
        best_line_counts = lc
    
    if best_score >= 15:
        return best_points
    
    # Phase 2: Try pairwise swaps
    if time.time() < deadline - 60.0:
        pts, score, lc = pairwise_swap_search(best_points, best_line_counts, max_iters=5000, time_limit=deadline - 60.0)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            best_line_counts = lc
    
    if best_score >= 15:
        return best_points
    
    # Phase 3: More single-point optimization
    if time.time() < deadline - 40.0:
        pts, score, lc = local_search(best_points, max_iters=10000, time_limit=deadline - 40.0)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            best_line_counts = lc
    
    if best_score >= 15:
        return best_points
    
    # Phase 4: Random restarts with diverse configurations
    restart_attempts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10.0 and restart_attempts < max_restarts:
        restart_attempts += 1
        
        mode = random.randint(0, 5)
        
        if mode == 0:
            # Perturbed 4x4 grid
            pts = [p[:] for p in initial_points]
            for i in range(16):
                if random.random() < 0.5:
                    pts[i] = [pts[i][0] + random.randint(-3, 3), pts[i][1] + random.randint(-3, 3)]
        elif mode == 1:
            # Random in small box
            pts = []
            while len(pts) < 16:
                x = random.randint(-6, 6)
                y = random.randint(-6, 6)
                if [x, y] not in pts:
                    pts.append([x, y])
        elif mode == 2:
            # Random in medium box
            pts = []
            while len(pts) < 16:
                x = random.randint(-15, 15)
                y = random.randint(-15, 15)
                if [x, y] not in pts:
                    pts.append([x, y])
        elif mode == 3:
            # Structured: shifted grid
            pts = []
            base = random.randint(-3, 3)
            shift = random.randint(1, 3)
            for x in range(4):
                for y in range(4):
                    px = x + (shift if x >= 2 else 0)
                    py = y + (shift if y >= 2 else 0)
                    pts.append([base + px, base + py])
        elif mode == 4:
            # Grid with one row/column shifted
            pts = []
            shift = random.randint(1, 4)
            axis = random.choice(['x', 'y'])
            pos = random.randint(0, 3)
            for x in range(4):
                for y in range(4):
                    if axis == 'x' and x == pos:
                        pts.append([x + shift, y])
                    elif axis == 'y' and y == pos:
                        pts.append([x, y + shift])
                    else:
                        pts.append([x, y])
        else:
            # Two clusters
            pts = []
            cx1, cy1 = random.randint(-10, 10), random.randint(-10, 10)
            cx2, cy2 = random.randint(-10, 10), random.randint(-10, 10)
            for i in range(8):
                pts.append([cx1 + random.randint(-3, 3), cy1 + random.randint(-3, 3)])
            for i in range(8):
                pts.append([cx2 + random.randint(-3, 3), cy2 + random.randint(-3, 3)])
            # Ensure distinct
            unique_pts = []
            for p in pts:
                if p not in unique_pts:
                    unique_pts.append(p)
            while len(unique_pts) < 16:
                p = [random.randint(-20, 20), random.randint(-20, 20)]
                if p not in unique_pts:
                    unique_pts.append(p)
            pts = unique_pts[:16]
        
        s = evaluate(pts)
        if s < 0:
            continue
        
        # Run local search
        remaining = deadline - 10.0 - time.time()
        if remaining < 5.0:
            break
        iters = max(500, min(int(remaining * 20), 5000))
        
        pts, score, lc = local_search(pts, max_iters=iters, time_limit=deadline - 10.0)
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            best_line_counts = lc
            
        if best_score >= 15:
            break
    
    # Phase 5: Final polish
    if time.time() < deadline - 3.0:
        pts, score, lc = local_search(best_points, max_iters=10000, time_limit=deadline - 2.0)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            best_line_counts = lc
    
    return best_points
