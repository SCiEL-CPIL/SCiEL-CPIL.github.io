import time
import random
import numpy as np
from collections import Counter

def _fast_score(points):
    """Calculate max multiplicity of a positive doubled triangle area."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)
    
    for i in range(n):
        V = P[i+1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)
    
    return int(np.max(counts))

def _score_and_best_area(points):
    """Return (max_multiplicity, best_area_value)."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)
    
    for i in range(n):
        V = P[i+1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)
    
    best_area = int(np.argmax(counts))
    return int(counts[best_area]), best_area

def _area_counts(points):
    """Return full area count array."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)
    
    for i in range(n):
        V = P[i+1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i]
        vj = V[idx_j]
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        mask = (areas > 0) & (areas < max_area)
        if np.any(mask):
            counts += np.bincount(areas[mask], minlength=max_area)
    
    return counts

def _generate_grid_candidates():
    """Generate various grid and structured candidates."""
    candidates = []
    
    # Core grids
    g1 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    candidates.append(g1)
    
    g2 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]
    candidates.append(g2)
    
    g3 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
    candidates.append(g3)
    
    g4 = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]
    candidates.append(g4)
    
    # Sheared variants
    g5 = [(x, y + x) for x, y in g1]
    valid = [p for p in g5 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid)) == 30:
        candidates.append(valid)
    
    g6 = [(x + y, y) for x, y in g2]
    valid = [p for p in g6 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid)) == 30:
        candidates.append(valid)
    
    # More sheared variants
    for bg in [g1, g2]:
        for sx in range(-2, 3):
            for sy in range(-2, 3):
                if sx == 0 and sy == 0:
                    continue
                sheared = [(x + sx * y, y + sy * x) for x, y in bg]
                valid = [p for p in sheared if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
                if len(set(valid)) == 30:
                    candidates.append(valid)
    
    # Parabolic
    for shift in range(-5, 6):
        pts = []
        for x in range(-5, 6):
            y = x * x + shift
            if -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) >= 30:
            pts = pts[:30]
            if len(set(pts)) == 30:
                candidates.append(pts)
        elif len(pts) == 30:
            candidates.append(pts)
    
    for a in [1, 2]:
        for shift_x in range(-3, 4):
            for shift_y in range(-10, 11):
                pts = []
                for x in range(-6, 7):
                    y = a * x * x + shift_y
                    px = x + shift_x
                    py = y
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if len(set(pts)) == 30:
                        candidates.append(pts)
                elif len(pts) == 30:
                    candidates.append(pts)
    
    # Lattice with period
    for px, py in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4)]:
        pts = []
        for x in range(-15, 16, px):
            for y in range(-15, 16, py):
                pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) >= 30:
                break
        if len(pts) == 30 and len(set(pts)) == 30:
            candidates.append(pts)
    
    # Random
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for seed in [42, 123, 777, 2024]:
        random.seed(seed)
        pts = random.sample(all_coords, 30)
        candidates.append(pts)
    
    # Random clustered
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed)
        cx = random.randint(-15, 15)
        cy = random.randint(-15, 15)
        pts = []
        for _ in range(30):
            x = cx + random.randint(-8, 8)
            y = cy + random.randint(-8, 8)
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.append((x, y))
        if len(pts) == 30:
            candidates.append(pts)
    
    return candidates

def _find_targeted_positions(points, target_area, occupied, num_candidates=20):
    """Find candidate positions that would create many triangles of target_area."""
    P = np.array(points, dtype=np.int64)
    n = len(P)
    candidates = []
    
    # For each pair of existing points, find positions that make area = target_area
    # |cross(p1-p0, p2-p0)| = target_area
    # This is a line of possible positions
    # Sample a few positions along these lines
    
    pairs_sampled = min(n * (n-1) // 2, 200)
    pair_indices = np.random.choice(n * (n-1) // 2, size=min(pairs_sampled, n*(n-1)//2), replace=False)
    
    pair_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_list.append((i, j))
            idx += 1
    
    for pi in pair_indices:
        i, j = pair_list[pi]
        p1 = P[i]
        p2 = P[j]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-9:
            continue
        
        # Perpendicular direction
        px = -dy / length
        py = dx / length
        
        # Distance from line for target area: area = 0.5 * base * height
        # doubled_area = base * height => height = target_area / length
        h = target_area / length
        
        # Two parallel lines at distance h from the segment
        for sign in [1, -1]:
            # Points on the line at distance h
            # Sample a few x positions
            for t in np.linspace(-5, 5, 11):
                # Point on the line: p1 + t*(dx,dy)/length + sign*h*(px,py)
                bx = p1[0] + t * dx / length + sign * h * px
                by = p1[1] + t * dy / length + sign * h * py
                nx = int(round(bx))
                ny = int(round(by))
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    np_ = (nx, ny)
                    if np_ not in occupied:
                        # Verify the area is exactly target_area
                        v1 = np.array([p1[0]-nx, p1[1]-ny])
                        v2 = np.array([p2[0]-nx, p2[1]-ny])
                        cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                        if cross == target_area:
                            candidates.append(np_)
    
    # Deduplicate and limit
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    
    if len(unique) > num_candidates:
        unique = random.sample(unique, num_candidates)
    
    return unique

def _local_search(points, time_limit, target_area=None):
    """Local search with targeted area boosting."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, current_best_area = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    
    T = 5.0
    cooling = 0.997
    min_T = 0.05
    occupied = set(points)
    
    target_a = target_area if target_area is not None else current_best_area
    targeted_counter = 0
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 29)
        old_p = points[idx]
        
        # 30% chance to use targeted move
        use_targeted = (random.random() < 0.3 and target_a > 0)
        
        found_move = False
        new_p = None
        
        if use_targeted:
            # Find targeted positions
            temp_points = points[:]
            temp_points[idx] = None  # placeholder
            # Remove idx from occupied temporarily for checking
            occ_copy = occupied - {old_p}
            
            targeted = _find_targeted_positions(
                [p for p in points if p is not None], 
                target_a, 
                occ_copy, 
                num_candidates=5
            )
            
            for tp in targeted:
                if tp != old_p and tp not in occupied:
                    new_p = tp
                    found_move = True
                    break
        
        if not found_move:
            # Standard random move
            for _ in range(8):
                if random.random() < 0.5:
                    ref_idx = random.randint(0, 29)
                    rx, ry = points[ref_idx]
                    new_x = rx + random.randint(-4, 4)
                    new_y = ry + random.randint(-4, 4)
                else:
                    new_x = random.randint(-30, 30)
                    new_y = random.randint(-30, 30)
                new_p = (new_x, new_y)
                if new_p == old_p or new_p in occupied:
                    continue
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                found_move = True
                break
        
        if not found_move:
            continue
        
        points[idx] = new_p
        occupied.remove(old_p)
        occupied.add(new_p)
        
        new_score, new_best_area = _score_and_best_area(points)
        delta = new_score - current_score
        
        # Update target area
        if new_best_area != target_a and new_score >= current_score:
            target_a = new_best_area
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            current_best_area = new_best_area
            if new_score > best_score:
                best_score = new_score
                best_points = list(points)
        else:
            points[idx] = old_p
            occupied.remove(new_p)
            occupied.add(old_p)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_points, best_score

def _polish(points, time_limit):
    """Focused polish with area-lock strategy."""
    start_time = time.time()
    points = [tuple(p) for p in points]
    current_score, best_area = _score_and_best_area(points)
    best_points = list(points)
    best_score = current_score
    occupied = set(points)
    
    # Phase 1: Standard multi-radius polish
    for radius in [2, 3, 4]:
        if time.time() - start_time > time_limit * 0.5:
            break
        for idx in range(30):
            if time.time() - start_time > time_limit * 0.5:
                break
            old_p = points[idx]
            ox, oy = old_p
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_p = (nx, ny)
                    if new_p in occupied:
                        continue
                    
                    points[idx] = new_p
                    occupied.remove(old_p)
                    occupied.add(new_p)
                    
                    new_score, _ = _score_and_best_area(points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = list(points)
                        current_score = new_score
                    else:
                        points[idx] = old_p
                        occupied.remove(new_p)
                        occupied.add(old_p)
    
    # Phase 2: Area-lock polish - fix the best area, only accept if count doesn't drop
    if time.time() - start_time < time_limit:
        counts = _area_counts(best_points)
        locked_area = int(np.argmax(counts))
        locked_count = int(counts[locked_area])
        
        points = list(best_points)
        occupied = set(points)
        
        for idx in range(30):
            if time.time() - start_time > time_limit:
                break
            old_p = points[idx]
            ox, oy = old_p
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_p = (nx, ny)
                    if new_p in occupied:
                        continue
                    
                    points[idx] = new_p
                    occupied.remove(old_p)
                    occupied.add(new_p)
                    
                    new_counts = _area_counts(points)
                    new_locked = int(new_counts[locked_area])
                    new_max = int(np.max(new_counts))
                    
                    # Accept if locked area count improves or max improves
                    if new_locked > locked_count or new_max > best_score:
                        best_score = max(new_max, best_score)
                        best_points = list(points)
                        if new_locked > locked_count:
                            locked_count = new_locked
                    else:
                        points[idx] = old_p
                        occupied.remove(new_p)
                        occupied.add(old_p)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    candidates = _generate_grid_candidates()
    
    scored = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s, _ = _score_and_best_area(c)
        scored.append((s, c))
    
    if not scored:
        return [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_points = [tuple(p) for p in scored[0][1]]
    best_score = scored[0][0]
    
    used_keys = set()
    top_n = min(5, len(scored))
    
    for i in range(top_n):
        if time.time() > deadline - 3.0:
            break
        s, c = scored[i]
        key = tuple(sorted(c))
        if key in used_keys:
            continue
        used_keys.add(key)
        
        remaining = deadline - time.time() - 2.0
        if remaining < 2.0:
            break
        
        if i == 0:
            t_limit = min(remaining * 0.35, 16.0)
        elif i == 1:
            t_limit = min(remaining * 0.25, 12.0)
        elif i == 2:
            t_limit = min(remaining * 0.18, 9.0)
        else:
            t_limit = min(remaining * 0.12, 6.0)
        
        t_limit = max(1.0, t_limit)
        
        _, target_a = _score_and_best_area(c)
        fp, fs = _local_search(c, time_limit=t_limit, target_area=target_a)
        if fs > best_score:
            best_score = fs
            best_points = fp
    
    # Polish
    if time.time() < deadline - 2.0:
        polish_time = min(deadline - time.time() - 1.0, 14.0)
        if polish_time > 1.0:
            fp, fs = _polish(best_points, time_limit=polish_time)
            if fs > best_score:
                best_score = fs
                best_points = fp
    
    # Final quick polish
    if time.time() < deadline - 0.5:
        final_time = min(deadline - time.time() - 0.3, 4.0)
        if final_time > 0.5:
            fp, fs = _polish(best_points, time_limit=final_time)
            if fs > best_score:
                best_score = fs
                best_points = fp
    
    return [tuple(p) for p in best_points]

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_fast_score(pts)}")
    print(f"Points: {pts}")
