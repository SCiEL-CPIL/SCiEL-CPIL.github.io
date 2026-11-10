import math
import time
import random
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = A * x1 + B * y1
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _build_lines(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _score_from_lines(lines):
    score = 0
    for key, indices in lines.items():
        cnt = len(indices)
        if cnt >= 5:
            return -1, False
        elif cnt == 4:
            score += 1
    return score, True

def _score(pts):
    lines = _build_lines(pts)
    return _score_from_lines(lines)

def _delta_score(pts, p_idx, old_pos, new_pos, lines):
    """Compute new score after moving point p_idx from old_pos to new_pos.
    Only recompute lines involving p_idx."""
    affected_keys = set()
    for j in range(16):
        if j == p_idx:
            continue
        key_old = _line_key(old_pos, pts[j])
        key_new = _line_key(new_pos, pts[j])
        affected_keys.add(key_old)
        affected_keys.add(key_new)
    
    new_lines = dict(lines)
    
    for j in range(16):
        if j == p_idx:
            continue
        key_old = _line_key(old_pos, pts[j])
        if key_old in new_lines:
            new_lines[key_old].discard(p_idx)
            if not new_lines[key_old]:
                del new_lines[key_old]
    
    for j in range(16):
        if j == p_idx:
            continue
        key_new = _line_key(new_pos, pts[j])
        if key_new not in new_lines:
            new_lines[key_new] = set()
        new_lines[key_new].add(p_idx)
        new_lines[key_new].add(j)
    
    for key, indices in new_lines.items():
        if len(indices) >= 5:
            return -1, False
    
    score = 0
    for key, indices in new_lines.items():
        if len(indices) == 4:
            score += 1
    
    return score, True

def _delta_score_swap(pts, i, j, lines):
    """Compute new score after swapping points i and j."""
    old_i = tuple(pts[i])
    old_j = tuple(pts[j])
    
    new_lines = dict(lines)
    
    # Remove i and j from all lines
    for key in list(new_lines.keys()):
        new_lines[key].discard(i)
        new_lines[key].discard(j)
        if not new_lines[key]:
            del new_lines[key]
    
    # Add i at old_j position and j at old_i position
    for k in range(16):
        if k == i or k == j:
            continue
        # Point i is now at old_j
        key_ij = _line_key(old_j, pts[k])
        if key_ij not in new_lines:
            new_lines[key_ij] = set()
        new_lines[key_ij].add(i)
        new_lines[key_ij].add(k)
        
        # Point j is now at old_i
        key_ji = _line_key(old_i, pts[k])
        if key_ji not in new_lines:
            new_lines[key_ji] = set()
        new_lines[key_ji].add(j)
        new_lines[key_ji].add(k)
    
    # Line between i and j
    key_ij_line = _line_key(old_j, old_i)
    if key_ij_line not in new_lines:
        new_lines[key_ij_line] = set()
    new_lines[key_ij_line].add(i)
    new_lines[key_ij_line].add(j)
    
    for key, indices in new_lines.items():
        if len(indices) >= 5:
            return -1, False
    
    score = 0
    for key, indices in new_lines.items():
        if len(indices) == 4:
            score += 1
    
    return score, True

def _generate_initial(seed=0):
    rng = random.Random(seed)
    pts = []
    for x in range(4):
        for y in range(4):
            px = x + rng.choice([0, 0, 0, 1, -1])
            py = y + rng.choice([0, 0, 0, 1, -1])
            pts.append([px, py])
    seen = set()
    unique_pts = []
    for p in pts:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append(p)
    while len(unique_pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        tp = (x, y)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append([x, y])
    return unique_pts[:16]

def _generate_structured_start(seed=0):
    """Generate a structured starting configuration from a 5x5 or 6x6 grid subset."""
    rng = random.Random(seed)
    grid_size = rng.choice([5, 6])
    all_pts = []
    for x in range(grid_size):
        for y in range(grid_size):
            all_pts.append((x, y))
    rng.shuffle(all_pts)
    pts = [list(p) for p in all_pts[:16]]
    return pts

def _generate_grid_subset(seed=0, grid_size=5):
    """Pick 16 points from a grid_size x grid_size grid with some randomness."""
    rng = random.Random(seed)
    all_pts = []
    for x in range(grid_size):
        for y in range(grid_size):
            all_pts.append((x, y))
    rng.shuffle(all_pts)
    pts = [list(p) for p in all_pts[:16]]
    return pts

def _simulated_annealing(pts, deadline, max_iters=5000):
    """Simulated annealing on a grid-based configuration."""
    rng = random.Random(int(time.time() * 1000) % 100000)
    cur_pts = [list(p) for p in pts]
    cur_score, cur_valid = _score(cur_pts)
    if not cur_valid:
        cur_score = -1
    
    best_pts = [list(p) for p in cur_pts]
    best_score = cur_score
    
    T = 5.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random point and move it to a random nearby position
        p_idx = rng.randint(0, 15)
        old_pos = tuple(cur_pts[p_idx])
        
        # Try several candidate moves
        best_move = None
        best_delta = 0
        
        for _ in range(8):
            xi = old_pos[0] + rng.randint(-2, 2)
            yi = old_pos[1] + rng.randint(-2, 2)
            cand = (xi, yi)
            if cand == old_pos:
                continue
            if cand in set(map(tuple, cur_pts)):
                continue
            if not (-1000 <= xi <= 1000 and -1000 <= yi <= 1000):
                continue
            
            lines = _build_lines(cur_pts)
            new_score, new_valid = _delta_score(cur_pts, p_idx, old_pos, cand, lines)
            if not new_valid:
                continue
            
            delta = new_score - cur_score
            if delta > best_delta:
                best_delta = delta
                best_move = cand
        
        if best_move is not None:
            if best_delta > 0 or rng.random() < math.exp(best_delta / T):
                cur_pts[p_idx] = list(best_move)
                cur_score += best_delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in cur_pts]
        
        T *= cooling
    
    return best_pts, best_score

def _local_search(cur_pts, cur_score, deadline, max_iters=300):
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        lines = _build_lines(cur_pts)
        cur_score, cur_valid = _score_from_lines(lines)
        if not cur_valid:
            for key, indices in lines.items():
                if len(indices) >= 5:
                    for idx in indices:
                        if time.time() > deadline:
                            return cur_pts, cur_score, False
                        old_pos = tuple(cur_pts[idx])
                        found = False
                        for attempt in range(30):
                            xi = old_pos[0] + random.randint(-3, 3)
                            yi = old_pos[1] + random.randint(-3, 3)
                            if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                                cand = (xi, yi)
                                if cand != old_pos and cand not in set(map(tuple, cur_pts)):
                                    new_score, new_valid = _delta_score(cur_pts, idx, old_pos, cand, lines)
                                    if new_valid:
                                        cur_pts[idx] = list(cand)
                                        lines = _build_lines(cur_pts)
                                        cur_score, cur_valid = _score_from_lines(lines)
                                        found = True
                                        break
                        if found:
                            improved = True
                            break
                    break
        
        if time.time() > deadline:
            return cur_pts, cur_score, False
        
        three_lines = {k: v for k, v in lines.items() if len(v) == 3}
        
        for p_idx in range(16):
            if time.time() > deadline:
                return cur_pts, cur_score, False
            
            current_pos = tuple(cur_pts[p_idx])
            candidates = set()
            
            # Intersections of pairs of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            for i in range(len(tl_keys)):
                for j in range(i + 1, len(tl_keys)):
                    A1, B1, C1 = tl_keys[i]
                    A2, B2, C2 = tl_keys[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            # Points on 3-lines extended
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
                idxs = list(v)
                p1 = cur_pts[idxs[0]]
                p2 = cur_pts[idxs[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                for t in range(-5, 6):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # Random nearby
            for _ in range(20):
                xi = current_pos[0] + random.randint(-6, 6)
                yi = current_pos[1] + random.randint(-6, 6)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            # Intersections of 2-lines
            two_lines = {k: v for k, v in lines.items() if len(v) == 2 and p_idx not in v}
            tl_keys2 = list(two_lines.keys())
            if len(tl_keys2) > 40:
                random.shuffle(tl_keys2)
                tl_keys2 = tl_keys2[:40]
            for i in range(len(tl_keys2)):
                for j in range(i + 1, len(tl_keys2)):
                    A1, B1, C1 = tl_keys2[i]
                    A2, B2, C2 = tl_keys2[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            best_new_pos = None
            best_new_score = cur_score
            current_set = set(map(tuple, cur_pts))
            
            for cand in candidates:
                if cand == current_pos or cand in current_set:
                    continue
                
                new_score, new_valid = _delta_score(cur_pts, p_idx, current_pos, cand, lines)
                
                if new_valid and new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
            
            if best_new_pos is not None:
                cur_pts[p_idx] = list(best_new_pos)
                cur_score = best_new_score
                improved = True
                lines = _build_lines(cur_pts)
    
    return cur_pts, cur_score, improved

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = None
    best_score = -1
    
    # Try multiple initial seeds with different strategies
    for seed in range(20):
        if time.time() > deadline:
            break
        
        if seed < 6:
            cur_pts = _generate_initial(seed)
        elif seed < 12:
            cur_pts = _generate_structured_start(seed)
        else:
            cur_pts = _generate_grid_subset(seed, 6)
        
        cur_score, cur_valid = _score(cur_pts)
        
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        if cur_valid:
            cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
        
        # Simulated annealing for some seeds
        if seed % 3 == 0 and time.time() < deadline:
            sa_pts, sa_score = _simulated_annealing(cur_pts, deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
    
    # If still no good result, use simple grid
    if best_pts is None or best_score < 10:
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, y])
        best_pts = pts
        best_score, _ = _score(best_pts)
    
    return best_pts
