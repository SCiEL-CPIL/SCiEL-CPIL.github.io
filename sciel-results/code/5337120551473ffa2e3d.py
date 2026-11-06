<<<<<<< SEARCH
"""Optimized search for 16 integer points maximizing lines with exactly 4 points."""
import random
import math
import time

def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _make_pair_info(pts):
    """Precompute line keys and per-pair other-point sets for 16 points."""
    n = len(pts)
    keys = [[None] * n for _ in range(n)]
    others = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Canonicalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Normal vector (perpendicular)
            nx, ny = -dy, dx
            c = nx * x1 + ny * y1
            key = (dx, dy, nx, ny, c)
            keys[i][j] = key
            keys[j][i] = key
            for k in range(n):
                if k != i and k != j:
                    others[i][j].add(k)
                    others[j][i].add(k)
    return keys, others

def _score(pts, keys, others):
    """Count lines with exactly 4 points; return (count, is_valid)."""
    n = len(pts)
    key_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = keys[i][j]
            key_count[k] = key_count.get(k, 0) + 1
    exact4 = 0
    valid = True
    for k, cnt in key_count.items():
        if cnt == 6:  # C(4,2) = 6 pairs for 4 collinear points
            exact4 += 1
        elif cnt > 6:
            # More than 4 points on this line
            valid = False
    # Also check: could have a line with exactly 4 points but also check
    # that no line has exactly 5 points (C(5,2)=10 pairs) or 6 points (C(6,2)=15)
    # cnt > 6 catches all of those since 5 points -> 10 pairs, 6 -> 15, etc.
    # But wait: what about a line with exactly 4 points where some pairs are
    # also on other lines? No, each pair defines exactly one line.
    return exact4, valid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    # Start with a 4x4 grid
    pts = [[x, y] for x in range(4) for y in range(4)]
    n = len(pts)
    
    # Precompute pair info
    keys, others = _make_pair_info(pts)
    
    # Evaluate initial state
    score, valid = _score(pts, keys, others)
    
    # Local search to fix validity: move points on 5-collinear lines
    # First, let's just run a quick local search to get a valid starting point
    max_local_iters = 5000
    for it in range(max_local_iters):
        if time.time() > deadline:
            break
        if valid and score >= 4:
            break
        # Find a point to move
        if not valid:
            # Find lines with 5+ points
            key_count = {}
            for i in range(n):
                for j in range(i + 1, n):
                    k = keys[i][j]
                    key_count[k] = key_count.get(k, 0) + 1
            bad_keys = [k for k, cnt in key_count.items() if cnt > 6]
            if not bad_keys:
                break
            # Pick a random bad key and move one of its points
            bk = random.choice(bad_keys)
            # Find all pairs with this key
            pair_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if keys[i][j] == bk:
                        pair_list.append((i, j))
            # Pick a random pair and move one endpoint
            pi, pj = random.choice(pair_list)
            move_idx = random.choice([pi, pj])
        else:
            move_idx = random.randint(0, n - 1)
        
        # Try a few random moves
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Move to a nearby point
            ox, oy = pts[move_idx]
            # Random offset in [-3, 3]
            new_x = ox + random.randint(-3, 3)
            new_y = oy + random.randint(-3, 3)
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
            # Check distinctness
            if any(new_x == pts[k][0] and new_y == pts[k][1] for k in range(n) if k != move_idx):
                continue
            
            old = pts[move_idx]
            pts[move_idx] = [new_x, new_y]
            
            # Recompute pair info for this point
            for j in range(n):
                if j == move_idx:
                    continue
                x1, y1 = pts[move_idx]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = _gcd(dx, dy)
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                nx, ny = -dy, dx
                c = nx * x1 + ny * y1
                key = (dx, dy, nx, ny, c)
                keys[move_idx][j] = key
                keys[j][move_idx] = key
                others[move_idx][j] = set()
                others[j][move_idx] = set()
            
            for k in range(n):
                if k == move_idx:
                    continue
                for l in range(n):
                    if l == move_idx or l == k:
                        continue
                    others[move_idx][k].add(l)
                    others[k][move_idx].add(l)
            
            new_score, new_valid = _score(pts, keys, others)
            
            if new_valid and (not valid or new_score > score):
                valid = new_valid
                score = new_score
                break
            else:
                # Revert
                pts[move_idx] = old
                for j in range(n):
                    if j == move_idx:
                        continue
                    x1, y1 = pts[move_idx]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    g = _gcd(dx, dy)
                    if g == 0:
                        continue
                    dx //= g
                    dy //= g
                    if dx < 0 or (dx == 0 and dy < 0):
                        dx = -dx
                        dy = -dy
                    nx, ny = -dy, dx
                    c = nx * x1 + ny * y1
                    key = (dx, dy, nx, ny, c)
                    keys[move_idx][j] = key
                    keys[j][move_idx] = key
                    others[move_idx][j] = set()
                    others[j][move_idx] = set()
                for k in range(n):
                    if k == move_idx:
                        continue
                    for l in range(n):
                        if l == move_idx or l == k:
                            continue
                        others[move_idx][k].add(l)
                        others[k][move_idx].add(l)
    
    # Now run simulated annealing from this valid starting point
    if not valid:
        # Fallback: just return the grid (shouldn't happen)
        return [[x, y] for x in range(4) for y in range(4)]
    
    best_score = score
    best_pts = [p[:] for p in pts]
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        ox, oy = pts[idx]
        
        # Generate a new position
        # Sometimes do a small move, sometimes a larger one
        if random.random() < 0.7:
            new_x = ox + random.randint(-4, 4)
            new_y = oy + random.randint(-4, 4)
        else:
            new_x = ox + random.randint(-15, 15)
            new_y = oy + random.randint(-15, 15)
        
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
        
        # Check distinctness
        if any(new_x == pts[k][0] and new_y == pts[k][1] for k in range(n) if k != idx):
            continue
        
        old = pts[idx]
        pts[idx] = [new_x, new_y]
        
        # Recompute pair info for this point
        for j in range(n):
            if j == idx:
                continue
            x1, y1 = pts[idx]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            nx, ny = -dy, dx
            c = nx * x1 + ny * y1
            key = (dx, dy, nx, ny, c)
            keys[idx][j] = key
            keys[j][idx] = key
            others[idx][j] = set()
            others[j][idx] = set()
        
        for k in range(n):
            if k == idx:
                continue
            for l in range(n):
                if l == idx or l == k:
                    continue
                others[idx][k].add(l)
                others[k][idx].add(l)
        
        new_score, new_valid = _score(pts, keys, others)
        
        if new_valid:
            if new_score > score:
                # Accept
                score = new_score
                if score > best_score:
                    best_score = score
                    best_pts = [p[:] for p in pts]
            else:
                # Maybe accept worse
                delta = score - new_score
                if random.random() < math.exp(-delta / max(T, 1e-9)):
                    score = new_score
                else:
                    # Revert
                    pts[idx] = old
                    for j in range(n):
                        if j == idx:
                            continue
                        x1, y1 = pts[idx]
                        x2, y2 = pts[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        g = _gcd(dx, dy)
                        if g == 0:
                            continue
                        dx //= g
                        dy //= g
                        if dx < 0 or (dx == 0 and dy < 0):
                            dx = -dx
                            dy = -dy
                        nx, ny = -dy, dx
                        c = nx * x1 + ny * y1
                        key = (dx, dy, nx, ny, c)
                        keys[idx][j] = key
                        keys[j][idx] = key
                        others[idx][j] = set()
                        others[j][idx] = set()
                    for k in range(n):
                        if k == idx:
                            continue
                        for l in range(n):
                            if l == idx or l == k:
                                continue
                            others[idx][k].add(l)
                            others[k][idx].add(l)
        else:
            # Invalid: revert
            pts[idx] = old
            for j in range(n):
                if j == idx:
                    continue
                x1, y1 = pts[idx]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = _gcd(dx, dy)
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                nx, ny = -dy, dx
                c = nx * x1 + ny * y1
                key = (dx, dy, nx, ny, c)
                keys[idx][j] = key
                keys[j][idx] = key
                others[idx][j] = set()
                others[j][idx] = set()
            for k in range(n):
                if k == idx:
                    continue
                for l in range(n):
                    if l == idx or l == k:
                        continue
                    others[idx][k].add(l)
                    others[k][idx].add(l)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return best found
    return best_pts
=======
"""Optimized search for 16 integer points maximizing lines with exactly 4 points."""
import random
import math
import time
from collections import defaultdict

def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Compute canonical line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    nx, ny = -dy, dx
    c = nx * x1 + ny * y1
    return (dx, dy, nx, ny, c)

def _compute_line_counts(pts):
    """Compute line counts and validity. Returns (exact4_count, valid, line_to_points_map)."""
    n = len(pts)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            if key is not None:
                line_points[key].add(i)
                line_points[key].add(j)
    
    exact4 = 0
    valid = True
    for key, pts_on_line in line_points.items():
        cnt = len(pts_on_line)
        if cnt == 4:
            exact4 += 1
        elif cnt > 4:
            valid = False
    return exact4, valid, line_points

def _incremental_update(pts, moved_idx, old_pos, line_points):
    """
    Update line_points after moving point moved_idx from old_pos to pts[moved_idx].
    Returns updated line_points, new_exact4, new_valid.
    """
    n = len(pts)
    # Remove old lines involving moved_idx
    old_lines = set()
    for j in range(n):
        if j == moved_idx:
            continue
        key = _line_key(old_pos, pts[j])
        if key is not None:
            old_lines.add(key)
    
    # For each old line, remove moved_idx
    for key in old_lines:
        if key in line_points:
            line_points[key].discard(moved_idx)
            if not line_points[key]:
                del line_points[key]
    
    # Add new lines involving moved_idx
    for j in range(n):
        if j == moved_idx:
            continue
        key = _line_key(pts[moved_idx], pts[j])
        if key is not None:
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(moved_idx)
            line_points[key].add(j)
    
    # Compute new score
    exact4 = 0
    valid = True
    for key, pts_on_line in line_points.items():
        cnt = len(pts_on_line)
        if cnt == 4:
            exact4 += 1
        elif cnt > 4:
            valid = False
    return line_points, exact4, valid

def _try_move(pts, idx, new_pos, line_points, current_score, current_valid):
    """
    Try moving point idx to new_pos. Returns (accepted, new_score, new_valid, line_points).
    """
    n = len(pts)
    # Check bounds
    if not (-1000 <= new_pos[0] <= 1000 and -1000 <= new_pos[1] <= 1000):
        return False, current_score, current_valid, line_points
    
    # Check distinctness
    for k in range(n):
        if k != idx and pts[k][0] == new_pos[0] and pts[k][1] == new_pos[1]:
            return False, current_score, current_valid, line_points
    
    old_pos = pts[idx][:]
    pts[idx] = new_pos[:]
    
    new_line_points, new_score, new_valid = _incremental_update(pts, idx, old_pos, line_points)
    
    return True, new_score, new_valid, new_line_points

def _generate_candidates(pts, idx, n):
    """Generate candidate positions for point idx."""
    ox, oy = pts[idx]
    candidates = []
    
    # Small moves
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == 0 and dy == 0:
                continue
            candidates.append((ox + dx, oy + dy))
    
    # Medium moves
    for _ in range(10):
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        candidates.append((ox + dx, oy + dy))
    
    # Large moves (occasionally)
    for _ in range(3):
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        candidates.append((ox + dx, oy + dy))
    
    return candidates

def _local_search(pts, line_points, score, valid, deadline, max_iters=2000):
    """Local search to improve score while maintaining validity."""
    n = len(pts)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if not valid:
            # Find and fix violations
            bad_lines = [key for key, pts_on_line in line_points.items() if len(pts_on_line) > 4]
            if not bad_lines:
                valid = True
                continue
            # Pick a random bad line and move one of its points
            bad_key = random.choice(bad_lines)
            pts_on_bad = list(line_points[bad_key])
            move_idx = random.choice(pts_on_bad)
            improved = False
            for attempt in range(30):
                if time.time() > deadline:
                    break
                new_x = pts[move_idx][0] + random.randint(-10, 10)
                new_y = pts[move_idx][1] + random.randint(-10, 10)
                new_pos = (new_x, new_y)
                accepted, new_score, new_valid, new_lp = _try_move(pts, move_idx, new_pos, line_points, score, valid)
                if accepted and new_valid and new_score >= score:
                    line_points = new_lp
                    score = new_score
                    valid = new_valid
                    improved = True
                    break
                elif accepted:
                    # Revert
                    old_pos = (pts[move_idx][0] - (new_x - pts[move_idx][0]), pts[move_idx][1] - (new_y - pts[make_idx][1])) if False else None
                    # Actually we need to properly revert - let's just skip this
                    pass
            if not improved:
                continue
        else:
            # Try to improve score
            move_idx = random.randint(0, n - 1)
            improved = False
            for attempt in range(20):
                if time.time() > deadline:
                    break
                new_x = pts[move_idx][0] + random.randint(-8, 8)
                new_y = pts[move_idx][1] + random.randint(-8, 8)
                new_pos = (new_x, new_y)
                accepted, new_score, new_valid, new_lp = _try_move(pts, move_idx, new_pos, line_points, score, valid)
                if accepted and new_valid and new_score > score:
                    line_points = new_lp
                    score = new_score
                    improved = True
                    break
            if not improved:
                continue
    
    return pts, line_points, score, valid

def _simulated_annealing(pts, line_points, score, valid, deadline):
    """Simulated annealing to escape local optima."""
    n = len(pts)
    T = 1.0
    T_min = 0.01
    cooling = 0.9997
    
    best_score = score
    best_pts = [p[:] for p in pts]
    best_lp = {k: set(v) for k, v in line_points.items()}
    
    no_improve_count = 0
    
    while time.time() < deadline:
        idx = random.randint(0, n - 1)
        
        # Generate candidate
        if random.random() < 0.5:
            new_x = pts[idx][0] + random.randint(-5, 5)
            new_y = pts[idx][1] + random.randint(-5, 5)
        elif random.random() < 0.8:
            new_x = pts[idx][0] + random.randint(-15, 15)
            new_y = pts[idx][1] + random.randint(-15, 15)
        else:
            new_x = pts[idx][0] + random.randint(-40, 40)
            new_y = pts[idx][1] + random.randint(-40, 40)
        
        new_pos = (new_x, new_y)
        accepted, new_score, new_valid, new_lp = _try_move(pts, idx, new_pos, line_points, score, valid)
        
        if not accepted:
            continue
        
        if new_valid:
            if new_score > score:
                line_points = new_lp
                score = new_score
                no_improve_count = 0
                if score > best_score:
                    best_score = score
                    best_pts = [p[:] for p in pts]
                    best_lp = {k: set(v) for k, v in line_points.items()}
            else:
                delta = score - new_score
                if random.random() < math.exp(-delta / max(T, 1e-9)):
                    line_points = new_lp
                    score = new_score
                    no_improve_count += 1
                else:
                    # Revert
                    old_pos = (pts[idx][0] - (new_x - pts[idx][0]), pts[idx][1] - (new_y - pts[idx][1]))
                    # This revert logic is wrong - let me fix it
                    # Actually, _try_move already modified pts[idx], so we need to restore
                    # Let me restructure: _try_move should not modify pts, just compute
                    pass
        else:
            # Invalid: always revert
            pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    random.seed(42)
    
    n = 16
    
    # Multiple diverse starting configurations
    start_configs = []
    
    # 4x4 grid
    start_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 5x4 grid subset (16 points)
    grid_5x4 = [[x, y] for x in range(5) for y in range(4)]
    # Randomly select 16 from 20
    for _ in range(5):
        subset = random.sample(grid_5x4, 16)
        start_configs.append(subset)
    
    # 6x4 grid subset
    grid_6x4 = [[x, y] for x in range(6) for y in range(4)]
    for _ in range(3):
        subset = random.sample(grid_6x4, 16)
        start_configs.append(subset)
    
    # 4x5 grid subset
    grid_4x5 = [[x, y] for x in range(4) for y in range(5)]
    for _ in range(3):
        subset = random.sample(grid_4x5, 16)
        start_configs.append(subset)
    
    # Random points in a larger grid
    for _ in range(5):
        pts = []
        while len(pts) < 16:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        start_configs.append(pts)
    
    best_overall_score = -1
    best_overall_pts = None
    
    # Time budget per starting config
    time_per_config = min(10.0, (deadline - time.time()) / max(1, len(start_configs)))
    
    for config_idx, initial_pts in enumerate(start_configs):
        if time.time() > deadline - 5.0:
            break
        
        config_deadline = min(deadline - 2.0, time.time() + time_per_config)
        
        pts = [p[:] for p in initial_pts]
        score, valid, line_points = _compute_line_counts(pts)
        
        # If invalid, try to fix with local search
        if not valid:
            pts, line_points, score, valid = _local_search(pts, line_points, score, valid, config_deadline, max_iters=500)
            if not valid:
                continue
        
        # Run local search to improve
        pts, line_points, score, valid = _local_search(pts, line_points, score, valid, config_deadline, max_iters=1000)
        
        # Run simulated annealing with remaining time
        if time.time() < config_deadline - 1.0:
            # Simple SA loop
            T = 0.5
            T_min = 0.01
            cooling = 0.9995
            best_score_sa = score
            best_pts_sa = [p[:] for p in pts]
            
            while time.time() < config_deadline:
                idx = random.randint(0, n - 1)
                ox, oy = pts[idx]
                
                if random.random() < 0.6:
                    new_x = ox + random.randint(-5, 5)
                    new_y = oy + random.randint(-5, 5)
                else:
                    new_x = ox + random.randint(-20, 20)
                    new_y = oy + random.randint(-20, 20)
                
                new_pos = (new_x, new_y)
                
                # Check bounds and distinctness
                if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                    continue
                dup = False
                for k in range(n):
                    if k != idx and pts[k][0] == new_x and pts[k][1] == new_y:
                        dup = True
                        break
                if dup:
                    continue
                
                old_pos = pts[idx][:]
                pts[idx] = [new_x, new_y]
                
                new_lp, new_score, new_valid = _incremental_update(pts, idx, old_pos, line_points)
                
                if new_valid:
                    if new_score > score:
                        line_points = new_lp
                        score = new_score
                        if score > best_score_sa:
                            best_score_sa = score
                            best_pts_sa = [p[:] for p in pts]
                    else:
                        delta = score - new_score
                        if random.random() < math.exp(-delta / max(T, 1e-9)):
                            line_points = new_lp
                            score = new_score
                        else:
                            pts[idx] = old_pos
                            # Revert line_points
                            line_points, score, valid = _compute_line_counts(pts)
                else:
                    pts[idx] = old_pos
                    line_points, score, valid = _compute_line_counts(pts)
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if best_score_sa > score:
                score = best_score_sa
                pts = best_pts_sa
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = [p[:] for p in pts]
        
        # Recalculate time_per_config for remaining configs
        remaining_configs = len(start_configs) - config_idx - 1
        if remaining_configs > 0:
            time_per_config = min(15.0, (deadline - time.time() - 5.0) / remaining_configs)
    
    if best_overall_pts is None:
        # Fallback: return 4x4 grid
        best_overall_pts = [[x, y] for x in range(4) for y in range(4)]
    
    # Final polish: try to improve the best solution
    if time.time() < deadline - 3.0:
        pts = [p[:] for p in best_overall_pts]
        score, valid, line_points = _compute_line_counts(pts)
        if valid:
            # Quick local search polish
            for _ in range(500):
                if time.time() > deadline - 2.0:
                    break
                idx = random.randint(0, n - 1)
                ox, oy = pts[idx]
                new_x = ox + random.randint(-3, 3)
                new_y = oy + random.randint(-3, 3)
                if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                    continue
                dup = False
                for k in range(n):
                    if k != idx and pts[k][0] == new_x and pts[k][1] == new_y:
                        dup = True
                        break
                if dup:
                    continue
                
                old_pos = pts[idx][:]
                pts[idx] = [new_x, new_y]
                new_lp, new_score, new_valid = _incremental_update(pts, idx, old_pos, line_points)
                
                if new_valid and new_score > score:
                    line_points = new_lp
                    score = new_score
                    best_overall_score = score
                    best_overall_pts = [p[:] for p in pts]
                else:
                    pts[idx] = old_pos
    
    return best_overall_pts
>>>>>>> REPLACE

Wait, I made an error in the revert logic. Let me fix the `_incremental_update` function and the main loop to properly handle reverts. The issue is that `_incremental_update` modifies `pts[idx]` in place, so reverting requires restoring the old position AND recomputing line_points. Let me rewrite this more carefully:

