import time
import random
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            if dx_n == 0:
                key = ('v', x1)
            else:
                c = dy_n * x1 - dx_n * y1
                key = ('s', dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y1) != dy * (x - x1):
            return True
    return False

def _line_key(x1, y1, x2, y2):
    """Return canonical line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    if dx_n == 0:
        return ('v', x1)
    else:
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _point_on_line(x1, y1, x2, y2, x3, y3):
    """Check if point 3 is on line through points 1 and 2."""
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _generate_structured_candidates():
    """Generate various structured configurations."""
    candidates = []
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Two rows of 6
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Three rows of 4
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 5x5 grid subset - center-heavy
    base_5x5 = [[x, y] for x in range(5) for y in range(5)]
    special_5x5 = []
    for x in range(5):
        for y in range(5):
            if x == y or x + y == 4 or x == 2 or y == 2:
                special_5x5.append([x, y])
    if len(special_5x5) >= 12:
        candidates.append(special_5x5[:12])
    elif len(special_5x5) > 0:
        remaining = [p for p in base_5x5 if p not in special_5x5]
        while len(special_5x5) < 12 and remaining:
            special_5x5.append(remaining.pop(0))
        if len(special_5x5) == 12:
            candidates.append(special_5x5)
    
    # Hexagonal arrangement
    hex_pts = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]
    extra = [[1, -1], [-1, 1], [2, 0], [-2, 0], [0, 2]]
    candidates.append(hex_pts + extra)
    
    # Parabolic
    parab = [[x, x*x] for x in range(-2, 4)]
    extra2 = [[-2, -1], [-1, -1], [0, -1]]
    candidates.append(parab + extra2)
    
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # Diagonal-shifted grid
    diag_pts = []
    for row in range(3):
        offset = row * 2
        for col in range(4):
            diag_pts.append([col + offset, row])
    candidates.append(diag_pts)
    
    # Additional structured configs designed to maximize collinear triples
    # Config: 4 points on each of 3 parallel lines
    for slope in [0, 1, -1]:
        pts = []
        if slope == 0:
            for row in range(3):
                for col in range(4):
                    pts.append([col, row])
        elif slope == 1:
            for row in range(3):
                for col in range(4):
                    pts.append([col, col + row])
        else:
            for row in range(3):
                for col in range(4):
                    pts.append([col, -col + row])
        candidates.append(pts)
    
    # Config: 3 points on each of 4 lines forming a quadrilateral pattern
    quad_pts = []
    for i in range(3):
        quad_pts.append([i, 0])
        quad_pts.append([i, 4])
    for i in range(1, 3):
        quad_pts.append([0, i*2])
        quad_pts.append([4, i*2])
    candidates.append(quad_pts[:12])
    
    # Config: points on conic-like curves with added collinearities
    conic_pts = []
    for x in range(-2, 4):
        conic_pts.append([x, x*x - 2])
    # Add points to create collinear triples
    conic_pts += [[-1, 3], [2, 3], [0, -3]]
    candidates.append(conic_pts[:12])
    
    # Config: 6 pairs of points, each pair extended to form a line with a third point
    pair_pts = []
    for i in range(6):
        x = i - 3
        pair_pts.append([x, 0])
        pair_pts.append([x, 1])
        pair_pts.append([x, 2])
    candidates.append(pair_pts)
    
    return candidates

def _try_create_collinearity(points, deadline):
    """Strategy: move a point onto a line defined by two other points to create a 3-point line."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        
        # For each pair (i,j), try to move a third point k onto line ij
        for i in range(12):
            if time.time() > deadline:
                break
            for j in range(i+1, 12):
                if time.time() > deadline:
                    break
                x1, y1 = current[i]
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                # Check if line already has 3+ points
                line_count = 0
                for k in range(12):
                    if k != i and k != j:
                        if _point_on_line(x1, y1, x2, y2, current[k][0], current[k][1]):
                            line_count += 1
                if line_count >= 1:
                    continue  # Already has 3+ points
                
                # Try to move each other point k onto this line
                for k in range(12):
                    if k == i or k == j:
                        continue
                    if time.time() > deadline:
                        break
                    xk, yk = current[k]
                    
                    # Find integer points on the line near (xk, yk)
                    # Line: dy_n * (x - x1) - dx_n * (y - y1) = 0
                    # So: dy_n * x - dx_n * y = dy_n * x1 - dx_n * y1
                    c = dy_n * x1 - dx_n * y1
                    
                    # Search for nearby integer points on this line
                    found_better = False
                    for t in range(-30, 31):
                        if time.time() > deadline:
                            break
                        # Parametrize: (x, y) = (x1 + t*dx_n, y1 + t*dy_n)
                        new_x = x1 + t * dx_n
                        new_y = y1 + t * dy_n
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check distinctness
                        is_dup = False
                        for m in range(12):
                            if m != k and current[m][0] == new_x and current[m][1] == new_y:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        # Check it's actually a different point from k
                        if new_x == xk and new_y == yk:
                            continue
                        current[k] = [new_x, new_y]
                        if not _not_all_collinear(current):
                            current[k] = [xk, yk]
                            continue
                        new_score = _count_ordinary(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            found_better = True
                            if new_score < best_score:
                                best_score = new_score
                                best = [p[:] for p in current]
                            break
                        else:
                            current[k] = [xk, yk]
                    if found_better:
                        break
            if improved:
                break
        
        if not improved:
            # Try random small perturbations to escape local minima
            idx = random.randint(0, 11)
            old_x, old_y = current[idx]
            found = False
            for _ in range(50):
                new_x = old_x + random.randint(-15, 15)
                new_y = old_y + random.randint(-15, 15)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                is_dup = False
                for j in range(12):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                current[idx] = [new_x, new_y]
                if not _not_all_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_score = _count_ordinary(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    found = True
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
            if not found:
                break
    
    return best, best_score

def _local_search(points, deadline, max_radius=5):
    """Local search with point moves and pair swaps."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to nearby positions
        for idx in range(12):
            if time.time() > deadline:
                break
            old_x, old_y = current[idx]
            for dx in range(-max_radius, max_radius + 1):
                if time.time() > deadline:
                    break
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check distinctness
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[idx] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                else:
                    continue
                break
            if improved:
                break
        
        # Try swapping two points with nearby positions
        if not improved and time.time() < deadline:
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i+1, 12):
                    if time.time() > deadline:
                        break
                    x1, y1 = current[i]
                    x2, y2 = current[j]
                    for dx1 in range(-3, 4):
                        for dy1 in range(-3, 4):
                            if dx1 == 0 and dy1 == 0:
                                continue
                            nx1, ny1 = x1 + dx1, y1 + dy1
                            if abs(nx1) > 10**9 or abs(ny1) > 10**9:
                                continue
                            for dx2 in range(-3, 4):
                                for dy2 in range(-3, 4):
                                    if dx2 == 0 and dy2 == 0:
                                        continue
                                    nx2, ny2 = x2 + dx2, y2 + dy2
                                    if abs(nx2) > 10**9 or abs(ny2) > 10**9:
                                        continue
                                    trial = [p[:] for p in current]
                                    trial[i] = [nx1, ny1]
                                    trial[j] = [nx2, ny2]
                                    if len(set(tuple(p) for p in trial)) != 12:
                                        continue
                                    if not _not_all_collinear(trial):
                                        continue
                                    new_score = _count_ordinary(trial)
                                    if new_score < current_score:
                                        current = trial
                                        current_score = new_score
                                        improved = True
                                        if new_score < best_score:
                                            best_score = new_score
                                            best = [p[:] for p in current]
                                        break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        if not improved:
            idx = random.randint(0, 11)
            old_x, old_y = current[idx]
            found = False
            for _ in range(100):
                new_x = old_x + random.randint(-20, 20)
                new_y = old_y + random.randint(-20, 20)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                is_dup = False
                for j in range(12):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                current[idx] = [new_x, new_y]
                if not _not_all_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_score = _count_ordinary(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    found = True
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
            if not found:
                break
    
    return best, best_score

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Generate structured candidates
    candidates = _generate_structured_candidates()
    
    # Add random candidates
    for _ in range(40):
        pts = []
        attempts = 0
        while len(pts) < 12 and attempts < 1000:
            x = random.randint(-25, 25)
            y = random.randint(-25, 25)
            if [x, y] not in pts:
                pts.append([x, y])
            attempts += 1
        if len(pts) == 12 and _not_all_collinear(pts):
            candidates.append(pts)
    
    # Evaluate all candidates
    best_config = None
    best_score = float('inf')
    
    for cfg in candidates:
        if time.time() > deadline - 15:
            break
        if len(cfg) != 12:
            continue
        if len(set(tuple(p) for p in cfg)) != 12:
            continue
        if not _not_all_collinear(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_config = [p[:] for p in cfg]
    
    if best_config is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    # Phase 1: Collinearity creation strategy (30% of time)
    collinearity_deadline = time.time() + (deadline - time.time()) * 0.3
    coll_best, coll_score = _try_create_collinearity(best_config, collinearity_deadline)
    if coll_score < best_score:
        best_score = coll_score
        best_config = coll_best
    
    # Phase 2: Local search from best config
    local_deadline = time.time() + (deadline - time.time()) * 0.4
    best_config, best_score = _local_search(best_config, local_deadline)
    
    # Phase 3: Try local search from other promising configs
    for cfg in candidates:
        if time.time() > deadline - 5:
            break
        if len(cfg) != 12:
            continue
        score = _count_ordinary(cfg)
        if score <= best_score + 3:
            cfg_best, cfg_score = _local_search(cfg, deadline - 5)
            if cfg_score < best_score:
                best_score = cfg_score
                best_config = cfg_best
    
    # Phase 4: Final collinearity polish
    if time.time() < deadline - 2:
        final_best, final_score = _try_create_collinearity(best_config, deadline - 1)
        if final_score < best_score:
            best_score = final_score
            best_config = final_best
    
    return best_config
