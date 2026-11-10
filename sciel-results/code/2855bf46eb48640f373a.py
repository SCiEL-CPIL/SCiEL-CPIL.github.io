import numpy as np
import time
import random
from math import gcd
from collections import defaultdict

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    # Make first non-zero entry positive
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)

def _is_valid(points, lines):
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in [p, l]:
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    # Check unique points
    pts_int = np.round(p, 6).astype(int)
    if len(set(map(tuple, pts_int.tolist()))) != 54:
        return False
    # Check unique lines
    seen_lines = set()
    for row in np.round(l, 6).astype(int).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    return True

def _count_incidences(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # Vectorized check
    # lns[:, 0, None, None] * pts[:, 0, None] -> (27, 54)
    # This might be memory heavy but 27*54 is small.
    vals = lns[:, 0, None, None] * pts[:, 0, None] + lns[:, 1, None, None] * pts[:, 1, None] + lns[:, 2, None, None]
    return int(np.sum(vals == 0))

def _line_intersection(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None
    x = x_num // det
    y = y_num // det
    if abs(x) > 50 or abs(y) > 50:
        return None
    return (x, y)

def _generate_static_candidates():
    """
    Generate a pool of 'good' candidate points and lines.
    Points: Integer grid in [-20, 20] x [-20, 20].
    Lines: Lines with small coefficients.
    """
    pts = set()
    for x in range(-20, 21):
        for y in range(-20, 21):
            pts.add((x, y))
    
    lns = set()
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                canon = _normalize_line(A, B, C)
                if canon:
                    lns.add(canon)
    
    return list(pts), list(lns)

def _seed_grid(rows, cols, offset_x=0, offset_y=0):
    points = []
    for x in range(cols):
        for y in range(rows):
            points.append((x + offset_x, y + offset_y))
    return points

def _seed_clustered():
    # 3x3 blocks of points, arranged in a 2x3 grid of blocks?
    # Total points: 2 * 3 * 9 = 54.
    points = []
    for bx in range(2):
        for by in range(3):
            for dx in range(3):
                for dy in range(3):
                    # Space blocks apart to allow lines between them
                    x = bx * 5 + dx
                    y = by * 5 + dy
                    points.append((x, y))
    return points

def _local_search(points, lines, deadline, rng, static_pts, static_lns):
    """
    Simulated Annealing local search.
    """
    pts = [tuple(p) for p in points]
    lns = [tuple(l) for l in lines]
    n_pts = 54
    n_lines = 27
    
    # Incidence matrix: M[i, j] = 1 if point i on line j
    M = np.zeros((n_pts, n_lines), dtype=np.int8)
    for i in range(n_pts):
        x, y = pts[i]
        for j in range(n_lines):
            A, B, C = lns[j]
            if A * x + B * y + C == 0:
                M[i, j] = 1
                
    current_I = int(np.sum(M))
    best_I = current_I
    best_pts = pts[:]
    best_lns = lns[:]
    
    # Build dynamic candidate sets
    # Points: Intersections of current lines
    dyn_pts = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(lns[i], lns[j])
            if pt:
                dyn_pts.add(pt)
    # Add neighbors of current points
    for p in pts:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                dyn_pts.add((p[0] + dx, p[1] + dy))
    
    # Lines: Lines through pairs of current points
    dyn_lns = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0: continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                canon = _normalize_line(A, B, C)
                if canon:
                    dyn_lns.add(canon)
    
    # Combine static and dynamic
    all_cand_pts = list(set(static_pts).union(dyn_pts))
    all_cand_lns = list(set(static_lns).union(dyn_lns))
    
    if not all_cand_pts or not all_cand_lns:
        return best_pts, best_lns, best_I

    T_start = 10.0
    T_end = 0.01
    # Estimate iterations possible
    # Each iter is ~100 ops. 1e6 iters/sec roughly.
    # We don't know time left, so just loop until deadline.
    
    iter_count = 0
    last_improve_iter = 0
    stagnation_limit = 5000
    
    while time.time() < deadline:
        iter_count += 1
        # Cooling
        # Use a fixed number of "virtual" steps for cooling to ensure T drops
        # Or just use time-based? Time-based is safer.
        # Let's use a simple exponential decay based on iter count, assuming ~50000 iters total.
        # If we run longer, T stays low.
        T = T_start * (T_end / T_start) ** (min(iter_count, 50000) / 50000.0)
        
        if rng.random() < 0.5:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            old_pos = pts[i]
            new_pos = all_cand_pts[rng.randint(0, len(all_cand_pts) - 1)]
            
            if new_pos == old_pos:
                continue
            if new_pos in pts:
                # Check if it's the same index
                # Since pts is a list, we can check if any other index has this pos
                # O(N) check. 54 is small.
                is_dup = False
                for k in range(n_pts):
                    if k != i and pts[k] == new_pos:
                        is_dup = True
                        break
                if is_dup:
                    continue
            
            delta = 0
            for j in range(n_lines):
                A, B, C = lns[j]
                was = M[i, j]
                now = 1 if (A * new_pos[0] + B * new_pos[1] + C == 0) else 0
                delta += now - was
            
            new_total = current_I + delta
            if new_total >= current_I or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_I = new_total
                pts[i] = new_pos
                for j in range(n_lines):
                    A, B, C = lns[j]
                    M[i, j] = 1 if (A * new_pos[0] + B * new_pos[1] + C == 0) else 0
                
                if current_I > best_I:
                    best_I = current_I
                    best_pts = pts[:]
                    best_lns = lns[:]
                    last_improve_iter = iter_count
        else:
            # Move a line
            j = rng.randint(0, n_lines - 1)
            old_line = lns[j]
            new_line = all_cand_lns[rng.randint(0, len(all_cand_lns) - 1)]
            
            new_canon = _normalize_line(new_line[0], new_line[1], new_line[2])
            old_canon = _normalize_line(old_line[0], old_line[1], old_line[2])
            
            if new_canon is None or new_canon == old_canon:
                continue
            
            # Check duplicate line
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    if _normalize_line(lns[k][0], lns[k][1], lns[k][2]) == new_canon:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            delta = 0
            A, B, C = new_line
            for i in range(n_pts):
                was = M[i, j]
                now = 1 if (A * pts[i][0] + B * pts[i][1] + C == 0) else 0
                delta += now - was
            
            new_total = current_I + delta
            if new_total >= current_I or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_I = new_total
                lns[j] = new_line
                for i in range(n_pts):
                    M[i, j] = 1 if (A * pts[i][0] + B * pts[i][1] + C == 0) else 0
                
                if current_I > best_I:
                    best_I = current_I
                    best_pts = pts[:]
                    best_lns = lns[:]
                    last_improve_iter = iter_count
        
        # Stagnation check: if no improvement for a long time, restart with perturbation?
        # For now, just continue. The cooling should handle it.
        # If we want to be aggressive:
        if iter_count - last_improve_iter > stagnation_limit and T > T_end * 2:
             # Perturb best solution and restart? 
             # This is risky. Let's just let it run.
             pass

    return best_pts, best_lns, best_I

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # Pre-compute static candidates
    static_pts, static_lns = _generate_static_candidates()
    
    # Define seeds
    seeds = []
    # 6x9 Grid
    seeds.append(_seed_grid(9, 6))
    # 9x6 Grid
    seeds.append(_seed_grid(6, 9))
    # 5x10 Grid
    seeds.append(_seed_grid(10, 5))
    # 3x18 Grid
    seeds.append(_seed_grid(18, 3))
    # Clustered
    seeds.append(_seed_clustered())
    
    best_I = 0
    best_points = None
    best_lines = None
    
    # Run local search on each seed
    # Allocate time based on number of seeds
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time) / n_seeds
    # Ensure at least 10s per seed if possible, or just divide evenly
    # If total time is 580, and 5 seeds, ~116s each.
    
    for i, seed_pts in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        # Initial line selection for this seed
        # Generate lines through pairs of points
        lns_set = set()
        for a in range(54):
            for b in range(a + 1, 54):
                x1, y1 = seed_pts[a]
                x2, y2 = seed_pts[b]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0: continue
                A = dy // g
                B = -dx // g
                C = -(A * x1 + B * y1)
                if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                    canon = _normalize_line(A, B, C)
                    if canon:
                        lns_set.add(canon)
        
        # Score lines
        line_scores = {}
        for line in lns_set:
            A, B, C = line
            cnt = 0
            for pt in seed_pts:
                x, y = pt
                if A * x + B * y + C == 0:
                    cnt += 1
            line_scores[line] = cnt
        
        # Select top 27
        sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
        selected_lines = []
        seen = set()
        for line, score in sorted_lines:
            if score < 2:
                break
            if line not in seen:
                selected_lines.append(line)
                seen.add(line)
            if len(selected_lines) == 27:
                break
        
        if len(selected_lines) < 27:
            # Pad with random lines from static pool
            rng_pad = random.Random(i)
            while len(selected_lines) < 27:
                l = static_lns[rng_pad.randint(0, len(static_lns) - 1)]
                if l not in seen:
                    selected_lines.append(l)
                    seen.add(l)
        
        # Run local search
        # Deadline for this seed
        seed_deadline = min(deadline - 10, time.time() + time_per_seed)
        
        rng = random.Random(1000 + i)
        res_pts, res_lns, res_I = _local_search(seed_pts, selected_lines, seed_deadline, rng, static_pts, static_lns)
        
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
    
    # If time remains, polish the best solution
    if best_points is not None and time.time() < deadline - 5:
        rng_polish = random.Random(9999)
        # Use a tighter deadline for polish
        polish_deadline = deadline - 2
        best_points, best_lines, best_I = _local_search(best_points, best_lines, polish_deadline, rng_polish, static_pts, static_lns)
        
    # Final validation and formatting
    if best_points is None:
        # Fallback
        best_points = _seed_grid(9, 6)
        best_lines = [(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)]
        best_lines = best_lines[:27]
        
    pts = np.array(best_points, dtype=np.int64)
    lns = np.array(best_lines, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback to simple valid grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
        lns = lines
    
    return pts, lns
