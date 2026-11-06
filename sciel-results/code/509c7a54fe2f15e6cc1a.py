import time
import random
from math import gcd

DEADLINE = 145

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    elif dy == 0:
        return ('h', y1)
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _build_lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _check_no_four_collinear(lines):
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _get_triple_bitmasks(lines):
    triples = set()
    for pts in lines.values():
        if len(pts) >= 3:
            pts_list = sorted(pts)
            m = len(pts_list)
            for a in range(m):
                for b in range(a+1, m):
                    for c in range(b+1, m):
                        triples.add((pts_list[a], pts_list[b], pts_list[c]))
    return triples

def _max_cap_free_fast(points, triples, deadline):
    n = len(points)
    triple_set = set(triples)
    best = [0]
    
    collinear_mask = [[0]*n for _ in range(n)]
    for (a, b, c) in triple_set:
        collinear_mask[a][b] |= (1 << c)
        collinear_mask[a][c] |= (1 << b)
        collinear_mask[b][a] |= (1 << c)
        collinear_mask[b][c] |= (1 << a)
        collinear_mask[c][a] |= (1 << b)
        collinear_mask[c][b] |= (1 << a)
    
    def backtrack(subset_mask, start, current_size):
        if time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        
        if current_size + (n - start) <= best[0]:
            return
        
        for i in range(start, n):
            if time.time() > deadline:
                return
            if subset_mask & (1 << i):
                continue
            valid = True
            temp = subset_mask
            while temp:
                j_bit = temp & (-temp)
                j = j_bit.bit_length() - 1
                if collinear_mask[i][j] & subset_mask:
                    valid = False
                    break
                temp -= j_bit
            
            if valid:
                backtrack(subset_mask | (1 << i), i+1, current_size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _compute_score(points, deadline):
    lines = _build_lines(points)
    if not _check_no_four_collinear(lines):
        return 20
    triples = _get_triple_bitmasks(lines)
    return _max_cap_free_fast(points, triples, deadline)

def _is_valid(points, deadline):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if time.time() > deadline:
        return True
    lines = _build_lines(points)
    return _check_no_four_collinear(lines)

def _count_perfect_triples(lines):
    count = 0
    for pts in lines.values():
        if len(pts) == 3:
            count += 1
    return count

def _greedy_cs_construction(seed, deadline):
    rng = random.Random(seed)
    
    # Generate candidate pool: points on a perturbed grid
    candidates = []
    for x in range(-30, 31, 3):
        for y in range(-30, 31, 3):
            px = x + rng.randint(-1, 1)
            py = y + rng.randint(-1, 1)
            if abs(px) <= 100 and abs(py) <= 100:
                candidates.append([px, py])
    
    rng.shuffle(candidates)
    
    selected = []
    seen = set()
    
    for cand in candidates:
        if len(selected) >= 20:
            break
        if (cand[0], cand[1]) in seen:
            continue
        
        test_points = selected + [cand]
        lines = _build_lines(test_points)
        
        if not _check_no_four_collinear(lines):
            continue
        
        seen.add((cand[0], cand[1]))
        selected.append(cand)
    
    while len(selected) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            selected.append([x, y])
    
    return selected[:20]

def _local_search_hypergraph(points, deadline):
    """
    Local search optimized for constraint satisfaction:
    - Try moving points to increase perfect triple count
    - Maintain no-four-collinear constraint
    - Use hypergraph matching perspective
    """
    best = [list(p) for p in points]
    lines = _build_lines(best)
    best_triples = _count_perfect_triples(lines)
    best_score = _compute_score(best, deadline)
    
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(20):
            if time.time() > deadline - 10:
                break
            orig = best[i][:]
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if (nx, ny) in [tuple(p) for p in best if p is not best[i]]:
                        continue
                    candidate = [list(p) for p in best]
                    candidate[i] = [nx, ny]
                    if _is_valid(candidate, deadline):
                        cand_lines = _build_lines(candidate)
                        cand_triples = _count_perfect_triples(cand_lines)
                        # Prefer more perfect triples, then lower score
                        if cand_triples > best_triples or (cand_triples == best_triples and _compute_score(candidate, deadline) < best_score):
                            best_score = _compute_score(candidate, deadline)
                            best_triples = cand_triples
                            best = candidate
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = None
    best_score = 20
    
    # Strategy 1: Multiple greedy CS constructions with different seeds
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        pts = _greedy_cs_construction(seed, deadline)
        if _is_valid(pts, deadline):
            s = _compute_score(pts, deadline)
            if s < best_score:
                best_score = s
                best_points = pts
    
    # Strategy 2: Local search from best points
    if best_points is not None and time.time() < deadline - 20:
        best_points, best_score = _local_search_hypergraph(best_points, deadline)
    
    # Strategy 3: Try a few more seeds if time permits
    if time.time() < deadline - 10:
        for seed in range(100, 200):
            if time.time() > deadline - 5:
                break
            pts = _greedy_cs_construction(seed, deadline)
            if _is_valid(pts, deadline):
                s = _compute_score(pts, deadline)
                if s < best_score:
                    best_score = s
                    best_points = pts
                    if time.time() < deadline - 15:
                        best_points, best_score = _local_search_hypergraph(best_points, deadline)
    
    if best_points is None:
        # Fallback: parabola
        best_points = [[x, x * x] for x in range(-10, 10)]
        best_score = 20
    
    return [list(p) for p in best_points]
