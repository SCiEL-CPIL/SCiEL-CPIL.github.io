import time
import random
from math import gcd

DEADLINE = 130

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
    """Build line -> set of point indices. Single pass."""
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
    """Get set of collinear triples as (a,b,c) with a<b<c."""
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
    """Find largest subset with no three collinear using branch and bound."""
    n = len(points)
    triple_set = set(triples)
    best = [0]
    
    # For each i, for each j != i, compute bitmask of k != i,j such that {i,j,k} is a triple
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
    lines = _build_lines(points)
    return _check_no_four_collinear(lines)

def _count_perfect_triples(lines):
    """Count lines with exactly 3 points (perfect triples)."""
    count = 0
    for pts in lines.values():
        if len(pts) == 3:
            count += 1
    return count

def _greedy_cs_construction(seed, deadline):
    """
    Constraint-satisfaction construction using structured bases:
    - Parabola points (naturally many 3-collinear, few 4-collinear)
    - Perturbed grids
    - Random with collision-aware selection
    """
    rng = random.Random(seed)
    mode = seed % 4
    
    if mode == 0:
        # Parabola: y = x^2 perturbed
        candidates = []
        for x in range(-15, 16):
            y = x * x
            px = x + rng.randint(-2, 2)
            py = y + rng.randint(-2, 2)
            if abs(px) <= 100 and abs(py) <= 100:
                candidates.append([px, py])
    elif mode == 1:
        # Dense small grid perturbed
        candidates = []
        for x in range(-10, 11, 2):
            for y in range(-10, 11, 2):
                px = x + rng.randint(-1, 1)
                py = y + rng.randint(-1, 1)
                if abs(px) <= 100 and abs(py) <= 100:
                    candidates.append([px, py])
    elif mode == 2:
        # Wider grid
        candidates = []
        for x in range(-15, 16, 3):
            for y in range(-15, 16, 3):
                px = x + rng.randint(-1, 1)
                py = y + rng.randint(-1, 1)
                if abs(px) <= 100 and abs(py) <= 100:
                    candidates.append([px, py])
    else:
        # Pure random
        candidates = []
        for _ in range(500):
            px = rng.randint(-100, 100)
            py = rng.randint(-100, 100)
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

def _local_search_cs(points, deadline):
    """
    Local search: minimize score via triple-count heuristic,
    with periodic exact score evaluation.
    """
    best = [list(p) for p in points]
    best_lines = _build_lines(best)
    best_triples = _count_perfect_triples(best_lines)
    best_score = _compute_score(best, deadline)
    
    # Precompute neighbor positions for faster checking
    for iteration in range(20):
        if time.time() > deadline - 15:
            break
        improved = False
        for i in range(20):
            if time.time() > deadline - 15:
                break
            orig_x, orig_y = best[i]
            # Try moves in increasing radius
            for radius in range(1, 6):
                if improved:
                    break
                for dx in range(-radius, radius + 1):
                    if improved:
                        break
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig_x + dx, orig_y + dy
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        # Check distinctness
                        dup = False
                        for k in range(20):
                            if k != i and best[k][0] == nx and best[k][1] == ny:
                                dup = True
                                break
                        if dup:
                            continue
                        
                        candidate = [list(p) for p in best]
                        candidate[i] = [nx, ny]
                        
                        # Quick validity check
                        cand_lines = _build_lines(candidate)
                        if not _check_no_four_collinear(cand_lines):
                            continue
                        
                        cand_triples = _count_perfect_triples(cand_lines)
                        
                        # Use triple count as proxy; only do exact score if promising
                        if cand_triples > best_triples:
                            # Verify with exact score periodically
                            exact_s = _compute_score(candidate, deadline)
                            if exact_s <= best_score:
                                best_score = exact_s
                                best_triples = cand_triples
                                best = candidate
                                best_lines = cand_lines
                                improved = True
                                break
                        elif cand_triples == best_triples:
                            # Occasionally check exact score for ties
                            if rng_seed_check(iteration, i, dx, dy):
                                exact_s = _compute_score(candidate, deadline)
                                if exact_s < best_score:
                                    best_score = exact_s
                                    best_triples = cand_triples
                                    best = candidate
                                    best_lines = cand_lines
                                    improved = True
                                    break
        if not improved:
            break
    
    return best, best_score

def rng_seed_check(iteration, i, dx, dy):
    """Deterministic pseudo-random gate for exact score checks on ties."""
    h = (iteration * 31 + i * 17 + dx * 7 + dy * 13) % 100
    return h < 10

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = None
    best_score = 20
    
    # Strategy 1: Multiple greedy CS constructions with different seeds
    # Evaluate exact score less frequently; use triple count as fast filter
    for seed in range(80):
        if time.time() > deadline - 40:
            break
        pts = _greedy_cs_construction(seed, deadline)
        if not _is_valid(pts, deadline):
            continue
        # Fast filter: count triples before expensive exact score
        lines = _build_lines(pts)
        triples = _count_perfect_triples(lines)
        # Only do exact score if triple count looks promising
        if triples >= 15 or best_points is None:
            s = _compute_score(pts, deadline)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
    
    # Strategy 2: Local search on best found
    if best_points is not None:
        if time.time() < deadline - 20:
            best_points, best_score = _local_search_cs(best_points, deadline)
    
    # Fallback: parabola-based
    if best_points is None:
        best_points = [[x, x * x + (x % 3)] for x in range(-10, 10)]
        if not _is_valid(best_points, deadline):
            best_points = [[x, y] for x in range(-5, 6) for y in range(-5, 6)][:20]
    
    return [list(p) for p in best_points]
