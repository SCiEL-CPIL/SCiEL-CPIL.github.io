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
                lines[key] = []
            lines[key].append(i)
            lines[key].append(j)
    # Deduplicate points in lines
    # Since we iterate i<j, each pair is added once.
    # But a line might be formed by multiple pairs.
    # e.g. (0,1), (0,2), (1,2) all same key.
    # lines[key] will have [0,1, 0,2, 1,2].
    # We need unique indices.
    unique_lines = {}
    for k, v in lines.items():
        unique_lines[k] = list(set(v))
    return unique_lines

def _check_no_four_collinear(lines):
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _count_triples(lines):
    count = 0
    for pts in lines.values():
        if len(pts) == 3:
            count += 1
        # if len > 3, invalid, but we check validity before counting
    return count

def _get_triple_set(lines):
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

def _max_cap_free_fast(points, triple_set, deadline):
    n = len(points)
    best = [0]
    
    # Precompute collinear masks
    # collinear_mask[i][j] is bitmask of k such that {i,j,k} is a triple
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
        
        # Upper bound
        if current_size + (n - start) <= best[0]:
            return
        
        for i in range(start, n):
            if time.time() > deadline:
                return
            if subset_mask & (1 << i):
                continue
            
            # Check if adding i is valid
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
    triples = _get_triple_set(lines)
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

def _gen_grid_perturbed(seed):
    rng = random.Random(seed)
    # 4x5 grid
    xs = [0, 1, 2, 3]
    ys = [0, 1, 2, 3, 4]
    points = []
    for x in xs:
        for y in ys:
            px = x * 10 + rng.randint(-2, 2)
            py = y * 10 + rng.randint(-2, 2)
            points.append([px, py])
    
    # Ensure distinct and within bounds
    seen = set()
    unique = []
    for p in points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    while len(unique) < 20:
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique[:20]

def _local_search(points, deadline):
    best = [list(p) for p in points]
    lines = _build_lines(best)
    if not _check_no_four_collinear(lines):
        return best, 20
        
    best_triples = _count_triples(lines)
    best_score = _compute_score(best, deadline)
    
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(20):
            if time.time() > deadline - 10:
                break
            orig = best[i][:]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    if (nx, ny) in [tuple(p) for p in best if p is not best[i]]:
                        continue
                    candidate = [list(p) for p in best]
                    candidate[i] = [nx, ny]
                    c_lines = _build_lines(candidate)
                    if not _check_no_four_collinear(c_lines):
                        continue
                    c_triples = _count_triples(c_lines)
                    if c_triples > best_triples:
                        # Check exact score periodically or if improvement is significant
                        # For now, accept based on triples, verify score at end of iteration
                        best = candidate
                        best_triples = c_triples
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # Re-evaluate exact score for the new best
                new_score = _compute_score(best, deadline)
                if new_score < best_score:
                    best_score = new_score
                # If score didn't improve, we might have traded triples for score?
                # Unlikely, but possible.
                # We keep best_score as the true metric.
                break
                
        if not improved:
            break
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(42)
    
    best_points = None
    best_score = 20
    
    candidates = []
    
    # Generate diverse candidates
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        pts = _gen_grid_perturbed(seed)
        if _is_valid(pts, deadline):
            lines = _build_lines(pts)
            triples = _count_triples(lines)
            candidates.append((triples, pts))
            
    # Sort by triple count
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Evaluate top candidates exactly
    for triples, pts in candidates[:20]:
        if time.time() > deadline - 20:
            break
        s = _compute_score(pts, deadline)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in pts]
            
    # Local search on best found
    if best_points is not None:
        if time.time() < deadline - 10:
            best_points, best_score = _local_search(best_points, deadline)
            
    if best_points is None:
        # Fallback
        best_points = [[x, x * x] for x in range(-10, 10)]
        
    return [list(p) for p in best_points]
