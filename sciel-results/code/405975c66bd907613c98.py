import time
import random
from math import gcd
from itertools import combinations

_START_TIME = None
_DEADLINE = 145.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return 0.0
    return time.time() - _START_TIME

def _is_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _check_no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _get_lines(points):
    """Returns a list of sets of indices for each line with >= 3 points."""
    n = len(points)
    lines = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            
            if dx == 0:
                a, b, c = 1, 0, -x1
            elif dy == 0:
                a, b, c = 0, 1, -y1
            else:
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                a, b = dy_n, -dx_n
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
            
            line_key = (a, b, c)
            if line_key in seen:
                continue
            seen.add(line_key)
            
            line_pts = []
            for k in range(n):
                x3, y3 = points[k]
                if a * x3 + b * y3 + c == 0:
                    line_pts.append(k)
            
            if len(line_pts) >= 3:
                lines.append(line_pts)
    return lines

def _build_triples(lines):
    """Returns a list of triples (tuples of indices) from lines."""
    triples = []
    for line in lines:
        m = len(line)
        if m >= 3:
            for p in range(m):
                for q in range(p + 1, m):
                    for r in range(q + 1, m):
                        triples.append((line[p], line[q], line[r]))
    return triples

def _max_independent_set_size(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    """
    lines = _get_lines(points)
    triples = _build_triples(lines)
    n = len(points)
    
    if not triples:
        return n
    
    # Precompute for each node, the triples it is in
    node_triples = [[] for _ in range(n)]
    for i, t in enumerate(triples):
        for idx in t:
            node_triples[idx].append(i)
            
    # Degree of each node
    degrees = [len(node_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    best = [0]
    current_selected = [False] * n
    triple_sets = [set(t) for t in triples]
    
    def backtrack(pos, count):
        if _elapsed() > _DEADLINE - 5:
            return
        
        # Upper bound
        if count + (n - pos) <= best[0]:
            return
            
        if pos == n:
            if count > best[0]:
                best[0] = count
            return
            
        node = order[pos]
        
        # Option 1: Include node
        can_include = True
        for t_idx in node_triples[node]:
            t = triple_sets[t_idx]
            others = [p for p in t if p != node]
            if current_selected[others[0]] and current_selected[others[1]]:
                can_include = False
                break
        
        if can_include:
            current_selected[node] = True
            backtrack(pos + 1, count + 1)
            current_selected[node] = False
        
        # Option 2: Exclude node
        backtrack(pos + 1, count)
        
    backtrack(0, 0)
    return best[0]

def _find_max_independent_set(points):
    """Returns the size and the set of indices of the max independent set."""
    lines = _get_lines(points)
    triples = _build_triples(lines)
    n = len(points)
    
    if not triples:
        return n, list(range(n))
    
    node_triples = [[] for _ in range(n)]
    for i, t in enumerate(triples):
        for idx in t:
            node_triples[idx].append(i)
            
    degrees = [len(node_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    best_size = [0]
    best_set = [[]]
    current_selected = [False] * n
    triple_sets = [set(t) for t in triples]
    
    def backtrack(pos, count):
        if _elapsed() > _DEADLINE - 5:
            return
        
        if count + (n - pos) <= best_size[0]:
            return
            
        if pos == n:
            if count > best_size[0]:
                best_size[0] = count
                best_set[0] = [i for i in range(n) if current_selected[i]]
            return
            
        node = order[pos]
        
        can_include = True
        for t_idx in node_triples[node]:
            t = triple_sets[t_idx]
            others = [p for p in t if p != node]
            if current_selected[others[0]] and current_selected[others[1]]:
                can_include = False
                break
        
        if can_include:
            current_selected[node] = True
            backtrack(pos + 1, count + 1)
            current_selected[node] = False
        
        backtrack(pos + 1, count)
        
    backtrack(0, 0)
    return best_size[0], best_set[0]

def _generate_initial_config():
    """Generate a random valid config with high triple potential."""
    for _ in range(1000):
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        
        if _check_no_four_collinear(points):
            return points
    # Fallback
    points = []
    seen = set()
    while len(points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
    if _check_no_four_collinear(points):
        return points
    return None

def _local_search(points, max_iters=2000):
    """
    Local search to minimize the max independent set size.
    """
    current_points = [list(p) for p in points]
    current_size, current_set = _find_max_independent_set(current_points)
    best_points = [list(p) for p in current_points]
    best_size = current_size
    
    for i in range(max_iters):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Generate candidates
        candidates = []
        
        # Type A: Moves that break the current max independent set
        # For each p in current_set, try moving it to a line defined by 2 other points in current_set
        if len(current_set) >= 3:
            for p_idx in current_set:
                others = [x for x in current_set if x != p_idx]
                # Sample some pairs to avoid O(N^2) explosion
                pairs = list(combinations(others, 2))
                random.shuffle(pairs)
                for q_idx, r_idx in pairs[:5]: # Limit pairs
                    q = current_points[q_idx]
                    r = current_points[r_idx]
                    # Line through q and r
                    dx = r[0] - q[0]
                    dy = r[1] - q[1]
                    # Find integer points on this line within bounds
                    # Parametric: (x, y) = q + t * (dx, dy)
                    # We need to find t such that (x, y) is integer and in bounds
                    # And not equal to q, r, or any other point
                    # And no 4 collinear
                    # Simplified: Just try a few random t values
                    for t in range(-5, 6):
                        if t == 0: continue # q
                        # Check if (q + t*dx, q + t*dy) is integer
                        # Since q, r are integers, dx, dy are integers, t is integer -> x, y are integers
                        new_x = q[0] + t * dx
                        new_y = q[1] + t * dy
                        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                            # Check distinctness
                            is_dup = False
                            for j in range(20):
                                if current_points[j] == [new_x, new_y]:
                                    is_dup = True
                                    break
                            if not is_dup:
                                new_points = [list(p) for p in current_points]
                                new_points[p_idx] = [new_x, new_y]
                                if _check_no_four_collinear(new_points):
                                    candidates.append(new_points)
        
        # Type B: Random moves
        for _ in range(5):
            idx = random.randint(0, 19)
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                is_dup = False
                for j in range(20):
                    if j != idx and current_points[j] == [new_x, new_y]:
                        is_dup = True
                        break
                if not is_dup:
                    new_points = [list(p) for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    if _check_no_four_collinear(new_points):
                        candidates.append(new_points)
        
        # Evaluate candidates
        best_candidate = None
        best_candidate_size = current_size
        
        for cand in candidates:
            if _elapsed() > _DEADLINE - 10:
                break
            cand_size, _ = _find_max_independent_set(cand)
            if cand_size < best_candidate_size:
                best_candidate_size = cand_size
                best_candidate = cand
            elif cand_size == best_candidate_size:
                # Tie-breaker: prefer more triples?
                # We can skip this for speed
                pass
        
        if best_candidate:
            current_points = best_candidate
            current_size = best_candidate_size
            if current_size < best_size:
                best_size = current_size
                best_points = [list(p) for p in current_points]
                if best_size <= 10:
                    break
            # Update current_set for next iteration
            _, current_set = _find_max_independent_set(current_points)
        else:
            # No improvement, maybe do a random restart of one point
            idx = random.randint(0, 19)
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                is_dup = False
                for j in range(20):
                    if j != idx and current_points[j] == [new_x, new_y]:
                        is_dup = True
                        break
                if not is_dup:
                    new_points = [list(p) for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    if _check_no_four_collinear(new_points):
                        current_points = new_points
                        current_size, current_set = _find_max_independent_set(current_points)

    return best_points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_score = 21
    best_points = None
    
    # Run multiple local search sessions
    for r in range(10):
        if _elapsed() > _DEADLINE - 20:
            break
            
        init_points = _generate_initial_config()
        if not init_points:
            continue
            
        improved_points = _local_search(init_points, max_iters=1000)
        
        score = _max_independent_set_size(improved_points)
        
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in improved_points]
            if best_score <= 10:
                break
                
    if best_points is None:
        best_points = _generate_initial_config()
        
    return best_points
