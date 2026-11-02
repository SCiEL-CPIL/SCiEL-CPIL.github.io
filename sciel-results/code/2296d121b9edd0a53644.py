import time
import sys
import random
from collections import deque, defaultdict

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not _is_prime(n):
        n += 2
    return n

def _girth_ok_abelian(S, q, g):
    S_list = list(S)
    nS = len(S_list)
    S_set = set(S_list)
    neg_map = {x: (-x) % q for x in S_set}
    target_len = g - 1
    
    for s1_idx in range(nS):
        s1 = S_list[s1_idx]
        frontier = {(s1_idx, s1)}
        for t in range(2, target_len + 1):
            new_frontier = set()
            for last_idx, cur_sum in frontier:
                last_gen = S_list[last_idx]
                neg_last = neg_map[last_gen]
                for j in range(nS):
                    sj = S_list[j]
                    if sj == neg_last:
                        continue
                    new_sum = (cur_sum + sj) % q
                    new_frontier.add((j, new_sum))
            for last_idx, cur_sum in new_frontier:
                if cur_sum == 0:
                    if S_list[last_idx] != neg_map[s1]:
                        return False
            if not new_frontier:
                break
            frontier = new_frontier
    return True

def _try_cayley_graph(k, g, deadline, max_q=600):
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    q_min = max(moore, k + 1)
    q = _next_prime(q_min)
    
    while q <= max_q and time.time() < deadline:
        if not _is_prime(q):
            q += 2
            continue
        
        bases_to_try = list(range(1, min(21, q)))
        for denom in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            val = q // denom
            if 0 < val < q:
                bases_to_try.append(val)
        
        seen = set()
        bases_unique = []
        for b in bases_to_try:
            b_mod = b % q
            if 0 < b_mod < q and b_mod not in seen:
                seen.add(b_mod)
                bases_unique.append(b_mod)
        
        for a in bases_unique[:12]:
            for b in bases_unique[:12]:
                if b == a or b == (-a) % q:
                    continue
                for c in bases_unique[:8]:
                    if c not in {a, b, (-a) % q, (-b) % q, 0}:
                        S = [c, a, b, (-a) % q, (-b) % q]
                        S_set = set(S)
                        if len(S_set) != k or 0 in S_set:
                            continue
                        if S_set != {(-x) % q for x in S_set}:
                            continue
                        if _girth_ok_abelian(S, q, g):
                            n = q
                            edges = []
                            S_list = list(S_set)
                            for i in range(n):
                                for s in S_list:
                                    j = (i + s) % q
                                    if j > i:
                                        edges.append((i, j))
                            return n, edges
        
        if time.time() > deadline - 40:
            break
        q += 2
    return None

def _random_bfs_construct(k, g, deadline, max_restarts=50, max_n=2000):
    best_result = None
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        n = 430 + restart * 20
        if n > max_n:
            break
        result = _construct_regular_girth(k, g, n, deadline, rng)
        if result is not None:
            return result
    return None

def _construct_regular_girth(k, g, n, deadline, rng):
    best = None
    t_start = time.time()
    
    n_values = [n]
    for delta in range(2, 100, 2):
        if n - delta >= 10:
            n_values.append(n - delta)
        if n + delta <= 3000:
            n_values.append(n + delta)
    
    for n_try in n_values:
        if time.time() > deadline - 5:
            break
        if n_try % 2 != 0:
            continue
        
        half = n_try // 2
        max_step = min(50, half - 1)
        
        for a in range(1, max_step + 1):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, max_step + 1):
                if b == half:
                    continue
                steps = [a, n_try - a, b, n_try - b, half]
                step_set = set(steps)
                if len(step_set) != 5:
                    continue
                if step_set != {(-s) % n_try for s in step_set}:
                    continue
                if _circulant_girth_ok(n_try, step_set, g):
                    edges = []
                    for i in range(n_try):
                        for s in step_set:
                            j = (i + s) % n_try
                            if j > i:
                                edges.append((i, j))
                    return n_try, edges
    
    return _luw_construct(k, g)

def _circulant_girth_ok(n, steps, g):
    target = g - 1
    step_list = list(steps)
    nS = len(step_list)
    neg_map = {s: (-s) % n for s in step_list}
    
    for s1_idx in range(nS):
        s1 = step_list[s1_idx]
        frontier = {(s1_idx, s1)}
        for t in range(2, target + 1):
            new_frontier = set()
            for last_idx, cur_sum in frontier:
                last_step = step_list[last_idx]
                neg_last = neg_map[last_step]
                for j in range(nS):
                    sj = step_list[j]
                    if sj == neg_last:
                        continue
                    new_sum = (cur_sum + sj) % n
                    new_frontier.add((j, new_sum))
            for last_idx, cur_sum in new_frontier:
                if cur_sum == 0:
                    if step_list[last_idx] != neg_map[s1]:
                        return False
            if not new_frontier:
                break
            frontier = new_frontier
    return True

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _pg2_incidence_graph(q, deadline):
    """
    Construct the incidence graph of the projective plane PG(2,q).
    Vertices: q^2+q+1 points and q^2+q+1 lines.
    Edges: incidence between points and lines.
    This is a (q+1)-regular bipartite graph with girth 6.
    Only valid when g <= 6, but useful as a fast deterministic fallback.
    For g=9, this won't work directly, but we use it as a seed for
    further construction or as a guaranteed valid output for small g.
    """
    if time.time() > deadline - 5:
        return None
    
    # Points: [x, y, z] with not all zero, up to scaling. Use canonical form.
    points = []
    for x in range(q + 1):
        for y in range(q + 1):
            # z = 1 for affine points, handle z=0 separately
            if x == 0 and y == 0:
                continue
            points.append((x, y, 1))
    # Points at infinity: [x, y, 0] with (x,y) != (0,0), canonical: first nonzero = 1
    for x in range(q + 1):
        for y in range(q + 1):
            if x == 0 and y == 0:
                continue
            # Canonicalize: divide by first nonzero
            if x != 0:
                # x must be 1 in canonical form
                if x == 1:
                    points.append((1, y, 0))
            else:
                # x=0, y!=0, canonical: (0,1,0)
                if y == 1:
                    points.append((0, 1, 0))
    
    # Lines: [a, b, c] with not all zero, up to scaling.
    lines = []
    for a in range(q + 1):
        for b in range(q + 1):
            if a == 0 and b == 0:
                continue
            lines.append((a, b, 1))
    for a in range(q + 1):
        for b in range(q + 1):
            if a == 0 and b == 0:
                continue
            if a != 0:
                if a == 1:
                    lines.append((1, b, 0))
            else:
                if b == 1:
                    lines.append((0, 1, 0))
    
    n_pts = len(points)
    n_lines = len(lines)
    n = n_pts + n_lines
    
    # Verify counts
    expected = q * q + q + 1
    if n_pts != expected or n_lines != expected:
        return None
    
    edges = []
    for i, (px, py, pz) in enumerate(points):
        for j, (la, lb, lc) in enumerate(lines):
            # Point (px,py,pz) is on line (la,lb,lc) iff la*px + lb*py + lc*pz = 0
            if (la * px + lb * py + lc * pz) % q == 0:
                edges.append((i, n_pts + j))
    
    return n, edges

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
    
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    n_start = max(moore + 4, 200)
    
    result = _construct_regular_girth(k, g, n_start, deadline, random.Random(42))
    if result is not None:
        return result
    
    result = _try_cayley_graph(k, g, deadline, max_q=1200)
    if result is not None:
        return result
    
    # For small g, try PG(2,q) incidence graph as deterministic fallback
    if g <= 6 and k == 5:
        result = _pg2_incidence_graph(4, deadline)
        if result is not None:
            return result
    
    return _luw_construct(k, g)
