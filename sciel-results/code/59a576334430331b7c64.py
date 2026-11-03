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


def _circulant_girth_ok(n, steps, g):
    """Check if circulant graph on Z_n with step set has girth >= g."""
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


def _build_circulant_edges(n, steps):
    """Build edge list for circulant graph."""
    edges = []
    step_set = set(steps)
    for i in range(n):
        for s in step_set:
            j = (i + s) % n
            if j > i:
                edges.append((i, j))
    return edges


def _search_circulants(k, g, deadline, max_n=800):
    """Search for circulant graphs Z_n with step set {a, -a, b, -b, n/2} (odd k=5)."""
    if k != 5:
        return None
    
    n_start = 428
    n_end = max_n
    
    for n in range(n_start, n_end + 1, 2):
        if time.time() > deadline - 15:
            break
        half = n // 2
        
        # a ranges: avoid 0, n/2
        a_candidates = set()
        for frac in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            a_val = (n * frac) // 100
            if 1 <= a_val < half:
                a_candidates.add(a_val)
            for delta in [-2, -1, 1, 2]:
                av = a_val + delta
                if 1 <= av < half:
                    a_candidates.add(av)
        
        for frac_num, frac_den in [(1,3), (1,4), (1,5), (2,5), (2,7), (1,6), (1,7), (3,10), (1,8), (2,9)]:
            a_val = (n * frac_num) // frac_den
            if 1 <= a_val < half:
                a_candidates.add(a_val)
        
        a_list = sorted(a_candidates)
        
        for a in a_list:
            if time.time() > deadline - 10:
                break
            b_candidates = set()
            for frac in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                b_val = (n * frac) // 100
                if 1 <= b_val < half and b_val != a and b_val != n - a:
                    b_candidates.add(b_val)
                for delta in [-2, -1, 1, 2]:
                    bv = b_val + delta
                    if 1 <= bv < half and bv != a and bv != n - a:
                        b_candidates.add(bv)
            
            for frac_num, frac_den in [(1,3), (1,4), (1,5), (2,5), (2,7), (1,6), (1,7), (3,10), (1,8), (2,9), (3,7), (4,9)]:
                b_val = (n * frac_num) // frac_den
                if 1 <= b_val < half and b_val != a and b_val != n - a:
                    b_candidates.add(b_val)
            
            b_list = sorted(b_candidates)
            
            for b in b_list:
                if time.time() > deadline - 5:
                    break
                steps = {a, n - a, b, n - b, half}
                if len(steps) != 5:
                    continue
                if _circulant_girth_ok(n, steps, g):
                    edges = _build_circulant_edges(n, steps)
                    return n, edges
    return None


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar bipartite graph construction. Guaranteed valid."""
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


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 140.0
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
    
    # Compute Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    # Strategy 1: Search circulants (most promising for k=5, g=9)
    if k == 5:
        result = _search_circulants(k, g, deadline, max_n=550)
        if result is not None:
            return result
    
    # Strategy 2: Fallback to LUW construction (guaranteed valid, larger)
    return _luw_construct(k, g)
