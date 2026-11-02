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


def _girth_check_abelian(S, q, g):
    """Check if Cayley graph on Z_q with symmetric generating set S has girth >= g.
    Uses non-backtracking walk enumeration up to length g-1."""
    S_list = list(S)
    nS = len(S_list)
    S_set = set(S_list)
    neg_map = {x: (-x) % q for x in S_set}
    target_len = g - 1

    for s1_idx in range(nS):
        s1 = S_list[s1_idx]
        # frontier: set of (last_gen_index, current_sum)
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
            # Check if any walk of length t returns to 0
            for last_idx, cur_sum in new_frontier:
                if cur_sum == 0:
                    # This is a closed non-backtracking walk of length t
                    # It's a valid cycle only if the last step != -s1
                    if S_list[last_idx] != neg_map[s1]:
                        return False
            if not new_frontier:
                break
            frontier = new_frontier
    return True


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


def _build_cayley_edges(n, S):
    """Build edge list for Cayley graph on Z_n with generating set S."""
    edges = []
    S_list = list(S)
    for i in range(n):
        for s in S_list:
            j = (i + s) % n
            if j > i:
                edges.append((i, j))
    return edges


def _search_circulants(k, g, deadline, max_n=800):
    """Search for circulant graphs Z_n with step set {a, -a, b, -b, n/2} (odd k=5)."""
    if k != 5:
        return None
    
    # For k=5, n must be even. Step set: {a, n-a, b, n-b, n/2}
    # Need: a != b, a != n/2, b != n/2, a != n-a, b != n-b
    # i.e., a != 0, b != 0, a != n/2, b != n/2, a != b, a != n-b, b != n-a
    
    # Moore bound for k=5, g=9: 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # Try n from 428 upward (even numbers)
    
    # For efficiency, try structured values of a and b
    # a should be roughly n/5 to n/4, b should be different
    
    n_start = 428
    n_end = max_n
    
    for n in range(n_start, n_end + 1, 2):
        if time.time() > deadline - 15:
            break
        half = n // 2
        
        # a ranges: avoid 0, n/2, and we want a < n/2 for uniqueness
        # b ranges: avoid 0, n/2, a, n-a
        # To keep search tractable, try a from a set of "nice" fractions
        a_candidates = set()
        for frac in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            a_val = (n * frac) // 100
            if 1 <= a_val < half:
                a_candidates.add(a_val)
            # Also try nearby
            for delta in [-2, -1, 1, 2]:
                av = a_val + delta
                if 1 <= av < half:
                    a_candidates.add(av)
        
        # Also try specific small values and fractions
        for frac_num, frac_den in [(1,3), (1,4), (1,5), (2,5), (2,7), (1,6), (1,7), (3,10), (1,8), (2,9)]:
            a_val = (n * frac_num) // frac_den
            if 1 <= a_val < half:
                a_candidates.add(a_val)
        
        a_list = sorted(a_candidates)
        
        for a in a_list:
            if time.time() > deadline - 10:
                break
            # b candidates: avoid a, n-a, 0, half
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


def _search_cayley(k, g, deadline, max_q=1500):
    """Search for Cayley graphs on Z_q (q prime) with symmetric generating set S of size k."""
    if k != 5:
        return None
    
    # For k=5, S = {c, a, b, -a, -b} where c is not ±a, ±b, 0
    # q must be prime and >= 5
    
    q = _next_prime(431)  # Start above Moore bound
    
    while q <= max_q and time.time() < deadline - 20:
        # Generate candidate values for a, b, c
        candidates = []
        for frac in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50]:
            val = (q * frac) // 100
            if 0 < val < q:
                candidates.append(val)
        for frac_num, frac_den in [(1,3), (1,4), (1,5), (2,5), (2,7), (1,6), (1,7), (3,10), (1,8), (2,9), (3,7), (4,9), (5,12), (7,13)]:
            val = (q * frac_num) // frac_den
            if 0 < val < q:
                candidates.append(val)
        # Add small values
        for v in range(1, min(30, q)):
            candidates.append(v)
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for v in candidates:
            v_mod = v % q
            if 0 < v_mod < q and v_mod not in seen:
                seen.add(v_mod)
                unique_candidates.append(v_mod)
        
        # Limit search to keep it tractable
        max_a = min(15, len(unique_candidates))
        max_b = min(12, len(unique_candidates))
        max_c = min(8, len(unique_candidates))
        
        for a in unique_candidates[:max_a]:
            if time.time() > deadline - 15:
                break
            na = (-a) % q
            for b in unique_candidates[:max_b]:
                if b == a or b == na:
                    continue
                nb = (-b) % q
                for c in unique_candidates[:max_c]:
                    if c in {a, na, b, nb, 0}:
                        continue
                    # Check S is symmetric: S = {c, a, b, -a, -b}
                    # For symmetry we need -c to also be in S, but -c isn't unless c=0
                    # So for odd k, we can't have a fully symmetric set with 5 elements
                    # unless one element is its own inverse (only 0, excluded)
                    # Actually for Cayley graph, S just needs to be symmetric: S = -S
                    # With 5 elements, we need one element to be self-inverse or pair them
                    # 5 is odd, so we can't pair all. This means pure cyclic Cayley
                    # with 5 generators won't be symmetric unless we use a different approach.
                    # 
                    # Actually, for the graph to be undirected, we need S = -S.
                    # With |S| = 5 (odd), this is impossible in Z_q unless one element
                    # is its own inverse (i.e., 2x = 0 mod q, so x = 0 for prime q).
                    # So we can't use pure cyclic Cayley for k=5!
                    # 
                    # We need to use circulant graphs (even n) or other groups.
                    # Let's skip this and rely on circulants.
                    pass
        q += 2
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


def _search_circulants_extended(k, g, deadline, n_start=428, n_end=600):
    """Extended circulant search with more systematic a, b values."""
    if k != 5:
        return None
    
    for n in range(n_start, n_end + 1, 2):
        if time.time() > deadline - 20:
            break
        half = n // 2
        
        # Systematic search: a from 1 to half-1, but with stride to keep it fast
        # For each a, try a set of b values
        stride_a = max(1, n // 200)  # Check about 200 values of a
        stride_b = max(1, n // 100)  # Check about 100 values of b
        
        for a in range(1, half, stride_a):
            if time.time() > deadline - 10:
                break
            na = n - a
            for b in range(1, half, stride_b):
                if b == a or b == na:
                    continue
                steps = {a, na, b, n - b, half}
                if len(steps) != 5:
                    continue
                if _circulant_girth_ok(n, steps, g):
                    edges = _build_circulant_edges(n, steps)
                    return n, edges
    return None


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0
    
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
        
        # Strategy 2: Extended circulant search with different strides
        result = _search_circulants_extended(k, g, deadline, n_start=428, n_end=500)
        if result is not None:
            return result
    
    # Strategy 3: Try Cayley graphs on Z_q for other k values
    result = _search_cayley(k, g, deadline, max_q=1000)
    if result is not None:
        return result
    
    # Strategy 4: Fallback to LUW construction (guaranteed valid, larger)
    return _luw_construct(k, g)
