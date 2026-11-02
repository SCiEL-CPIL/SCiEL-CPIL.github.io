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
    """
    Construct a k-regular graph with girth >= g using randomized greedy BFS.
    Key idea: build graph level by level. At each level, for each vertex,
    choose k neighbors from the next level (or current level for the last step).
    To ensure girth >= g, we track all vertices within distance (g-1)//2 - 1
    of each other and avoid creating cycles shorter than g.
    """
    # For girth g, we need no cycles of length < g.
    # A BFS tree of depth (g-1)//2 has no cycles within that depth.
    # We build a "tree-like" structure up to depth d = (g-1)//2,
    # then close it up while avoiding short cycles.
    
    # Strategy: 
    # 1. Build a BFS tree to depth d = (g-3)//2 (for odd g=9, d=3)
    # 2. The frontier at depth d has many vertices. Connect them to form
    #    a k-regular graph on the remaining vertices, ensuring no short cycles.
    
    # Actually, a cleaner approach: 
    # Build a k-regular graph where the BFS tree from any vertex has no
    # repeated vertices up to depth (g-1)//2. This is equivalent to girth >= g.
    
    # Greedy construction:
    # - Start with vertex 0.
    # - BFS: at each step, for each vertex at depth i, assign k neighbors.
    #   For depth < d, neighbors go to depth i+1 (new vertices).
    #   For depth == d, neighbors can go to depth d-1 (already assigned) or
    #   to new vertices at depth d+1, but we must avoid creating cycles < g.
    
    # Simpler: use a "lifted" construction or random regular graph with girth check.
    
    # Let's try: random k-regular graph on n vertices, check girth, repeat.
    # But n must be small enough and girth check is expensive.
    
    # Better: constructive approach using the fact that for large enough n,
    # random k-regular graphs have girth growing as log_{k-1}(n).
    # For g=9, we need n > (k-1)^{g/2} roughly = 4^{4.5} ~ 512.
    # So try n around 500-1000.
    
    best_result = None
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Choose n: start near Moore bound, grow if needed
        # Moore bound for k=5, g=9: 1 + 5*(1+4+16+64) = 426
        n = 430 + restart * 20
        if n > max_n:
            break
        
        # Try to construct a k-regular graph on n vertices with girth >= g
        result = _construct_regular_girth(k, g, n, deadline, rng)
        if result is not None:
            return result
    
    return None

def _construct_regular_girth(k, g, n, deadline, rng):
    """
    Try to construct a k-regular graph on n vertices with girth >= g
    using circulant graph search with broadened parameter space.
    """
    # For k=5, we need 5 distinct non-zero steps mod n that form an
    # inverse-closed set. General form: {a, -a, b, -b, c} where c = -c mod n
    # (i.e., c = n/2) OR {a, -a, b, -b, c} where c != -c but we include both.
    # Since k=5 is odd, one step must be its own inverse: c = n/2 (n even)
    # OR we use a non-symmetric set which won't work for undirected graphs.
    # So for k=5 (odd), we need n even and one step = n/2.
    # Connection set: {a, -a, b, -b, n/2}
    
    # Alternative: use a voltage graph approach where we don't require
    # the set to be symmetric, but since we want undirected, it must be.
    
    # For n even: S = {a, n-a, b, n-b, n//2}
    # Need a, b, n//2 all distinct and non-zero mod n.
    # Also need a != b, a != n-b, etc.
    
    # Let's search over n and step values systematically.
    # Start from the given n and try nearby values.
    
    best = None
    t_start = time.time()
    
    # Try the given n first, then expand
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
            continue  # Need even n for odd-degree circulant with self-inverse step
        
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
                # Verify inverse-closed
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
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _circulant_girth_ok(n, steps, g):
    """Check if circulant graph C_n(steps) has girth >= g using non-backtracking walk enumeration."""
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
    """Fallback: Lazebnik-Ustimenko-Woldar bipartite graph construction."""
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
    """
    Construct a k-regular graph with girth at least g.
    Tries circulant graphs with systematic step search, then Cayley on Z_q,
    falls back to LUW.
    """
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
    
    # Moore bound for reference
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    # Start circulant search near the Moore bound
    n_start = max(moore + 4, 200)
    
    # Try circulant graphs first (often finds good solutions quickly)
    result = _construct_regular_girth(k, g, n_start, deadline, random.Random(42))
    if result is not None:
        return result
    
    # Try Cayley graph approach on prime-order groups
    result = _try_cayley_graph(k, g, deadline, max_q=1200)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
