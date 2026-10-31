import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Systematic circulant search C_n(1, a, b)
        # This is fast and structured.
        result = _find_circulant_optimized(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Random local search
        if not check_deadline():
            result2 = _random_local_search(3, 16, deadline)
            if result2 is not None:
                return result2

        # Fallback: LUW (large, but valid)
        try:
            return _luw_construct(k, g)
        except Exception:
            pass

    # Generic fallback
    return _luw_construct(k, g)


def _find_circulant_optimized(k, g, deadline):
    """
    Search for C_n(1, a, b) with girth >= g.
    """
    min_n = 512
    max_n = 550  # Limit search space to keep within time budget
    
    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        half = n // 2
        # a and b must be in [2, n//2 - 1]
        # Heuristic: a and b should be "far" from 1 and each other
        # Sort candidates to try "good" ones first? 
        # For now, iterate all, but break early if found.
        
        # Precompute squares or just check conditions
        for a in range(2, half):
            if time.time() > deadline - 5:
                return None
            
            # Quick check for 4-cycles involving 'a'
            # 4-cycle exists if a+a = 2a is a generator or 0, or a-b is generator, etc.
            # Generators: 1, -1, a, -a, b, -b
            # 2a % n must not be in {1, n-1, a, n-a} (0 is impossible for a!=0)
            two_a = (2 * a) % n
            if two_a in (1, n - 1, a, n - a):
                continue
                
            for b in range(a + 1, half):
                if time.time() > deadline - 5:
                    return None

                # Check 4-cycle conditions
                # Sums of two generators must not be a generator or 0
                # Pairs: (1,1)->2, (1,a)->1+a, (1,b)->1+b, (a,a)->2a, (a,b)->a+b, (b,b)->2b
                # Negatives handled by symmetry.
                
                if (2 * b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (1 + a) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (1 + b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (a + b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (a - b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (1 - a) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (1 - b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue

                # Check 6-cycles (heuristic)
                # If (a+a+a) % n is a generator, etc.
                three_a = (3 * a) % n
                if three_a in (1, n - 1, a, n - a, b, n - b):
                    continue
                three_b = (3 * b) % n
                if three_b in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (a + a + b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (a + b + b) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (a + b - 1) % n in (1, n - 1, a, n - a, b, n - b):
                    continue
                if (a - b + 1) % n in (1, n - 1, a, n - a, b, n - b):
                    continue

                # Full girth check
                if _girth_check_circ_fast(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _girth_check_circ_fast(n, a, b, g):
    """
    Non-backtracking BFS from vertex 0 in C_n(1, a, b).
    Returns True if girth >= g.
    """
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    # State: (current_node, prev_node)
    # We only need to track the minimum depth reached for each state.
    # Since n is small (~512), we can use a dict or a 2D array.
    # 2D array: n x n. 512x512 = 262k entries. Fast.
    
    # dist[v][u] = min depth to reach state (v, u)
    # Initialize with infinity
    INF = 10**9
    dist = [[INF] * n for _ in range(n)]
    
    queue = deque()
    # Start: from 0, we can go to any neighbor.
    # State is (v, prev). For start, prev is -1.
    # We handle start specially.
    for g_step in gens:
        v = (0 + g_step) % n
        # State (v, 0)
        if dist[v][0] > 1:
            dist[v][0] = 1
            queue.append((v, 0, 1))
            
    max_len = g - 1
    
    while queue:
        u, prev, d = queue.popleft()
        
        if d >= max_len:
            continue
            
        for g_step in gens:
            v = (u + g_step) % n
            if v == prev:
                continue
                
            nd = d + 1
            if nd > max_len:
                continue
                
            if dist[v][u] > nd:
                dist[v][u] = nd
                
                if v == 0 and u != 0:
                    # Found a cycle back to 0
                    # Length is nd
                    if nd < g:
                        return False
                else:
                    queue.append((v, u, nd))
                    
    return True


def _random_local_search(k, g, deadline):
    """
    Random 3-regular graph generation + local search.
    """
    random.seed(42)
    best_result = None
    best_n = float('inf')
    
    # Try a few sizes
    for n in [512, 514, 516, 520]:
        if time.time() > deadline - 10:
            break
            
        # Generate random 3-regular graph
        adj = _random_k_regular(n, 3)
        if adj is None:
            continue
            
        # Initial girth
        girth = _girth_nonbacktracking(adj, g + 1)
        
        if girth >= g:
            edges = _adj_to_edges(adj)
            if n < best_n:
                best_n = n
                best_result = (n, edges)
            break
            
        # Local search
        current_girth = girth
        no_improve = 0
        max_iters = 2000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge (u,v)
            u, v = _random_edge(adj, n)
            # Pick random non-adjacent edge (w,x)
            w, x = _random_nonadjacent_partner(adj, n, u, v)
            if w is None:
                no_improve += 1
                continue
                
            # Check validity
            if adj[u][w] or adj[v][x]:
                no_improve += 1
                continue
                
            # Perform swap
            adj[u][v] = False
            adj[v][u] = False
            adj[w][x] = False
            adj[x][w] = False
            adj[u][w] = True
            adj[w][u] = True
            adj[v][x] = True
            adj[x][v] = True
            
            # Check girth
            new_girth = _girth_nonbacktracking(adj, g + 1)
            
            if new_girth > current_girth:
                current_girth = new_girth
                no_improve = 0
                if current_girth >= g:
                    edges = _adj_to_edges(adj)
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break
            else:
                # Revert
                adj[u][w] = False
                adj[w][u] = False
                adj[v][x] = False
                adj[x][v] = False
                adj[u][v] = True
                adj[v][u] = True
                adj[w][x] = True
                adj[x][w] = True
                no_improve += 1
                
            if no_improve > 300:
                break
                
    return best_result


def _random_k_regular(n, k):
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None

    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)

    adj = [[False] * n for _ in range(n)]
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or adj[u][v]:
            return None
        adj[u][v] = True
        adj[v][u] = True

    return adj


def _random_edge(adj, n):
    while True:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and adj[u][v]:
            return u, v


def _random_nonadjacent_partner(adj, n, u, v):
    for _ in range(50):
        w = random.randint(0, n - 1)
        x = random.randint(0, n - 1)
        if w == x or w == u or w == v or x == u or x == v:
            continue
        if not adj[w][x]:
            continue
        if adj[u][w] or adj[v][w] or adj[u][x] or adj[v][x]:
            continue
        return w, x
    return None, None


def _girth_nonbacktracking(adj, max_girth):
    n = len(adj)
    min_girth = max_girth

    for src in range(n):
        best = {}
        queue = deque()
        queue.append((src, -1, 0))
        best[(src, -1)] = 0

        while queue:
            u, prev, d = queue.popleft()

            if d >= min_girth - 1:
                continue

            for v in range(n):
                if not adj[u][v]:
                    continue
                if v == prev:
                    continue

                nd = d + 1

                if v == src and u != -1 and nd >= 2:
                    if nd < min_girth:
                        min_girth = nd
                    continue

                state = (v, u)
                if state not in best or nd < best[state]:
                    best[state] = nd
                    if nd < min_girth - 1:
                        queue.append((v, u, nd))

        if min_girth == 2:
            break

    return min_girth


def _adj_to_edges(adj):
    edges = []
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges


def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

# EVOLVE-BLOCK-END
