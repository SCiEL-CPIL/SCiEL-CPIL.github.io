"""
Construct a k-regular graph with girth at least g.
For k=3, g=16: use circulant + voltage lift + LUW fallback.
Optimized with fast non-backtracking verification and efficient cycle checks.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Optimized Circulant Search
        result = _find_circulant_optimized(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Voltage Lift over Petersen
        if not check_deadline():
            result2 = _petersen_lift_optimized(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: LUW Fallback
        if not check_deadline():
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, 11)
                return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _find_circulant_optimized(k, g, deadline):
    """
    Search for C_n(1, a, b) with girth >= g.
    Uses fast pre-checks to reject candidates early.
    """
    min_n = 512
    max_n = 766  # We know 766 is possible (tree), so search up to there

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        half = n // 2
        # Heuristic: a and b should be "far" apart
        for a in range(max(3, n // 4), half):
            if time.time() > deadline - 5:
                return None

            # Quick 4-cycle check
            if _has_4cycle_circ_ab(n, a):
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 3:
                    return None

                # Quick 4-cycle check with b
                if _has_4cycle_circ_ab(n, b) or _has_4cycle_circ_ab_ab(n, a, b):
                    continue

                # Quick 6-cycle check
                if _has_6cycle_circ(n, a, b):
                    continue

                # Quick 8-cycle check
                if _has_8cycle_circ(n, a, b):
                    continue

                # Full girth check
                if _full_girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_circ_ab(n, a):
    """Check for 4-cycles using shifts 1 and a."""
    # 4-cycle: 0 -> x -> y -> z -> 0
    # Steps from {±1, ±a}
    G = [1, -1, a, -a]
    G_mod = [x % n for x in G]

    for i in range(4):
        for j in range(4):
            if G_mod[i] == (-G_mod[j]) % n:
                continue
            for k in range(4):
                if G_mod[j] == (-G_mod[k]) % n:
                    continue
                for l in range(4):
                    if G_mod[k] == (-G_mod[l]) % n:
                        continue
                    s = (G_mod[i] + G_mod[j] + G_mod[k] + G_mod[l]) % n
                    if s == 0:
                        v1 = G_mod[i]
                        v2 = (G_mod[i] + G_mod[j]) % n
                        v3 = (G_mod[i] + G_mod[j] + G_mod[k]) % n
                        if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                            return True
    return False

def _has_4cycle_circ_ab_ab(n, a, b):
    """Check for 4-cycles using all shifts 1, a, b."""
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    # 4-cycle: 0 -> x -> y -> z -> 0
    # Steps: g1, g2, g3, g4 from {±1, ±a, ±b}
    # Sum = 0 mod n.
    # No backtracking.
    for i in range(6):
        for j in range(6):
            if G_mod[i] == (-G_mod[j]) % n:
                continue
            for k in range(6):
                if G_mod[j] == (-G_mod[k]) % n:
                    continue
                for l in range(6):
                    if G_mod[k] == (-G_mod[l]) % n:
                        continue
                    s = (G_mod[i] + G_mod[j] + G_mod[k] + G_mod[l]) % n
                    if s == 0:
                        v1 = G_mod[i]
                        v2 = (G_mod[i] + G_mod[j]) % n
                        v3 = (G_mod[i] + G_mod[j] + G_mod[k]) % n
                        if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                            return True
    return False

def _has_6cycle_circ(n, a, b):
    """Check for 6-cycles in C_n(1, a, b)."""
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    # BFS to depth 3
    dist = {0: 0}
    parent = {0: None}
    level = [0]
    
    for depth in range(3):
        next_level = []
        for u in level:
            for g in G_mod:
                v = (u + g) % n
                if v == 0 and depth < 2:
                    continue
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    for v in level:
        for g in G_mod:
            w = (v + g) % n
            if w in dist and dist[w] == 3 and parent[w] != v and parent[v] != w:
                return True
    return False

def _has_8cycle_circ(n, a, b):
    """Check for 8-cycles in C_n(1, a, b)."""
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    # BFS to depth 4
    dist = {0: 0}
    parent = {0: None}
    level = [0]
    
    for depth in range(4):
        next_level = []
        for u in level:
            for g in G_mod:
                v = (u + g) % n
                if v == 0 and depth < 3:
                    continue
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    for v in level:
        for g in G_mod:
            w = (v + g) % n
            if w in dist and dist[w] == 4 and parent[w] != v and parent[v] != w:
                return True
    return False

def _full_girth_check_circ(n, a, b, g):
    """Full non-backtracking girth check for C_n(1,a,b)."""
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    dist = {}
    queue = deque()
    start = (0, -1)
    dist[start] = 0
    queue.append(start)
    max_len = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_len:
            continue
        for g_step in G_mod:
            v = (u + g_step) % n
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_len:
                continue
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                if v == 0 and u != -1:
                    return False
                queue.append(ns)
    return True


def _petersen_lift_optimized(k, g, deadline):
    """
    Voltage lift over Petersen graph.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Search m such that 10*m < 766 => m < 76
    # Moore bound 510 => m >= 51
    for m in range(51, 76):
        if time.time() > deadline - 10:
            return None
        
        # Try random voltage assignments
        for _ in range(50):
            if time.time() > deadline - 5:
                return None
            
            voltages = {}
            for u, v in petersen_edges:
                vol = random.randint(0, m - 1)
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
            
            if _lift_girth_check_fast(petersen_adj, voltages, m, g):
                n = 10 * m
                edges = []
                for u, v in petersen_edges:
                    vol = voltages[(u, v)]
                    for i in range(m):
                        a = u * m + i
                        b = v * m + ((i + vol) % m)
                        if a < b:
                            edges.append((a, b))
                        else:
                            edges.append((b, a))
                
                if len(edges) == 15 * m:
                    return n, edges

    return None

def _lift_girth_check_fast(base_adj, voltages, m, g):
    """
    Fast girth check for voltage lift.
    """
    bn = len(base_adj)
    for start in range(bn):
        visited = set()
        queue = deque()
        start_state = (start, -1, 0)
        visited.add(start_state)
        queue.append((start, -1, 0, 0))
        
        while queue:
            u, prev, vol, depth = queue.popleft()
            if depth >= g - 1:
                continue
            
            for v in base_adj[u]:
                if v == prev:
                    continue
                new_vol = (vol + voltages.get((u, v), 0)) % m
                new_state = (v, u, new_vol)
                if new_state in visited:
                    continue
                visited.add(new_state)
                new_depth = depth + 1
                
                if v == start and new_vol == 0 and new_depth >= 2:
                    return False
                
                if new_depth < g - 1:
                    queue.append((v, u, new_vol, new_depth))
    return True


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

def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False
    return True

def _bfs_short_cycle(src, adj, n, g):
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True
    return False

# EVOLVE-BLOCK-END
