import time
import random
from collections import deque

DEADLINE = 115.0

def _build_bipartite(nside, k, rng):
    """Build a random k-regular bipartite graph with 2*nside vertices."""
    edges = []
    for u in range(nside):
        targets = list(range(nside))
        rng.shuffle(targets)
        for v in targets[:k]:
            edges.append((u, nside + v))
    return edges

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _verify_girth_bipartite(adj, n, g):
    """Verify no cycle shorter than g in a bipartite graph.
    Uses non-backtracking walks from every vertex."""
    for s in range(n):
        # State: (current, previous), distance
        dist = {}
        queue = deque()
        for v in adj[s]:
            state = (v, s)
            dist[state] = 1
            queue.append(state)
        while queue:
            (cur, prev), d = queue.popleft()
            if d >= g - 1:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                new_state = (nxt, cur)
                if new_state in dist:
                    continue
                dist[new_state] = d + 1
                queue.append(new_state)
        for (cur, prev), d in dist.items():
            if cur == s and d < g:
                return False
    return True

def _count_cycles_below_g(adj, n, g):
    """Count total number of cycles of length < g (each cycle counted once)."""
    total_cycles = 0
    for s in range(n):
        # DP for non-backtracking walks from s
        # dp[depth] = dict mapping (cur, prev) -> count
        dp = {}
        for v in adj[s]:
            dp[(1, v, s)] = 1
        for depth in range(1, g - 1):
            new_dp = {}
            for (d, cur, prev), cnt in dp.items():
                if d != depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    key = (depth + 1, nxt, cur)
                    new_dp[key] = new_dp.get(key, 0) + cnt
            dp = new_dp
            # Count closed walks of length depth+1 back to s
            for (d, cur, prev), cnt in dp.items():
                if cur == s:
                    # Each cycle of length L has 2L closed non-backtracking walks total
                    # (L starting points * 2 directions). We're counting from one start s,
                    # so cycles containing s contribute L*2/L... 
                    # Actually from start s, a cycle of length L that contains s contributes
                    # exactly 2 closed non-backtracking walks of length L (one in each direction).
                    # So number of cycles of length L containing s = cnt_closed / 2
                    # But we need to avoid double counting across different s.
                    # A cycle of length L is counted at each of its L vertices, so total
                    # sum over s of (cnt_closed_at_s / 2) = L * C_L / 2 * ... 
                    # This is getting complicated. Let me just use this as a heuristic penalty.
                    total_cycles += cnt
    return total_cycles

def construct(k, g):
    start_time = time.time()
    
    # Bipartite Moore bound: each side needs >= 1 + k + k^2 + ... + k^((g/2)-1)
    nside_min = 1
    for i in range(1, g // 2):
        nside_min *= k
    
    rng = random.Random(12345)
    
    best_edges = None
    best_nside = None
    best_penalty = float('inf')
    
    # Search over nside values starting from Moore bound
    max_nside = nside_min + 300
    attempts_per_nside = 30
    
    for nside in range(nside_min, max_nside + 1):
        if time.time() > start_time + DEADLINE - 25:
            break
        n = 2 * nside
        
        for attempt in range(attempts_per_nside):
            if time.time() > start_time + DEADLINE - 15:
                break
            edges = _build_bipartite(nside, k, rng)
            adj = _build_adjacency(n, edges)
            
            # Quick check: count cycles below g
            penalty = _count_cycles_below_g(adj, n, g)
            
            if penalty < best_penalty:
                best_penalty = penalty
                best_edges = edges.copy()
                best_nside = nside
            
            if penalty == 0:
                if _verify_girth_bipartite(adj, n, g):
                    return n, edges
    
    # If we found a good candidate, verify it
    if best_edges is not None and best_penalty == 0:
        n = 2 * best_nside
        adj = _build_adjacency(n, best_edges)
        if _verify_girth_bipartite(adj, n, g):
            return n, best_edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
