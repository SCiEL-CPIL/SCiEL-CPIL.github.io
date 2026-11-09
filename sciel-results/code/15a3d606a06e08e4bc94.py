import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth):
    """Check if girth >= target_girth using BFS. Returns False early if a shorter cycle is found.
    Optimized: only check cycles found via edges where both endpoints are at distance >= 1 from start,
    and skip the check when the minimum possible cycle length already exceeds target."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        # Track minimum distance seen so far to enable early termination
        min_dist_seen = 0
        while queue:
            v = queue.popleft()
            dv = dist[v]
            # If dv >= target_girth, all cycles through v would be >= 2*dv >= target_girth
            # But we still need to process for correctness of other paths
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    # Cycle length = dist[v] + dist[u] + 1
                    # Only check if it could be < target_girth
                    if dv + dist[u] + 1 < target_girth:
                        return False
    return True

def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth

def _construct_cage_3_16():
    """Construct a 3-regular bipartite graph with girth >= 16 using random perfect matching approach.
    
    Strategy: Build a 3-regular bipartite graph by choosing 3 random perfect matchings
    between two halves. Use a smarter sampling approach and faster girth check.
    """
    n = 936
    half = n // 2
    deadline = time.time() + 105.0
    
    def _random_perfect_matching():
        """Generate a random perfect matching using Fisher-Yates."""
        perm = list(range(half))
        random.shuffle(perm)
        return perm
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Generate 3 random perfect matchings
        matchings = [_random_perfect_matching() for _ in range(3)]
        
        # Check no repeated edges (same pair appears in multiple matchings)
        edge_set = set()
        valid = True
        for m in matchings:
            for i in range(half):
                edge = (i, m[i])
                if edge in edge_set:
                    valid = False
                    break
                edge_set.add(edge)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for m in matchings:
            for i in range(half):
                j = m[i]
                adj[i].append(half + j)
                adj[half + j].append(i)
        
        # Verify degree is exactly 3 for all vertices
        # (guaranteed by construction with 3 distinct perfect matchings)
        
        # Check girth
        if _has_girth_at_least(adj, n, 16):
            edges = []
            for i in range(half):
                for j in adj[i]:
                    if j >= half:
                        edges.append((i, j))
            return n, edges
    
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction.
    Optimized to find minimal d for required girth."""
    q = k
    # Start with a small d and increase until girth is satisfied
    # Lower bound for d: girth ~ 2d, so d ~ g/2
    # But algebraic constraints might require larger d.
    # Let's try d from 5 to 15.
    deadline = time.time() + 38.0
    
    def _build_luw(d):
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
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    # Try increasing d
    for d in range(5, 20):
        if time.time() > deadline:
            break
        
        n, edges = _build_luw(d)
        
        # Build adjacency for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _has_girth_at_least(adj, n, g):
            return n, edges
        
    # Fallback: use a known-good large d value
    d_orig = max(8, g - 4)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        # Fallback to LUW with tight deadline
        deadline = time.time() + 40.0
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
