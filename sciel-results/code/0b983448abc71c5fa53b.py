"""Construct a 4-regular graph of girth >= g using random bipartite permutation ensembles."""
import time
import random
from collections import deque

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _build_adj(perm_list, n):
    """Build adjacency list for a bipartite graph from permutation list."""
    adj = [[] for _ in range(2 * n)]
    for u in range(n):
        nbrs = set()
        for i in range(len(perm_list)):
            v = (perm_list[i][u] + u) % n
            nbrs.add(n + v)
        adj[u] = list(nbrs)
        for v in nbrs:
            adj[v].append(u)
    return adj

def _verify_girth(adj, g):
    """Verify girth >= g using non-backtracking BFS from every vertex."""
    nv = len(adj)
    for start in range(nv):
        # State: (current_vertex, previous_vertex) in the line graph
        # BFS level = number of edges in the walk
        visited = set()
        queue = deque()
        for nb in adj[start]:
            visited.add((start, nb))
            queue.append((nb, start, 1))
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= g:
                break
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if (cur, nb) not in visited:
                    visited.add((cur, nb))
                    queue.append((nb, cur, depth + 1))
        else:
            # If we exhausted the queue before reaching depth g,
            # no cycle of length < g was found from this start.
            # Actually, the 'else' on while runs when queue is empty.
            # We need to check: if we found a cycle, it would mean
            # a non-backtracking walk of length < g returns to start.
            # But our BFS just explores all non-backtracking walks.
            # A cycle exists if we can reach (start, x) for some x in adj[start]
            # at depth < g. Let me re-check.
            pass
    # The above BFS explores all non-backtracking walks from start.
    # A cycle of length L < g exists iff there's a non-backtracking walk
    # of length L from start back to start. In the line graph BFS,
    # this means we reach a state (start, x) where x is a neighbor of start,
    # at depth L. But we don't track that directly.
    # Let me redo: BFS in the directed line graph.
    # A cycle of length L corresponds to a closed non-backtracking walk of length L.
    # In the line graph, this is a cycle of length L.
    # We can detect it by checking if any state (v, u) where u is a neighbor of start
    # and v == start is reached at depth < g.
    # Actually, simpler: just do BFS and check if we can return to start.
    return True  # placeholder

def _verify_girth_v2(adj, g):
    """Verify girth >= g by checking all non-backtracking walks from each vertex."""
    nv = len(adj)
    for start in range(nv):
        # BFS in the directed line graph starting from all (start, nb)
        # We want to detect if we can return to start via a non-backtracking walk
        # of length < g.
        visited = set()
        queue = deque()
        for nb in adj[start]:
            visited.add((start, nb))
            queue.append((nb, start, 1))
        found_cycle = False
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= g:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    found_cycle = True
                    break
                if (cur, nb) not in visited:
                    visited.add((cur, nb))
                    queue.append((nb, cur, depth + 1))
            if found_cycle:
                break
        if found_cycle:
            return False
    return True

def _try_construct(n, k, g, rng, time_limit):
    """Try to construct a k-regular bipartite graph on 2n vertices with girth >= g."""
    # Start with random permutations
    perm_list = [list(range(n)) for _ in range(k)]
    for p in perm_list:
        rng.shuffle(p)
    
    # Adjust to ensure no duplicate neighbors for any vertex
    # For each u, we need perm_i[u] + u mod n to be distinct for i=0..k-1
    # We can fix this by adjusting the permutations iteratively.
    
    for attempt in range(200):
        if time.time() > time_limit:
            return None
        
        # Check if current perm_list gives a valid k-regular graph
        adj = _build_adj(perm_list, n)
        
        # Check degree
        valid = True
        for v in range(2 * n):
            if len(adj[v]) != k:
                valid = False
                break
        if not valid:
            # Try to fix by swapping in one permutation
            for i in range(k):
                for u in range(n):
                    # Try swapping perm_i[u] with perm_i[u+1]
                    perm_list[i][u], perm_list[i][(u + 1) % n] = perm_list[i][(u + 1) % n], perm_list[i][u]
                    adj2 = _build_adj(perm_list, n)
                    if all(len(adj2[v]) == k for v in range(2 * n)):
                        break
                    # swap back
                    perm_list[i][u], perm_list[i][(u + 1) % n] = perm_list[i][(u + 1) % n], perm_list[i][u]
                else:
                    continue
                break
            else:
                # Randomly re-shuffle one permutation
                i = rng.randint(0, k - 1)
                rng.shuffle(perm_list[i])
                continue
        
        # Verify girth
        if _verify_girth_v2(adj, g):
            # Build edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if v > n:  # only add edges from A to B
                        edges.append((u, v))
            return 2 * n, edges
    
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 4:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction
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
    
    # k == 4: use random bipartite permutation ensemble
    rng = random.Random(42)
    time_limit = _START + _DEADLINE
    
    # Try increasing n starting from a reasonable lower bound
    # Moore bound for 4-regular girth 10: 2 * (1 + 4 * (1 + 4^3) / (4-1)) ≈ 2 * (1 + 4*65/3) ≈ 2*87.67 ≈ 176
    # But the known best is 320, so let's try from 160 upward
    for n in range(160, 350):
        if time.time() > time_limit:
            break
        result = _try_construct(n, 4, g, rng, time_limit)
        if result is not None:
            return result
    
    # If we didn't find anything, fall back to a larger construction
    # Use a simple circulant-like construction
    n = 320
    perm_list = []
    for i in range(4):
        perm = [(u * (i + 1) + i * 7) % n for u in range(n)]
        perm_list.append(perm)
    
    adj = _build_adj(perm_list, n)
    # Fix degree issues
    for _ in range(1000):
        if all(len(adj[v]) == 4 for v in range(2 * n)):
            break
        # Find a vertex with wrong degree and fix
        for v in range(2 * n):
            if len(adj[v]) != 4:
                # Shuffle one permutation
                i = rng.randint(0, 3)
                rng.shuffle(perm_list[i])
                adj = _build_adj(perm_list, n)
                break
    
    # Verify
    if _verify_girth_v2(adj, g):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if v > n:
                    edges.append((u, v))
        return 2 * n, edges
    
    # Last resort: return a valid but possibly larger graph
    # Use the L-U-W construction as fallback
    q = 4
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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
