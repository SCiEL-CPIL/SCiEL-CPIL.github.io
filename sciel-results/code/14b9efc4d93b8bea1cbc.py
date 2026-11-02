# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def _moore_n(k, g):
    """Moore bound: minimum n for k-regular girth g."""
    if g % 2 == 0:
        r = g // 2
        if k == 2:
            return 2 * r  # cycle of length 2r
        return 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        if k == 2:
            return 2 * r + 1
        return k * sum((k-1)**i for i in range(r))

def _girth_check(n, edges, k, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Verify degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check for short cycles via non-backtracking walks
    max_depth = (g - 1) // 2
    for start in range(n):
        visited = {(start, -1): 0}
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nbr in adj[node]:
                if nbr == prev:
                    continue
                new_key = (nbr, node)
                if new_key in visited:
                    cycle_len = dist + 1 + visited[new_key]
                    if cycle_len < g:
                        return False
                else:
                    visited[new_key] = dist + 1
                    queue.append((nbr, node, dist + 1))
    return True

def _random_k_regular_swap(n, k, rng, max_swaps=100000):
    """Generate a random k-regular simple graph using configuration model + swap repair."""
    if n * k < 2 or n * k % 2 != 0 or k >= n:
        return None
    
    # Start with a valid k-regular graph: use the circulant-like construction
    # For k-regular on n vertices, use edges (i, (i+j) mod n) for j in a set of k/2 steps
    # This works when k is even and n is large enough
    if k % 2 == 0:
        # Use symmetric difference set
        steps = list(range(1, k//2 + 1))
        adj = [set() for _ in range(n)]
        for i in range(n):
            for s in steps:
                j = (i + s) % n
                if i != j and j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
        # Check if all degrees are k
        valid = all(len(adj[i]) == k for i in range(n))
        if not valid:
            return None
    else:
        # For odd k, use a different base: take k-1 even part + one perfect matching
        if k == 1:
            if n % 2 != 0:
                return None
            adj = [set() for _ in range(n)]
            for i in range(0, n, 2):
                adj[i].add(i+1)
                adj[i+1].add(i)
        else:
            # Use (k-1) even steps + random perfect matching
            steps = list(range(1, (k-1)//2 + 1))
            adj = [set() for _ in range(n)]
            for i in range(n):
                for s in steps:
                    j = (i + s) % n
                    if i != j and j not in adj[i]:
                        adj[i].add(j)
                        adj[j].add(i)
            # Add a random perfect matching
            remaining = list(range(n))
            rng.shuffle(remaining)
            for i in range(0, n, 2):
                u, v = remaining[i], remaining[i+1]
                if u == v or v in adj[u]:
                    return None
                adj[u].add(v)
                adj[v].add(u)
            valid = all(len(adj[i]) == k for i in range(n))
            if not valid:
                return None
    
    # Now randomize via 2-switches to get a random k-regular graph
    for _ in range(max_swaps):
        # Pick two random edges (a,b) and (c,d), replace with (a,c) and (b,d) if valid
        # Or pick a random vertex and swap one of its neighbors
        i = rng.randint(0, n-1)
        if len(adj[i]) < 2:
            continue
        nbrs = list(adj[i])
        a, b = nbrs[rng.randint(0, len(nbrs)-1)], nbrs[rng.randint(0, len(nbrs)-1)]
        if a == b:
            continue
        # Find a neighbor c of a (c != i) and d of b (d != i)
        a_nbrs = [x for x in adj[a] if x != i]
        b_nbrs = [x for x in adj[b] if x != i]
        if not a_nbrs or not b_nbrs:
            continue
        c = a_nbrs[rng.randint(0, len(a_nbrs)-1)]
        d = b_nbrs[rng.randint(0, len(b_nbrs)-1)]
        # Check that (a,d) and (b,c) don't already exist and aren't loops
        if d == a or c == b or d in adj[a] or c in adj[b]:
            continue
        # Perform swap
        adj[a].remove(i)
        adj[b].remove(i)
        adj[a].add(d)
        adj[b].add(c)
        adj[i].remove(a)
        adj[i].remove(b)
        adj[i].add(d)
        adj[i].add(c)
        adj[d].remove(a)
        adj[d].add(i)
        adj[c].remove(b)
        adj[c].add(i)
    
    # Convert to edge list
    edge_set = set()
    for v in range(n):
        for u in adj[v]:
            if v < u:
                edge_set.add((v, u))
    if len(edge_set) != n * k // 2:
        return None
    return list(edge_set)

def construct(k, g):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # Correct Moore bound
    if g % 2 == 0:
        r = g // 2
        if k == 2:
            n_moore = 2 * r
        else:
            n_moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        if k == 2:
            n_moore = 2 * r + 1
        else:
            n_moore = k * sum((k-1)**i for i in range(r))
    
    best_n = None
    best_edges = None
    
    # For k=4, g=10: n_moore = 2*(1+3+9+27+81) = 2*121 = 242
    # Known best: 320. Search from n_moore upward.
    
    # Number of attempts per n value
    attempts_per_n = 30
    
    # Search range: from n_moore to n_moore + 300
    max_search_range = 300
    
    for n in range(n_moore, n_moore + max_search_range + 1):
        if time.time() > deadline - 15:
            break
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
        
        for attempt in range(attempts_per_n):
            if time.time() > deadline - 10:
                break
            
            edges = _random_k_regular_swap(n, k, rng, max_swaps=50000)
            if edges is None:
                continue
            
            if _girth_check(n, edges, k, g):
                best_n = n
                best_edges = edges
                break
        
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the Lazebnik-style construction
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
    index, order, edges_list = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline - 1:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_list.append((index[vertex], index[other]))
    return len(order), edges_list
# EVOLVE-BLOCK-END
