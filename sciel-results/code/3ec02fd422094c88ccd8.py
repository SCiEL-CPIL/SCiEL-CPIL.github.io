"""Construct a 3-regular graph of girth >= g using tree-pruning and leaf matching."""
import sys
from collections import deque

def construct(k, g):
    """Construct a 3-regular graph with girth >= g."""
    assert k == 3, "This implementation only handles k=3"
    
    # For 3-regular graphs, the girth must be even (no odd cycles in bipartite-like structure
    # isn't strictly required, but cage graphs for (3,g) with even g are well-studied).
    # We'll target girth exactly g (or at least g).
    
    # Moore bound for 3-regular graph of girth g (even):
    # n >= 2 * (3-1)^(g/2 - 1) = 2 * 2^(g/2 - 1)
    if g % 2 == 0:
        moore_bound = 2 * (2 ** (g // 2 - 1))
    else:
        moore_bound = 2 * (2 ** ((g - 1) // 2))
    
    # We'll use a tree-pruning approach:
    # 1. Build a rooted tree of depth d = g//2 - 1 (for even g)
    #    Each internal node has 2 children (since root has degree 3, but one edge is to parent,
    #    so 2 children; non-root internal nodes have 1 parent + 2 children = 3).
    #    Actually, for a 3-regular graph, if we think of it as a tree from a root:
    #    - Root: 3 neighbors
    #    - Each other node: 2 neighbors besides parent
    #    
    #    For girth g (even), the tree has depth g//2 - 1 from the root.
    #    Number of vertices at depth i:
    #    - Depth 0: 1 (root)
    #    - Depth 1: 3
    #    - Depth i (i >= 1): 3 * 2^(i-1)
    #    
    #    Total vertices in tree: 1 + 3 + 3*2 + 3*2^2 + ... + 3*2^(d-1)
    #    where d = g//2 - 1
    #    = 1 + 3 * (2^d - 1) = 3*2^d - 2
    #
    #    For g=16: d = 7, n_tree = 3*2^7 - 2 = 384 - 2 = 382.
    #    But we need to close the tree into a graph, which adds edges between leaves.
    #    The number of leaves is 3 * 2^(d-1) = 3 * 2^6 = 192 for d=7.
    #    We need to pair leaves with edges such that no short cycles are created.
    
    # Let's use a different approach: build the tree, then use a matching on leaves.
    # The key insight: if we have a tree of depth d = g//2 - 1, the leaves are at distance
    # 2*d from each other through the root (if they're in different branches).
    # To avoid cycles shorter than g, we need to ensure that any two leaves that are
    # connected don't create a cycle shorter than g.
    
    # A simpler approach: use the known construction for (3, 2g) cages.
    # For girth 16, we need a 3-regular graph with no cycles of length < 16.
    
    # Let's try a circulant-like approach with careful parameter selection.
    # For 3-regular circulant graphs on n vertices (n odd), generators {a, b, c} with a+b+c=0.
    
    # Actually, let's use a more systematic search with BFS-based girth verification.
    
    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def _next_prime(n):
        if n <= 2:
            return 2
        candidate = n if n % 2 else n + 1
        while not _is_prime(candidate):
            candidate += 2
        return candidate
    
    def _compute_girth(n, adj):
        """Compute girth using BFS from each directed edge."""
        best_girth = float('inf')
        for start_u in range(n):
            for start_v in adj[start_u]:
                # BFS from (start_v, start_u)
                visited = set()
                queue = deque()
                queue.append((start_v, start_u, 1))
                visited.add((start_v, start_u))
                
                while queue:
                    curr, prev, depth = queue.popleft()
                    
                    if curr == start_u and depth >= 3:
                        best_girth = min(best_girth, depth)
                        break
                    
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        state = (nxt, curr)
                        if state not in visited:
                            visited.add(state)
                            if depth + 1 >= best_girth:
                                continue
                            queue.append((nxt, curr, depth + 1))
                
                if best_girth <= g:
                    break
            if best_girth <= g:
                break
        
        return best_girth if best_girth != float('inf') else n
    
    # Search over odd n (for 3-regular circulant with a+b+c=0)
    # Start from Moore bound
    best_n = None
    best_edges = None
    
    n_start = max(moore_bound + 1, 101)
    n = n_start if n_start % 2 == 1 else n_start + 1
    
    max_n = 2000
    trials = 0
    max_trials = 50
    
    while n <= max_n and trials < max_trials:
        trials += 1
        
        # Try generator pairs (a, b) with c = (-a-b) % n
        found = False
        for a in range(1, min(30, n // 4)):
            if found:
                break
            for b in range(a + 1, min(a + 100, n // 2)):
                c = (-a - b) % n
                if c == 0 or c == a or c == b:
                    continue
                if len({a, b, c}) != 3:
                    continue
                
                # Build adjacency
                adj = [[] for _ in range(n)]
                for v in range(n):
                    adj[v] = [(v + a) % n, (v + b) % n, (v + c) % n]
                
                # Check girth
                girth = _compute_girth(n, adj)
                if girth >= g:
                    # Build edges
                    edges = []
                    for v in range(n):
                        for w in adj[v]:
                            if v < w:
                                edges.append((v, w))
                    
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    
                    found = True
                    break
        
        n += 2  # Only odd n
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use Lazebnik-Ustimenko-Woldar construction
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
