import random
import time
import sys

def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    
    Strategy:
    1. Try random k-regular graph construction with girth verification.
    2. Fall back to LUW CD(d,q) algebraic construction if random fails.
    3. Return the smallest valid graph found within time budget.
    """
    deadline = time.time() + 110  # 10s safety margin
    
    # Moore bound lower estimate for n
    # For girth g and degree k: n >= 2 * sum_{i=0}^{(g-2)/2} (k-1)^i for even g
    # For odd g: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    if g % 2 == 0:
        moore = 2 * sum((k-1)**i for i in range((g)//2))
    else:
        moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    # LUW fallback
    def luw_construct(k, g):
        q = k
        d = 3
        while d + 5 < g:
            d += 2
        # GF(q) arithmetic
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
        rule = {}
        for c in range(2, d+1):
            if c <= 4:
                rule[c] = {2:(1,1), 3:(2,1), 4:(1,2)}[c]
            else:
                r = (c-5)%4
                c0 = c - (c-5)%4
                rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+ (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # Girth check using BFS with non-backtracking walks
    def check_girth(n, edges, g):
        """Check if graph has no cycle shorter than g using BFS from each vertex."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS tracking (node, parent) to avoid backtracking
        # A cycle of length L is found if BFS from v reaches a neighbor of v at depth >= g-1
        # More precisely: no cycle shorter than g means BFS tree has no back edges within depth < g
        for start in range(n):
            # BFS from start, tracking depth and parent
            visited = {}  # (node, parent) -> depth
            queue = [(start, -1, 0)]
            visited[(start, -1)] = 0
            found_short = False
            while queue:
                node, parent, depth = queue.pop(0)
                if depth >= g - 1:
                    continue  # no need to expand further for cycle detection
                for nb in adj[node]:
                    if nb == parent:
                        continue  # non-backtracking
                    if (nb, node) in visited:
                        continue
                    new_depth = depth + 1
                    visited[(nb, node)] = new_depth
                    if nb == start:
                        # Found a cycle of length new_depth + 1
                        if new_depth + 1 < g:
                            return False
                    if new_depth < g - 1:
                        queue.append((nb, node, new_depth))
            # Also check: if any neighbor of start is reached at depth < g-1
            # Actually the BFS above handles this: if start is reached at depth d, cycle is d+1
        return True
    
    def check_girth_fast(n, edges, g):
        """Faster girth check: BFS from each vertex, detect cycles of length < g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS tracking shortest path to each node
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                if dist[node] >= g - 1:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif nb != parent[node] and node != parent[nb]:
                        # Found a cycle: dist[node] + 1 + dist[nb]... 
                        # Actually cycle length = dist[node] + dist[nb] + 1
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return False
        return True
    
    # Generate random k-regular graph
    def random_k_regular(n, k):
        """Generate a random k-regular graph on n vertices using configuration model."""
        # Each vertex has k stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        # Pair up stubs
        edges_set = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # loop, retry
            if (min(u,v), max(u,v)) in edges_set:
                return None  # multi-edge, retry
            edges_set.add((min(u,v), max(u,v)))
        
        edges = list(edges_set)
        if len(edges) != n * k // 2:
            return None
        return edges
    
    # Try random construction for various n
    best_n = None
    best_edges = None
    
    # Start from a reasonable lower bound
    # For k=5, g=9: Moore bound ~ 1+5*(1+4+16+64) = 1+5*85 = 426
    # Best known is 1116, so try from ~500 upward
    
    start_n = max(moore + 1, 500)
    
    for trial_n in range(start_n, start_n + 5000):
        if time.time() > deadline:
            break
        if trial_n * k % 2 != 0:
            continue  # need even sum of degrees
        
        # Try multiple random graphs at this n
        attempts = 0
        while attempts < 50:
            if time.time() > deadline:
                break
            edges = random_k_regular(trial_n, k)
            attempts += 1
            if edges is None:
                continue
            # Check girth
            if check_girth_fast(trial_n, edges, g):
                best_n = trial_n
                best_edges = edges
                break
        
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fall back to LUW
    return luw_construct(k, g)
