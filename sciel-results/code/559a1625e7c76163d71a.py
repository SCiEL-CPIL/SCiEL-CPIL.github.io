import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k == 4 and g == 10:
        # Try to find a small valid (4,10)-cage via algebraic construction
        result = _try_algebraic_cage_4_10(start_time, deadline)
        if result is not None:
            return result
        
        # Fallback: LUW with d=5 (Known valid, n=2048)
        n, edges = _luw_graph_with_d(k, g, 5)
        if n is not None:
            return n, edges
        
        return _luw_graph(k, g)
    else:
        # For other parameters, use LUW as a safe fallback
        return _luw_graph(k, g)

def _try_algebraic_cage_4_10(start_time, deadline):
    """Try known algebraic constructions for (4,10)-cage with n < 2048."""
    # Moore bound for (4,10): 1 + 4*(1+3+9+27+81) = 485
    # Known best: 320. Lower bound: 243.
    # Try circulant-type constructions: Cayley graphs on Z_n x Z_m or similar
    
    # Attempt: Use the known (4,10)-cage construction if we can find one
    # Try various n values from lower bound up to 2047
    
    # For efficiency, try a few promising constructions:
    # 1. Generalized Petersen-like
    # 2. Circulant graphs
    
    # Actually, let's try a direct search over small algebraic structures
    # The (4,10) cage on 320 vertices can be constructed as follows:
    # Vertices: (i, j) where i in Z_8, j in Z_40, with specific edge rules
    
    # Let me try the construction from the literature:
    # A (4,10)-cage can be built from the Hoffman-Singleton-like approach
    
    # Try: Cayley graph on Z_5 x Z_64 with generators
    # or Z_8 x Z_40
    
    # For now, let's try a systematic but bounded search
    # over small circulant/Cayley constructions
    
    # Try n = 320 with a specific known construction
    # The (4,10)-cage can be constructed using the following:
    # Group: Z_5 x Z_64, generators: (1,0), (-1,0), (0,1), (0,-1) won't work (too small girth)
    
    # Let me try a different approach: use the fact that n must be even
    # and try to construct using tensor products or known small cages
    
    # Given time constraints, let's try a few specific known constructions
    # and verify them
    
    # Construction attempt: Use the (4,10) cage from Exoo et al.
    # This is typically constructed as a voltage graph over a base graph
    
    # For a quick win, let's try n values that are multiples of small numbers
    # and check if a simple circulant works
    
    # Actually, the most reliable approach within time limits:
    # Try to find any valid graph with n < 2048 using a limited search
    
    # Try generalized Petersen graphs GP(n, k) - these are 3-regular, not 4
    # So we need something else
    
    # Let's try: take two copies of a 2-regular graph and connect them
    # to make 4-regular
    
    # Simpler: try circulant graphs C(n, a, b, c, d) for small n
    # A 4-regular circulant on n vertices has edges (i, i+a), (i, i-a), (i, i+b), (i, i-b) mod n
    
    # For girth 10, we need no cycles of length < 10
    # This means: a, b, a+b, a-b, 2a, 2b, etc. must not create short cycles
    
    # Let's try a bounded search over n from 243 to min(2047, deadline)
    # For each n, try a few (a,b) pairs
    
    # This is heuristic but might find something
    
    # Actually, for a 4-regular circulant C(n, a, b):
    # Short cycles can be formed by sequences of +a, -a, +b, -b
    # The girth is determined by the shortest non-trivial relation
    
    # Let's try n = 320 (known best) and see if a simple circulant works
    # C(320, a, b) with gcd(a,320)=1, gcd(b,320)=1
    
    # Try a=1, b=some value
    # For girth >= 10, we need that no combination of up to 9 steps gives 0 mod 320
    # except the trivial back-and-forth
    
    # This is complex. Let me just try a few and verify
    
    import itertools
    
    # Try n = 320
    n_target = 320
    found = None
    
    # Try circulant C(n, a, b) where edges are (i, i±a) and (i, i±b) mod n
    # Need: gcd(a,n)=1, gcd(b,n)=1, a≠b, a≠-b mod n
    # And girth >= 10
    
    # For efficiency, try a=1 and vary b
    for b in range(2, n_target // 2):
        if b == 1 or b == n_target - 1:
            continue
        if (n_target - b) == 1:  # b = n-1 same as b=1
            continue
        
        # Check if this circulant has girth >= 10
        # Build adjacency and check
        adj = {}
        for i in range(n_target):
            neighbors = [(i + 1) % n_target, (i - 1) % n_target, 
                        (i + b) % n_target, (i - b) % n_target]
            # Remove duplicates
            adj[i] = list(set(neighbors))
        
        # Verify 4-regular
        valid = True
        for i in range(n_target):
            if len(adj[i]) != 4:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check girth using BFS
        if _check_girth_fast(adj, n_target, 10):
            edges = []
            for i in range(n_target):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            found = (n_target, edges)
            break
    
    if found is not None:
        return found
    
    # Try other n values if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 5:
        # Try a few other n values
        for n_try in [244, 246, 248, 250, 260, 280, 300, 320, 340, 360]:
            if n_try >= 2048:
                break
            if time.time() > deadline - 5:
                break
            
            for b in range(2, n_try // 2):
                if b == 1 or n_try - b == 1:
                    continue
                
                adj = {}
                for i in range(n_try):
                    neighbors = [(i + 1) % n_try, (i - 1) % n_try, 
                                (i + b) % n_try, (i - b) % n_try]
                    adj[i] = list(set(neighbors))
                
                valid = True
                for i in range(n_try):
                    if len(adj[i]) != 4:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if _check_girth_fast(adj, n_try, 10):
                    edges = []
                    for i in range(n_try):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return (n_try, edges)
    
    return None

def _check_girth_fast(adj, n, g):
    """Fast girth check using BFS from each vertex."""
    max_depth = (g - 1) // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_graph_with_d(k, g, d):
    """Construct LUW graph with specific d."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule generation depends on d
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

def _check_girth(n, edges, g):
    """Check if graph has girth at least g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex up to depth (g-1)//2
    max_depth = (g - 1) // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_graph(k, g):
    # Original fallback
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

def _circulant_construct(n, k, g):
    # Removed or disabled to save time, as it is ineffective for (4,10)
    return None, None

def _circulant_girth(n, a, b):
    # Unused
    return float('inf')
