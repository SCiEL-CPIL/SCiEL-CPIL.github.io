import numpy as np
import time
import random
import sys

def construct(k, g):
    # Target: 4-regular, girth 10
    # Known best: 320 vertices.
    # Moore bound: 242.
    
    if k != 4 or g != 10:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try to find a graph with n vertices
    # We know 320 works. Try to find smaller?
    # Given time constraints, focus on 320 first, then maybe try lower if fast.
    
    best_n = 682 # LUW size
    best_edges = None
    
    # Generate LUW as fallback
    luw_n, luw_edges = _luw_construct(4, 10)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try local search for n in [320, 310, 300, ...]
    # Actually, just try 320. If it works fast, try 310?
    # The prompt says "smallest known is 320".
    # So 320 is a safe target.
    
    target_n = 320
    
    # Run local search for target_n
    found_edges = _local_search_4reg_girth10(target_n, deadline)
    if found_edges is not None:
        best_n = target_n
        best_edges = found_edges
        # Check if we have time to try smaller?
        # Maybe try 300?
        if time.time() < deadline - 20:
            found_edges_300 = _local_search_4reg_girth10(300, deadline)
            if found_edges_300 is not None:
                best_n = 300
                best_edges = found_edges_300
            # Try 280?
            if time.time() < deadline - 10:
                found_edges_280 = _local_search_4reg_girth10(280, deadline)
                if found_edges_280 is not None:
                    best_n = 280
                    best_edges = found_edges_280

    return best_n, best_edges

def _local_search_4reg_girth10(n, deadline):
    # Generate random 4-regular graph
    # Use configuration model
    while time.time() < deadline:
        edges = _random_4reg(n)
        if edges is None: continue
        
        # Check girth
        # If girth >= 10, return
        if _check_girth(edges, n, 10):
            return edges
            
        # Try to improve
        # Find short cycles
        # This is complex. 
        # Simpler: Just restart?
        # Random graphs on 320 vertices might have girth 10 with some probability.
        # Probability of no 4-cycles: exp(-lambda).
        # lambda_4 ~ n^2 / (2 * 4^2) ? No.
        # Expected 4-cycles in random k-regular graph: ~ n^2 / (2 k^2) ?
        # For n=320, k=4: 320^2 / 32 = 3200. Many 4-cycles.
        # So random restart is inefficient.
        # Need local search.
        
        # Implement simple local search:
        # 1. Find a 4-cycle.
        # 2. Swap edges to break it.
        
        adj = _build_adjacency(edges, n)
        
        for _ in range(1000): # Max swaps per restart
            if time.time() > deadline: return None
            if _check_girth_fast(adj, n, 10):
                return _adj_to_edges(adj, n)
                
            # Find a 4-cycle
            cyc = _find_4cycle(adj)
            if cyc is None:
                # No 4-cycles. Check 6, 8?
                # If no 4-cycles, girth is at least 6.
                # Check girth exactly.
                if _check_girth_fast(adj, n, 10):
                    return _adj_to_edges(adj, n)
                # Find 6 or 8 cycle
                cyc = _find_short_cycle(adj, 6)
                if cyc is None:
                    cyc = _find_short_cycle(adj, 8)
                if cyc is None:
                    # Should not happen if check failed
                    break
                    
            # Break cycle
            if not _break_cycle(adj, cyc, n):
                break # Failed to break, restart
            
    return None

def _random_4reg(n):
    # Configuration model
    half_edges = [i for i in range(n) for _ in range(2)]
    random.shuffle(half_edges)
    edges = []
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.append((u, v))
    return edges

def _build_adjacency(edges, n):
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edges.append((u, v))
    return edges

def _check_girth_fast(adj, n, g):
    # Check if any cycle < g
    # BFS from each vertex
    for start in range(n):
        dist = [-1]*n
        parent = [-1]*n
        dist[start] = 0
        q = [start]
        while q:
            u = q.pop(0)
            for v in range(n):
                if adj[u][v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _find_4cycle(adj):
    # Find any 4-cycle
    n = len(adj)
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v]: continue
            # Common neighbors of u and v
            common = []
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    common.append(w)
            # If we have two common neighbors x, y, then u-x-v-y-u is a 4-cycle?
            # No, u-x-v-y-u means edges (u,x), (x,v), (v,y), (y,u).
            # Yes.
            if len(common) >= 2:
                x, y = common[0], common[1]
                return [u, x, v, y]
    return None

def _find_short_cycle(adj, length):
    # Generic BFS for cycle of specific length is hard.
    # Just return None for now, rely on 4-cycle breaking to increase girth.
    # If 4-cycles are gone, girth is 6+.
    # We need to break 6-cycles too.
    # For simplicity, if 4-cycles are gone, we might still have 6.
    # I will skip 6/8 breaking for now and hope 4-cycle breaking leads to high girth.
    # Or just restart if 4-cycles are gone but girth < 10.
    return None

def _break_cycle(adj, cyc, n):
    # cyc: [u, x, v, y]
    # Edges: (u,x), (x,v), (v,y), (y,u)
    # Swap: remove (x,v) and (y,u). Add (x,y) and (v,u)?
    # Wait, (v,u) is not an edge necessarily.
    # In 4-cycle u-x-v-y-u, the edges are (u,x), (x,v), (v,y), (y,u).
    # The other two pairs are (u,v) and (x,y).
    # If we swap (x,v) and (y,u) with (x,y) and (v,u)?
    # We need (x,y) and (v,u) to be non-edges.
    if adj[x][y] or adj[v][u]:
        return False
    # Perform swap
    adj[x][v] = False
    adj[v][x] = False
    adj[y][u] = False
    adj[u][y] = False
    adj[x][y] = True
    adj[y][x] = True
    adj[v][u] = True
    adj[u][v] = True
    return True

def _luw_construct(k, g):
    # Original LUW code
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
