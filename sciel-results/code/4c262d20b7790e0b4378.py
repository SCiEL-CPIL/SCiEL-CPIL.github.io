import time
import random
import sys
from collections import deque

def _seed_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _check_girth(n, edges, g):
    # Check if any cycle is shorter than g
    # Use BFS from each vertex, tracking parent to avoid backtracking
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        # dist[v][parent] = distance
        # We want to find a cycle of length < g
        # A cycle of length L corresponds to a path of length L-1 in the non-backtracking sense?
        # Standard BFS on the graph, but we must not go back to the immediate parent.
        # Actually, simpler: BFS from start, keeping track of the path.
        # If we find a node that is already visited, the cycle length is dist[u] + dist[v] + 1.
        # But this finds *any* cycle. We need to ensure it's simple.
        # BFS tree paths are simple. So if we find a cross edge (u,v) where v is visited and u != parent[v] and v != parent[u],
        # the cycle length is dist[u] + dist[v] + 1.
        # We must check if this length < g.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # This is valid if v is not the parent of u and u is not the parent of v (handled by skip above)
                    # And we must ensure it's not the same edge (handled by simple graph)
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # For k=3, g=16, Moore bound is 512. Known best is 936.
    # We try to find a graph with n close to the known best.
    # If k != 3 or g != 16, we might just use the seed to be safe, 
    # but the task says "evaluator passes this task's degree k and girth g".
    # The prompt implies k=3, g=16 is the specific case.
    
    # Strategy:
    # 1. Try Random Greedy for n in [512, 1000].
    # 2. If successful, return.
    # 3. If time runs out, fallback to seed.
    
    if k == 3 and g == 16:
        # Try to construct a better graph
        # The seed graph has ~35000 vertices.
        # We want n < 35000.
        # Let's try n = 936 first.
        
        for n_try in [936, 940, 950, 1000]:
            if time.time() > deadline:
                break
            
            # Random Greedy Construction
            # We need a 3-regular graph on n_try vertices.
            # Sum of degrees = 3 * n_try. Must be even. 3*n is even => n is even.
            if n_try % 2 != 0:
                n_try += 1
            
            success = False
            # Multiple restarts for Random Greedy
            max_restarts = 50
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Initialize
                adj = [[] for _ in range(n_try)]
                edges = []
                
                # List of vertices with remaining degree
                # We will pick random pairs
                # To avoid short cycles, we check girth incrementally?
                # Checking girth after every edge is too slow (O(n^2)).
                # Instead, we construct the graph, then check girth.
                # If girth < g, we try to fix it via local search.
                
                # Simple random 3-regular graph generation
                # Use the "configuration model" or just random pairing
                # Ensure no multi-edges and no loops.
                
                try:
                    # Generate a random 3-regular graph
                    # 1. Create list of stubs: [0]*3, [1]*3, ...
                    stubs = []
                    for i in range(n_try):
                        stubs.extend([i, i, i])
                    random.shuffle(stubs)
                    
                    # Pair up stubs
                    # This might create loops or multi-edges.
                    # We'll build the edge list and check validity.
                    temp_edges = []
                    used = [False] * len(stubs)
                    
                    for i in range(len(stubs)//2):
                        if used[i]:
                            continue
                        j = random.randint(i+1, len(stubs)-1)
                        while used[j]:
                            j = random.randint(i+1, len(stubs)-1)
                        
                        u, v = stubs[i], stubs[j]
                        if u == v:
                            # Loop, retry or swap
                            # For simplicity, if we get too many loops, restart
                            pass 
                        else:
                            # Check for multi-edge
                            # This is O(E) per edge, too slow.
                            # We'll just build and check at the end.
                            temp_edges.append((u, v))
                        used[i] = True
                        used[j] = True
                    
                    # Check if temp_edges forms a simple 3-regular graph
                    # Count degrees
                    deg = [0] * n_try
                    edge_set = set()
                    valid = True
                    for u, v in temp_edges:
                        if u == v:
                            valid = False
                            break
                        if (min(u,v), max(u,v)) in edge_set:
                            valid = False
                            break
                        edge_set.add((min(u,v), max(u,v)))
                        deg[u] += 1
                        deg[v] += 1
                    
                    if not valid or any(d != 3 for d in deg):
                        continue
                    
                    # Check Girth
                    if _check_girth(n_try, temp_edges, g):
                        return n_try, temp_edges
                    success = True # We found a valid 3-reg graph, but girth was too small.
                    # We could try local search here, but it's complex.
                    # For now, just restart.
                    
                except Exception:
                    continue
            
            # If we found a valid 3-reg graph but girth was insufficient, 
            # we might try a slightly larger n or just continue.
            
        # If Random Greedy failed to find a valid graph in time
        pass
    
    # Fallback: Use the seed construction
    # This ensures we return a valid graph (though large)
    n, edges = _seed_construct(k, g)
    return n, edges
