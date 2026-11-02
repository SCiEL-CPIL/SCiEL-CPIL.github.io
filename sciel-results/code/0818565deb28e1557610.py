"""Construct a 4-regular graph of girth 10 (the (4,10)-cage)."""
import numpy as np
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    For k=4, g=10, attempts to find a small cage (target 320 vertices) 
    using MCMC search. Falls back to the Lazebnik-Ustimenko-Woldar 
    construction (8192 vertices) if search fails.
    """
    if k == 4 and g == 10:
        # Try to find a smaller graph first
        result = _search_cage_4_10()
        if result:
            return result
        else:
            return _construct_lw_4_10()
    else:
        return _construct_general(k, g)

def _search_cage_4_10():
    """Search for a 4-regular graph of girth 10 with small N."""
    start_time = time.time()
    time_limit = 100.0  # Leave 20s for overhead/fallback
    
    # Target sizes: known cage is 320. Try nearby.
    target_sizes = [320, 340, 360, 400, 500]
    
    for n in target_sizes:
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a valid graph for this n
        graph = _find_graph_mcmc(n, time_limit - (time.time() - start_time))
        if graph:
            return n, graph

    return None

def _find_graph_mcmc(n, time_budget):
    """Use MCMC to find a 4-regular graph of girth >= 10 on n vertices."""
    start_time = time.time()
    
    # 1. Generate initial random 4-regular graph
    adj = _generate_random_regular_graph(n, 4)
    if not adj:
        return None
    
    # 2. MCMC Loop
    best_girth = 0
    for _ in range(10000): # Max iterations
        if time.time() - start_time > time_budget:
            break
        
        # Check girth
        girth = _calculate_girth(adj, n)
        
        if girth >= 10:
            # Convert adjacency list to edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return edges
        
        # If girth < 10, try to improve
        # Find a short cycle and break it
        # For simplicity, pick a random edge and try to swap it
        # This is a crude heuristic, but might work
        
        # Pick a random edge (u, v)
        u = random.randint(0, n-1)
        if not adj[u]:
            continue
        v = random.choice(adj[u])
        
        # Pick another random edge (w, x)
        w = random.randint(0, n-1)
        if w == u or w == v:
            continue
        if not adj[w]:
            continue
        x = random.choice(adj[w])
        if x == u or x == v or x == w:
            continue
        
        # Try swap: remove (u,v) and (w,x), add (u,w) and (v,x)
        # Check for multi-edges
        if w in adj[u] or x in adj[v]:
            continue
        
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].append(w)
        adj[w].append(u)
        adj[v].append(x)
        adj[x].append(v)

    return None

def _generate_random_regular_graph(n, k):
    """Generate a random simple k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for _ in range(100):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adj = {i: [] for i in range(n)}
        valid = True
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            
            adj[u].append(v)
            adj[v].append(u)
        
        if valid:
            return adj
            
    return None

def _calculate_girth(adj, n):
    """Calculate the girth of the graph. Returns 0 if not found (should not happen for connected regular graphs)."""
    min_girth = float('inf')
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= 10:
                            return min_girth # Early exit if we find a short one
        
        if min_girth <= 10:
            return min_girth
            
    return min_girth if min_girth != float('inf') else 0

def _construct_lw_4_10():
    """Construct the 8192-vertex (4,10)-cage using LW construction."""
    q = 4
    d = 5
    
    # GF(4) arithmetic
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: mul_table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    n = len(order)
    return n, edges

def _construct_general(k, g):
    """Fallback for other parameters."""
    # Simple Moore bound estimation
    if g % 2 == 0:
        half_g = g // 2
        n_min = 2 * sum((k-1)**i for i in range(half_g))
    else:
        half_g = (g - 1) // 2
        n_min = 1 + k * sum((k-1)**i for i in range(half_g))
        
    n = max(n_min, 2 * k)
    
    # Return a trivial graph (might not satisfy girth, but valid interface)
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if j < n - i:
                edges.append((i, (i + j) % n))
                
    return n, edges
# EVOLVE-BLOCK-END
