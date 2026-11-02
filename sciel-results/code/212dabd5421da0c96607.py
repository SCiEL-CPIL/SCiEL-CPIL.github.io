import time
import random
import heapq
import sys

def _generate_random_regular(n, k):
    """Generate a random simple k-regular graph on n vertices using configuration model + 2-switches."""
    if n * k % 2 != 0:
        return None
    
    # 1. Configuration Model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # We need to pair stubs to form edges.
    # To ensure simplicity (no loops/multi-edges), we use a rejection/switching approach.
    # Start with a random pairing.
    edges = set()
    adjacency = [set() for _ in range(n)]
    
    # Random pairing
    random.shuffle(stubs)
    current_edges = []
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u != v: # No loops initially in pairing logic? 
            # But u and v are vertex indices.
            # If u==v, it's a loop.
            current_edges.append((u, v))
        else:
            current_edges.append((u, v)) # Keep for now, we will fix
    
    # This naive pairing often creates loops and multi-edges.
    # We need to clean it up.
    # A better way: Use a 2-switch process to eliminate loops and multi-edges.
    
    # Initialize adjacency from a valid simple graph?
    # It's hard to start simple.
    # Let's try a different approach: 
    # Start with a cycle graph (2-regular) and add edges?
    # No, k=5.
    
    # Let's use the "Barabasi-Albert" style? No, that's not regular.
    
    # I will use a standard algorithm to generate a simple k-regular graph:
    # 1. Start with a random pairing.
    # 2. While there are loops or multi-edges:
    #    Find a loop or multi-edge.
    #    Perform a 2-switch to remove it.
    
    # For simplicity and speed, I will use a known method:
    # Generate a random permutation of stubs, pair them.
    # Then use 2-switches to fix.
    
    # Given time constraints, I will implement a simpler "Greedy Random" construction which is safer.
    # But the directive asks for Edge Switching.
    
    # I will implement the switching on the multigraph.
    
    # Let's represent edges as a list of (u, v) with u < v.
    # And a counter for multi-edges.
    
    # Actually, for n=1116, k=5, the probability of a simple graph from a random pairing is non-negligible, but not 1.
    # We can just retry the pairing until it's simple?
    # Probability of no loops: (1 - 1/n)^{n/2} approx e^{-1/2} ~ 0.6.
    # Probability of no multi-edges is harder.
    
    # I will implement a function that generates a simple k-regular graph.
    # Reference: "Generating random regular graphs"
    
    # I will use the "Giant Component" method? No.
    
    # I will use a simple retry loop for the configuration model, checking simplicity.
    max_attempts = 100
    for _ in range(max_attempts):
        random.shuffle(stubs)
        edges_list = []
        is_simple = True
        edge_counts = {}
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                is_simple = False
                break
            if u > v: u, v = v, u
            if (u, v) in edge_counts:
                is_simple = False
                break
            edge_counts[(u, v)] = 1
            edges_list.append((u, v))
        
        if is_simple:
            # Build adjacency
            adj = [set() for _ in range(n)]
            for u, v in edges_list:
                adj[u].add(v)
                adj[v].add(u)
            # Check degrees (should be k)
            # Return
            return adj
            
    # If failed, return None
    return None

def _girth_check(adj, n, max_girth):
    """Check if girth is at least max_girth. Returns girth if < max_girth, else max_girth."""
    # BFS from each vertex
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Length = dist[u] + dist[v] + 1
                    # Need to trace back to LCA?
                    # For girth, we just need the shortest cycle.
                    # A BFS from start finds the shortest cycle *through* start?
                    # No, BFS from start finds shortest path from start.
                    # A cycle of length L implies a path of length L-1 between two neighbors?
                    # Actually, the standard way to find girth is:
                    # For each edge (u,v), remove it, find shortest path between u and v.
                    # Cycle length = dist + 1.
                    # This is O(E * (V+E)) = O(n * n * k) = O(n^2 k).
                    # For n=1116, 1116^2 * 5 ~ 6 * 10^6. This is fast.
                    pass
        
    # This BFS from start approach is tricky for girth.
    # I will use the edge-removal method.
    min_g = max_girth
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Temporarily remove (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                
                # BFS from u to v
                dist = [-1] * n
                dist[u] = 0
                queue = [u]
                found = False
                while queue:
                    node = queue.pop(0)
                    if node == v:
                        min_g = min(min_g, dist[node] + 1)
                        found = True
                        break
                    if dist[node] >= min_g - 1: # Pruning
                        continue
                    for neighbor in adj[node]:
                        if dist[neighbor] == -1:
                            dist[neighbor] = dist[node] + 1
                            queue.append(neighbor)
                
                # Restore (u,v)
                adj[u].add(v)
                adj[v].add(u)
                
                if min_g < max_girth:
                    return min_g
                    
    return min_g

def construct(k, g):
    # For k=5, g=9
    if k == 5 and g == 9:
        target_n = 1116
        start_time = time.time()
        
        # Try Edge Switching
        # 1. Generate random simple 5-regular graph
        adj = _generate_random_regular(target_n, k)
        if adj is None:
            # Fallback: Return a known valid structure? 
            # If we can't generate, we fail.
            # But we should try to generate one.
            return None
        
        # 2. Check initial girth
        current_girth = _girth_check(adj, target_n, g)
        
        # 3. Switching Loop
        while current_girth < g and time.time() - start_time < 110:
            # Find a short cycle
            # We need to identify the cycle to break it.
            # For simplicity, we perform random 2-switches and accept if girth increases.
            
            # Pick a random edge (u, v)
            u = random.randint(0, target_n - 1)
            if not adj[u]: continue
            v = random.choice(list(adj[u]))
            
            # Pick another random edge (x, y)
            x = random.randint(0, target_n - 1)
            if not adj[x]: continue
            # Ensure (x,y) is not (u,v) or (v,u)
            y_candidates = [y for y in adj[x] if y != u and y != v]
            if not y_candidates: continue
            y = random.choice(y_candidates)
            
            # Check if switch is valid (no loops, no multi-edges)
            # New edges: (u, x) and (v, y)
            # Remove: (u, v) and (x, y)
            
            if x == v or x == u: continue # Loop check
            if y == u or y == v: continue # Loop check
            
            if x in adj[u] or y in adj[v]: continue # Multi-edge check
            
            # Perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            # Check new girth
            new_girth = _girth_check(adj, target_n, g)
            
            if new_girth < current_girth:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].add(v)
                adj[v].add(u)
                adj[x].add(y)
                adj[y].add(x)
            else:
                current_girth = new_girth
                
        if current_girth >= g:
            edges = []
            for i in range(target_n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return target_n, edges
        else:
            # Failed to reach girth g
            return None
            
    # Fallback for other k,g
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Keep the original fallback just in case
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
