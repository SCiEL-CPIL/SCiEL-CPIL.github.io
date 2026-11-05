import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Heuristic search for smaller graphs
    # The LUW graph for k=5, g=9 has 1250 vertices.
    # We try to find graphs with N < 1250.
    # Moore bound for k=5, g=9 is 426.
    # We scan N downwards from 1200 to 500.
    
    # Determine search range based on k, g
    # For general k, g, LUW size is 2 * k^d where d ~ g-5.
    # We just try a range of N.
    
    max_search_n = best_n - 50
    min_search_n = max(100, best_n // 2)
    
    # Step size for N
    step = 25
    
    for N in range(max_search_n, min_search_n - 1, -step):
        if time.time() > deadline:
            break
        
        # Allocate time for this N
        # Prioritize smaller N? Or just sequential?
        # Sequential is fine. If we find one at 1200, we try 1175, etc.
        
        n_alloc = 2.0  # 2 seconds per N value
        local_deadline = min(deadline, time.time() + n_alloc)
        
        found_edges = _try_find_graph(k, g, N, local_deadline)
        
        if found_edges is not None:
            best_n = N
            best_edges = found_edges
            # Do not break, try to find even smaller
            # But if we are getting close to the lower bound, it might take longer.
            # We'll just let the loop run.
            
    return best_n, best_edges

def _construct_luw(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
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

def _try_find_graph(k, g, N, deadline):
    """
    Attempt to find a k-regular graph on N vertices with girth >= g
    using random initialization and local search.
    """
    if N < 2 * k:
        return None
        
    # 1. Generate random k-regular graph
    graph = _generate_random_regular(k, N)
    if graph is None:
        return None
        
    adj = _graph_to_adj(graph, N)
    
    # 2. Local Search
    # We want to eliminate cycles of length < g.
    # We use a greedy approach: find a short cycle, swap edges to break it.
    
    for _ in range(10000): # Max iterations
        if time.time() > deadline:
            return None
            
        cycles = _find_short_cycles(adj, N, g)
        
        if not cycles:
            # Success
            return _adj_to_edges(adj, N)
        
        # Pick a random cycle
        cycle = random.choice(cycles)
        
        # Pick a random edge in the cycle to break
        # Cycle is a list of vertices [v0, v1, ..., vm, v0]
        # Edges are (v0,v1), (v1,v2), ...
        edge_idx = random.randint(0, len(cycle) - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % len(cycle)]
        
        # Find another edge to swap with
        # We need an edge (x, y) such that swapping (u,v) and (x,y)
        # with (u,x) and (v,y) (or u,y and v,x) is valid and breaks the cycle.
        
        # Strategy: Pick a random edge (x,y) from the graph.
        # To pick a random edge, pick a random node x, then a random neighbor y.
        x = random.randint(0, N - 1)
        if not adj[x]:
            continue
        y = random.choice(adj[x])
        
        # Check if (x,y) is the same as (u,v)
        if (x == u and y == v) or (x == v and y == u):
            continue
            
        # Check if swap is valid (no loops, no multi-edges)
        # Swap (u,v) and (x,y) -> (u,x) and (v,y)
        # Must ensure u!=x, u!=y, v!=x, v!=y
        if u in (x, y) or v in (x, y):
            # Try the other pairing: (u,y) and (v,x)
            if u == x or u == y or v == x or v == y:
                continue
            
        # Perform swap
        # Remove (u,v) and (x,y)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Add (u,x) and (v,y)
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
        # Check if this swap reduced the number of short cycles
        # We don't strictly need to check count, just if it's still invalid.
        # But to avoid oscillation, we can check if the specific cycle is broken.
        # For now, greedy: just keep the swap. 
        # To prevent getting stuck, we could revert if cycle count increases.
        # Let's add a simple check: if the new graph has MORE short cycles, revert.
        
        new_cycles = _find_short_cycles(adj, N, g)
        
        if len(new_cycles) > len(cycles):
            # Revert
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
    return None

def _generate_random_regular(k, N):
    """
    Generate a random k-regular graph using the configuration model.
    Returns a list of edges or None if failed.
    """
    for attempt in range(100):
        stubs = []
        for i in range(N):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        seen = set()
        valid = True
        
        for i in range(0, N * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in seen or (v, u) in seen:
                valid = False
                break
            seen.add((u, v))
            edges.append((u, v))
        
        if valid:
            return edges
            
    return None

def _graph_to_adj(edges, N):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _adj_to_edges(adj, N):
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _find_short_cycles(adj, N, g):
    """
    Find all cycles of length < g.
    Returns a list of cycles (each cycle is a list of vertices).
    """
    cycles = []
    # BFS from each node
    for start in range(N):
        # BFS
        visited = {start: 0}
        parent = {start: -1}
        queue = [(start, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            
            if depth >= g - 1:
                continue
                
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                    
                if neighbor in visited:
                    # Cycle found
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [node]
                        curr = node
                        while curr != neighbor:
                            curr = parent[curr]
                            cycle.append(curr)
                        # neighbor is the other end
                        # We need to traverse from neighbor back to LCA?
                        # Simple reconstruction:
                        # Path 1: start -> ... -> neighbor
                        # Path 2: start -> ... -> node -> neighbor
                        # This reconstruction is complex in BFS tree.
                        # For simplicity, we just record that a cycle exists.
                        # We need the actual edges to break it.
                        # Let's store the path.
                        
                        path1 = []
                        curr = neighbor
                        while curr != -1:
                            path1.append(curr)
                            curr = parent[curr]
                        path1.reverse()
                        
                        path2 = []
                        curr = node
                        while curr != -1:
                            path2.append(curr)
                            curr = parent[curr]
                        path2.reverse()
                        
                        # Find LCA of neighbor and node in the BFS tree
                        # Since it's a tree, LCA is the first common ancestor.
                        # Actually, the cycle is path1 + path2 reversed?
                        # No.
                        # The cycle is the path from start to neighbor, plus path from start to node, plus edge node-neighbor.
                        # This is not a simple cycle in the graph if they diverge.
                        # Wait, BFS tree paths from start are unique.
                        # The cycle is formed by the unique path in the tree from start to neighbor,
                        # and the unique path in the tree from start to node,
                        # and the edge (node, neighbor).
                        # This forms a cycle of length len(path1) + len(path2) - 2*len(path_to_lca) + 1.
                        # This is getting complicated to reconstruct exactly.
                        
                        # Simplified: Just return a dummy cycle marker?
                        # No, we need to break it.
                        # We can just pick the edge (node, neighbor) to break.
                        # The cycle involves this edge.
                        cycles.append([node, neighbor])
                        
                else:
                    visited[neighbor] = depth + 1
                    parent[neighbor] = node
                    queue.append((neighbor, depth + 1))
                    
    return cycles
