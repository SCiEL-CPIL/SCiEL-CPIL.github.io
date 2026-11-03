import random
import numpy as np
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using random regular graph
    generation followed by local search (2-switches) to increase girth.
    """
    # Target number of vertices
    # For k=5, g=9, lower bound 428. Let's aim for ~500-600.
    # The random regular graph will have small girth initially.
    # We will try to "fix" it.
    
    # If the task is k=5, g=9, we know n=1116 is best known.
    # Can we find a smaller one? The lower bound is 428.
    # Let's try n around 500-800.
    
    # We will try a few different n values.
    # Start with a value close to the Moore bound but slightly higher to allow existence.
    # Moore bound for k=5, g=9 is 428.
    # Let's try n = 500, 550, 600, 650, 700.
    
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead
    
    for target_n in [500, 520, 550, 600, 650, 700, 750, 800]:
        if time.time() - start_time > time_limit:
            break
            
        # 1. Generate a random k-regular graph
        graph = _generate_random_regular(target_n, k)
        if graph is None:
            continue
        
        # 2. Initial Girth Check
        current_girth = _compute_girth(graph)
        
        # 3. Local Search: 2-Switches to increase girth
        # We want to reach girth g.
        # If current_girth >= g, we are done.
        if current_girth >= g:
            return _finalize(graph)
            
        best_girth = current_girth
        best_graph = [list(neigh) for neigh in graph]
        
        # Optimization: Precompute adjacency sets for fast lookup
        adj_sets = [set(neigh) for neigh in graph]
        
        max_iterations = 50000 # Adjust based on time
        for i in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            # Perform a random 2-switch
            # Pick two random edges
            # To ensure we pick distinct edges, we can pick random u, v in edge list
            # But maintaining an edge list is costly.
            # Alternative: Pick random u, then random v in adj[u] (u<v to avoid dup? No, just any)
            # Then pick random x, then random y in adj[x] (x!=u, y!=v, etc)
            
            # Let's pick random u
            u = random.randint(0, target_n - 1)
            # Pick random v from adj[u]
            if not graph[u]:
                continue
            v = random.choice(graph[u])
            
            # Pick random x (x != u, x != v)
            x = random.randint(0, target_n - 1)
            if x == u or x == v:
                continue
            # Pick random y from adj[x]
            if not graph[x]:
                continue
            y = random.choice(graph[x])
            
            # Check validity of switch (u,v) and (x,y) -> (u,x) and (v,y)
            # Conditions:
            # 1. u != x, v != y (no loops)
            # 2. x != v, y != u (no loops)
            # 3. (u,x) not in adj[u] (no multi-edge)
            # 4. (v,y) not in adj[v] (no multi-edge)
            # 5. u != v, x != y (already true for simple graph edges)
            
            if x == u or y == v or x == v or y == u:
                continue
            if x in adj_sets[u] or y in adj_sets[v]:
                continue
                
            # Apply switch
            # Remove (u,v) and (x,y)
            graph[u].remove(v)
            graph[v].remove(u)
            graph[x].remove(y)
            graph[y].remove(x)
            adj_sets[u].discard(v)
            adj_sets[v].discard(u)
            adj_sets[x].discard(y)
            adj_sets[y].discard(x)
            
            # Add (u,x) and (v,y)
            graph[u].append(x)
            graph[x].append(u)
            graph[v].append(y)
            graph[y].append(v)
            adj_sets[u].add(x)
            adj_sets[x].add(u)
            adj_sets[v].add(y)
            adj_sets[y].add(v)
            
            # Check new girth
            # Optimization: We can check if the shortest cycle is broken.
            # But computing full girth is safer.
            new_girth = _compute_girth(graph)
            
            if new_girth > best_girth:
                best_girth = new_girth
                best_graph = [list(neigh) for neigh in graph]
                if new_girth >= g:
                    return _finalize(graph)
            else:
                # Revert
                # Remove (u,x) and (v,y)
                graph[u].remove(x)
                graph[x].remove(u)
                graph[v].remove(y)
                graph[y].remove(v)
                adj_sets[u].discard(x)
                adj_sets[x].discard(u)
                adj_sets[v].discard(y)
                adj_sets[y].discard(v)
                
                # Add (u,v) and (x,y)
                graph[u].append(v)
                graph[v].append(u)
                graph[x].append(y)
                graph[y].append(x)
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                adj_sets[x].add(y)
                adj_sets[y].add(x)
                
        # If we didn't reach g, try next n
        # Or return the best we found? No, must satisfy girth g.
        # If we fail for this n, try larger n.
        
    # Fallback: Use LUW if local search fails for all n
    return _fallback_luw(k, g)

def _finalize(graph):
    n = len(graph)
    edges = []
    for i in range(n):
        for j in graph[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

def _generate_random_regular(n, k):
    """Generate a simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        for j in range(k):
            stubs.append(i)
            
    # We need to pair them up without loops or multi-edges.
    # This is a standard problem.
    # We can use the "configuration model" with rejection.
    # For n=500, k=5, this is fast.
    
    for _ in range(100):
        random.shuffle(stubs)
        edges_set = set()
        adjacency = [set() for _ in range(n)]
        valid = True
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges_set:
                valid = False
                break
            edges_set.add((min(u,v), max(u,v)))
            adjacency[u].add(v)
            adjacency[v].add(u)
            
        if valid and all(len(a) == k for a in adjacency):
            return [list(a) for a in adjacency]
            
    return None

def _compute_girth(graph):
    """Compute the girth of the graph using BFS from each node."""
    n = len(graph)
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start
        # dist[node] = distance from start
        # prev[node] = previous node in BFS tree (for path reconstruction if needed, but we just need dist)
        # Actually, to find cycles, we need to check non-tree edges.
        # A cycle exists if there is an edge (u,v) such that dist[u] + dist[v] + 1 >= 3?
        # No, that's for finding *a* cycle.
        # To find the *shortest* cycle through `start`:
        # We do BFS. If we encounter an edge (u,v) where v is already visited:
        # The cycle length is dist[u] + dist[v] + 1.
        # However, this might not be the shortest cycle through start if the BFS tree is not unique?
        # Actually, BFS gives shortest distances.
        # The shortest cycle containing `start` is min(dist[u] + dist[v] + 1) for all edges (u,v) where u,v are in the BFS tree?
        # Wait, if (u,v) is a back edge, the cycle is dist[u] + dist[v] + 1.
        # Is it possible that a shorter cycle exists that doesn't use the BFS tree edges in that way?
        # Yes, but BFS explores in order of distance.
        # The first time we see a "collision" (an edge connecting two visited nodes) gives a candidate.
        # To be exact:
        # Run BFS from start.
        # For each node u popped, for each neighbor v:
        #   If v is not visited: visit v, dist[v] = dist[u]+1, parent[v]=u.
        #   Else if v != parent[u]: 
        #       cycle_len = dist[u] + dist[v] + 1
        #       min_girth = min(min_girth, cycle_len)
        
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in graph[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # v is visited
                    # If v is not the parent of u, we found a cycle
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            
        # Optimization: If min_girth is already 3, we can break early?
        # No, we need the global min. But if we find 3, it's the min possible.
        if min_girth == 3:
            return 3
            
    return min_girth if min_girth != float('inf') else 2 * n # Fallback

def _fallback_luw(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
