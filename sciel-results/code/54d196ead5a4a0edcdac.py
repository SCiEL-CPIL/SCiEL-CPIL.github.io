import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: return empty or trivial if we ran out of time (unlikely)
            results.append([])
            continue
            
        graph = generate_instance(n, p, seed)
        pieces = decompose(graph)
        results.append(pieces)
        
    return results

def generate_instance(n, p, seed):
    # Exact copy from verifier logic
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i+1]
        if u > v: u, v = v, u
        edges.add((u, v))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges

def decompose(edges):
    remaining_edges = set(edges)
    n = max([max(u, v) for u, v in edges] + [0]) + 1
    
    pieces = []
    
    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    while remaining_edges:
        # Find a long cycle
        cycle = find_long_cycle(adj, remaining_edges)
        
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (u, v) if u < v else (v, u)
                remaining_edges.remove(e)
                adj[u].remove(v)
                adj[v].remove(u)
        else:
            # No cycle left, add remaining edges as single pieces
            for u, v in remaining_edges:
                pieces.append([u, v])
            break
            
    return pieces

def find_long_cycle(adj, edges):
    # Heuristic to find a long simple cycle
    # Try starting from each vertex, do a greedy DFS
    best_cycle = None
    best_len = 0
    
    # To speed up, we can just try a few random starts or all starts
    # n is small, so all starts is fine
    for start in adj:
        if not adj[start]:
            continue
        cycle = dfs_find_cycle(start, adj, edges)
        if cycle and len(cycle) > best_len:
            best_cycle = cycle
            best_len = len(cycle)
            
    return best_cycle

def dfs_find_cycle(start, adj, edges):
    # Recursive DFS with backtracking to find a cycle
    # This might be slow if not careful, but n=30 is small.
    # We can limit the depth or use a greedy approach.
    
    # Greedy approach:
    # Path: list of vertices
    # Visited: set of vertices in current path
    
    # Actually, a simple DFS that returns the first cycle found might be short.
    # We want a LONG cycle.
    # Let's use a heuristic: always pick the neighbor with the highest degree that is not visited.
    
    path = [start]
    visited = {start}
    
    def backtrack(u, path, visited):
        # Check if we can close the cycle
        # If neighbor is start and len(path) >= 3
        # Or if neighbor is in path (cycle found)
        
        # To find a LONG cycle, we should try to extend the path as much as possible.
        # If we hit a dead end, backtrack.
        
        # Heuristic: Sort neighbors by degree (descending) to keep options open?
        # Or just try all?
        
        # For n=30, backtracking might explode.
        # Let's use a simpler heuristic: 
        # 1. Start at u.
        # 2. Pick a neighbor v not in visited.
        # 3. If no such v, check if any neighbor is in visited (cycle).
        
        # Let's try a randomized greedy approach with a few restarts.
        pass

    # Better: Use a simple iterative deepening or just a greedy walk.
    # Given the small size, let's use a randomized greedy walk that tries to maximize length.
    
    # We'll implement a simple recursive backtracking but with a node limit or just rely on n being small.
    # Actually, finding *a* cycle is easy. Finding the *longest* is NP-hard.
    # But we just need to minimize pieces.
    # Any cycle is better than edges.
    # A 3-cycle covers 3 edges (1 piece).
    # A 10-cycle covers 10 edges (1 piece).
    # So longer is significantly better.
    
    # Let's use a heuristic: 
    # Try to find a cycle by walking.
    # If stuck, backtrack.
    
    # To avoid exponential blowup, we can limit the search depth or use a greedy heuristic.
    # Greedy: Always go to the neighbor with the highest degree that is unvisited.
    # If stuck, backtrack.
    
    def greedy_dfs(u, path, visited):
        # Check for cycle
        for v in adj[u]:
            if v == start and len(path) >= 3:
                return path[:]
            if v in visited:
                # Found a cycle, but not necessarily starting at start.
                # We can extract the cycle part.
                # But for simplicity, let's just return if we hit start.
                # If we hit another visited node, we have a cycle.
                # Let's just focus on cycles closing at start for simplicity of extraction, 
                # or handle general cycles.
                # Actually, if v is in visited, we can find the index of v in path and extract.
                # But let's stick to "cycle closing at start" for the main loop, 
                # and if we find a cycle not closing at start, we can still use it.
                pass
        
        # Try to extend
        neighbors = [v for v in adj[u] if v not in visited]
        # Sort by degree to prefer high connectivity
        neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
        
        for v in neighbors:
            path.append(v)
            visited.add(v)
            result = greedy_dfs(v, path, visited)
            if result:
                return result
            path.pop()
            visited.remove(v)
            
        return None

    return greedy_dfs(start, path, visited)
