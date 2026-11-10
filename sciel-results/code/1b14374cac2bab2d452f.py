import time
import sys
from random import Random

def instance_graph(n, p, seed):
    """
    Generates the specific random graph instance used by the verifier.
    """
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    # The verifier logic:
    # edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    # edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    # return sorted(edges)
    
    # We need to replicate this exactly.
    edges = set()
    # Add the path edges
    for i in range(len(order) - 1):
        u, v = order[i], order[i+1]
        edges.add(tuple(sorted((u, v))))
    
    # Add random edges
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
                
    return sorted(edges)

def decompose_graph(edges):
    """
    Decomposes the graph into edge-disjoint cycles and single edges.
    Returns a list of pieces.
    """
    if not edges:
        return []
    
    # Build adjacency list
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
        
    remaining_edges = set(edges)
    pieces = []
    
    # Helper to check if an edge exists in remaining_edges
    def edge_exists(u, v):
        return (u, v) in remaining_edges or (v, u) in remaining_edges

    def remove_edge(u, v):
        if (u, v) in remaining_edges:
            remaining_edges.remove((u, v))
        elif (v, u) in remaining_edges:
            remaining_edges.remove((v, u))
        
    # Heuristic to find a cycle
    # We want to find a cycle that uses as many edges as possible.
    # We will try to find a cycle starting from each vertex with degree > 0.
    # To bias towards longer cycles, we can use a greedy DFS that prefers
    # neighbors with higher remaining degree.
    
    def find_cycle():
        # Pick a starting vertex. Prefer vertices with high degree.
        # To make it deterministic and effective, we can iterate through vertices
        # sorted by degree descending.
        
        # Get current degrees
        degrees = {v: len(adj[v]) for v in adj}
        
        # We need to consider only vertices that are part of remaining edges.
        # Since we remove edges from 'remaining_edges', we should update adj or check remaining_edges.
        # It's safer to rebuild a "live" adjacency or just check remaining_edges in DFS.
        
        # Let's use a DFS on the graph defined by remaining_edges.
        
        # To find a "long" cycle, we can try to find a cycle in the largest
        # biconnected component, but that's complex.
        
        # Simple heuristic:
        # 1. Find a vertex with degree >= 2.
        # 2. Do a DFS to find a cycle.
        # 3. To encourage long cycles, during DFS, sort neighbors by degree descending.
        
        # Let's try multiple starting points if the first cycle is short?
        # Given time limits, let's stick to one pass per removal.
        
        # Find a vertex with degree >= 2 in the remaining graph
        start = None
        for v in adj:
            # Count remaining edges incident to v
            deg = sum(1 for u in adj[v] if edge_exists(v, u))
            if deg >= 2:
                if start is None or deg > sum(1 for u in adj[start] if edge_exists(start, u)):
                    start = v
                    # Just take the first one with max degree to save time, or break early if high enough?
                    # Let's just pick the first one with max degree found so far.
        
        if start is None:
            return None
            
        # DFS to find a cycle
        # State: (current_node, path, visited_in_path)
        # We want to return to start.
        
        # We can use a stack for iterative DFS or recursion.
        # n is small (<=30), recursion is fine.
        
        def dfs(node, path, visited):
            if node == start and len(path) > 2:
                return path
            for neighbor in sorted(adj[node], key=lambda x: -sum(1 for u in adj[x] if edge_exists(x, u))):
                if not edge_exists(node, neighbor):
                    continue
                if neighbor == start and len(path) > 2:
                    return path
                if neighbor not in visited:
                    visited.add(neighbor)
                    result = dfs(neighbor, path + [neighbor], visited)
                    if result:
                        return result
                    visited.remove(neighbor)
            return None
        
        # To improve cycle length, we might want to avoid closing the cycle too early.
        # The standard DFS above closes as soon as it hits 'start'.
        # This might find a short cycle.
        
        # Alternative: Find any cycle, then try to "extend" it?
        # Or: Find a cycle in the graph.
        
        # Let's try a different approach:
        # 1. Find a cycle C.
        # 2. Remove C.
        
        # The DFS above is standard. Let's see if we can make it find longer cycles.
        # One trick: Don't allow closing the cycle until all other options are exhausted?
        # That's basically finding a Hamiltonian cycle in the component, which is NP-hard.
        
        # Let's stick to the greedy DFS but maybe try a few different starting vertices
        # if the first cycle is very short (e.g., length 3).
        
        # For now, let's just use the DFS.
        cycle = dfs(start, [start], {start})
        return cycle
        
    # Main loop
    while remaining_edges:
        cycle = find_cycle()
        if cycle:
            # Remove edges of the cycle
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                remove_edge(u, v)
            # The closing edge is implied by the cycle definition, 
            # but we must remove it from remaining_edges too.
            u, v = cycle[-1], cycle[0]
            remove_edge(u, v)
            
            pieces.append(cycle)
        else:
            # No more cycles. All remaining edges are single edges.
            for u, v in remaining_edges:
                pieces.append([u, v])
            break
            
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback to trivial solution if timeout
            edges = instance_graph(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance_graph(n, p, seed)
        pieces = decompose_graph(edges)
        results.append(pieces)
        
    return results
