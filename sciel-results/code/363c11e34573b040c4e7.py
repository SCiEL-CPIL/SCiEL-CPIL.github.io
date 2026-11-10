"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    """
    Constructs a graph on vertices 0..11.
    First, searches for a 3-regular bipartite graph with minimum distinct cycle lengths.
    Falls back to two disjoint K3,3 components (raw value 2) if no better graph is found.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Known good fallback: Two K3,3
    # Part A: 0, 1, 2, 6, 7, 8
    # Part B: 3, 4, 5, 9, 10, 11
    # Edges for first K3,3 (0-5)
    edges_fallback = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges_fallback.append([u, v])
    # Edges for second K3,3 (6-11)
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges_fallback.append([u, v])
    
    def get_cycle_lengths(edges):
        # Build adjacency list
        adj = {i: [] for i in range(12)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find all simple cycles
        cycles = []
        def dfs(start, current, path, visited):
            if current == start and len(path) > 1:
                # Normalize cycle to avoid duplicates (rotate and reverse)
                # Simple cycle in undirected graph
                cycle = path[:]
                # Rotate to start with smallest node
                min_idx = cycle.index(min(cycle))
                cycle = cycle[min_idx:] + cycle[:min_idx]
                # Reverse if needed to ensure consistent orientation
                # For undirected, cycle and reverse are same.
                # We just need the length.
                cycles.append(len(cycle))
                return
            
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    dfs(start, start, path, visited)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(12):
            # To avoid counting same cycle multiple times, we only start DFS from the smallest node in the cycle?
            # Or just collect all and unique them.
            # Since n=12 is small, we can just collect lengths.
            # However, standard DFS from every node will find cycles multiple times.
            # We just need the SET of lengths.
            visited = set()
            visited.add(start)
            dfs(start, start, [start], visited)
            
        return set(cycles)
        
    def evaluate_graph(edges):
        # Check validity: 3-regular, simple, bipartite (no odd cycles)
        # We generate bipartite graphs, so no odd cycles.
        # Check 3-regular
        degrees = [0] * 12
        for u, v in edges:
            if u == v: return None
            degrees[u] += 1
            degrees[v] += 1
        if any(d != 3 for d in degrees): return None
        
        # Check simple (no multi-edges)
        edge_set = set()
        for u, v in edges:
            key = tuple(sorted((u, v)))
            if key in edge_set: return None
            edge_set.add(key)
            
        # Get cycle lengths
        lengths = get_cycle_lengths(edges)
        if not lengths: return None # Must contain a cycle
        
        return len(lengths), lengths, edges

    best_raw = 2 # Fallback raw value
    best_edges = edges_fallback
    best_lengths = {4, 6}
    
    # Random search for 3-regular bipartite graphs
    # Left: 0..5, Right: 6..11
    # Each left node connects to 3 right nodes.
    # We can represent the graph by 6 permutations of 3 choices?
    # Actually, we can generate random 3-regular bipartite graphs by randomizing the adjacency matrix.
    
    attempts = 0
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        # Generate random 3-regular bipartite graph
        # For each left node i in 0..5, pick 3 distinct right nodes from 6..11
        # We need to ensure the right nodes also have degree 3.
        # This is equivalent to finding a 3-regular bipartite multigraph, but we want simple.
        # We can use a random permutation approach or just reject sampling.
        
        # Reject sampling:
        left_choices = []
        valid = True
        right_degrees = [0] * 6
        for i in range(6):
            # Pick 3 distinct neighbors from 0..5 (representing right nodes 6..11)
            neighbors = random.sample(range(6), 3)
            left_choices.append(neighbors)
            for n in neighbors:
                right_degrees[n] += 1
        
        # Check if right degrees are all 3
        if any(d != 3 for d in right_degrees):
            continue
            
        # Construct edges
        edges = []
        for i in range(6):
            for j in left_choices[i]:
                edges.append([i, j + 6])
                
        # Evaluate
        res = evaluate_graph(edges)
        if res:
            raw, lengths, e = res
            if raw < best_raw:
                best_raw = raw
                best_lengths = lengths
                best_edges = e
                if best_raw == 1:
                    break # Can't do better than 1
    
    # Return the best edges found
    # Ensure types are correct
    final_edges = []
    seen = set()
    for e in best_edges:
        u, v = e
        if u == v: continue
        key = tuple(sorted((u, v)))
        if key not in seen:
            seen.add(key)
            final_edges.append([int(u), int(v)])
            
    return final_edges
# EVOLVE-BLOCK-END
