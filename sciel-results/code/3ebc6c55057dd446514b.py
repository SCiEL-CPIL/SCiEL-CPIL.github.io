"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(list)
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
        key = (min(u, v), max(u, v))
        edge_to_idx[key] = i
    return adj, edge_to_idx

def _is_connected(vertices, adj, used_edges):
    """Check if the graph induced by vertices with edges not in used_edges is connected."""
    if not vertices:
        return True
    
    vset = set(vertices)
    start = next(iter(vset))
    visited = {start}
    stack = [start]
    
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in vset and v not in visited:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    visited.add(v)
                    stack.append(v)
    
    return len(visited) == len(vset)

def _find_ham_cycle(vertices, adj, edge_to_idx, used_mask, deadline, rng):
    """Find a Hamiltonian cycle on the given vertex set using DFS with backtracking and bitmasks."""
    n = len(vertices)
    if n < 3:
        return None
    
    vset = set(vertices)
    start = vertices[0]
    
    # Precompute adjacency lists
    adj_list = {u: list(v) for u, v in adj.items()}
    
    def dfs(current, visited_mask, path):
        if time.time() > deadline:
            return None
            
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj_list[current]:
                e_key = (min(current, start), max(current, start))
                if e_key in edge_to_idx:
                    e_idx = edge_to_idx[e_key]
                    if not (used_mask >> e_idx) & 1:
                        return path
            return None
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb in adj_list[current]:
            if nb in vset and not (visited_mask >> nb) & 1:
                e_key = (min(current, nb), max(current, nb))
                if e_key in edge_to_idx:
                    e_idx = edge_to_idx[e_key]
                    if not (used_mask >> e_idx) & 1:
                        unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Sort by degree in remaining graph (Warnsdorff's heuristic)
        def heuristic_key(x):
            deg = 0
            for y in adj_list[x]:
                if y in vset and not (visited_mask >> y) & 1:
                    e_key = (min(x, y), max(x, y))
                    if e_key in edge_to_idx:
                        e_idx = edge_to_idx[e_key]
                        if not (used_mask >> e_idx) & 1:
                            deg += 1
            return (deg, rng.random())
            
        unvisited_neighbors.sort(key=heuristic_key)
        
        for nb in unvisited_neighbors:
            new_visited = visited_mask | (1 << nb)
            path.append(nb)
            
            e_key = (min(current, nb), max(current, nb))
            e_idx = edge_to_idx[e_key]
            new_used_mask = used_mask | (1 << e_idx)
            
            result = dfs(nb, new_visited, path)
            if result is not None:
                return result
            
            # Backtrack
            path.pop()
        
        return None
    
    visited_mask = 1 << start
    path = [start]
    return dfs(start, visited_mask, path)

def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # 85 seconds to be safe
    n = P['n']
    edges = P['edges']
    adj, edge_to_idx = _build_adj(n, edges)
    
    all_vertices = list(range(n))
    max_possible_cycles = 3 # Theoretical max for 21 vertices, 70 edges
    
    best_cycles = []
    
    # Try multiple random restarts to ensure we find the maximum set
    for attempt in range(20):
        if time.time() > total_deadline:
            break
            
        used_mask = 0
        cycles = []
        
        for i in range(max_possible_cycles):
            if time.time() > total_deadline:
                break
                
            # Check connectivity before expensive search
            # We need to check connectivity on the graph with used edges removed
            # Re-implementing connectivity check with bitmask is complex, 
            # so we stick to a simple check or skip if it's slow.
            # Given the speedup, we can afford a simple check.
            # However, _is_connected expects used_edges as a set.
            # Let's just run the search; if it fails, it fails.
            # Or, we can convert used_mask to a set for the connectivity check?
            # That's expensive. Let's assume the graph remains connected or the search will handle it.
            # Actually, if the graph becomes disconnected, _find_ham_cycle will return None quickly.
            
            # Use a unique RNG for each attempt to vary the search
            rng = random.Random(attempt * 1000 + i)
            
            cycle = _find_ham_cycle(all_vertices, adj, edge_to_idx, used_mask, total_deadline, rng)
            
            if cycle is None:
                break
            
            # Add edges of this cycle to used_mask
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e_key = (min(u, v), max(u, v))
                if e_key in edge_to_idx:
                    e_idx = edge_to_idx[e_key]
                    used_mask |= (1 << e_idx)
            
            cycles.append(cycle)
            
        # Update best if we found more
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == max_possible_cycles:
                break # Found the theoretical max, stop searching

    # Convert to proper types
    result = [list(c) for c in best_cycles]
    
    if not result:
        result = [list(range(n))]

    return result
# EVOLVE-BLOCK-END
