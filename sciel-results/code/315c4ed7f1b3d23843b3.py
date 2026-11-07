"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

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

def _find_ham_cycle(vertices, adj, used_edges, deadline, rng, max_depth=5000):
    """Find a Hamiltonian cycle on the given vertex set using DFS with backtracking."""
    vertices = list(vertices)
    n = len(vertices)
    if n < 3:
        return None
    
    vset = set(vertices)
    start = vertices[0]
    
    # Precompute adjacency as frozensets for faster lookup
    adj_frozen = {u: frozenset(v) for u, v in adj.items()}
    
    # Precompute all edges in the graph for quick reference
    all_graph_edges = set()
    for u in adj_frozen:
        for v in adj_frozen[u]:
            all_graph_edges.add((min(u, v), max(u, v)))
    
    # Count available edges (not in used_edges)
    available_edges = all_graph_edges - used_edges
    
    # Check if enough edges remain for a Hamiltonian cycle
    if len(available_edges) < n:
        return None
    
    def _connected_remaining(visited_set, current):
        """Check if unvisited vertices + current form a connected subgraph with available edges."""
        unvisited = vset - visited_set
        if not unvisited:
            return True
        # BFS from current through available edges to see if all unvisited are reachable
        # (plus we need to be able to return to start at the end)
        visited_bfs = set()
        stack = [current]
        visited_bfs.add(current)
        while stack:
            u = stack.pop()
            for v in adj_frozen[u]:
                if v in vset and v not in visited_bfs:
                    e = (min(u, v), max(u, v))
                    if e in available_edges:
                        visited_bfs.add(v)
                        stack.append(v)
        # All unvisited must be reachable
        return unvisited.issubset(visited_bfs)
    
    def dfs(current, visited, path):
        if time.time() > deadline:
            return None
            
        if len(visited) == n:
            # Check if we can close the cycle
            if start in adj_frozen[current]:
                e = (min(current, start), max(current, start))
                if e in available_edges:
                    return path
            return None
        
        # Get unvisited neighbors with available edges
        unvisited_neighbors = []
        for nb in adj_frozen[current]:
            if nb in vset and nb not in visited:
                e = (min(current, nb), max(current, nb))
                if e in available_edges:
                    unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Sort by degree in remaining graph (Warnsdorff's heuristic)
        def heuristic_key(x):
            deg = 0
            for y in adj_frozen[x]:
                if y in vset and y not in visited:
                    e2 = (min(x, y), max(x, y))
                    if e2 in available_edges:
                        deg += 1
            return (deg, rng.random())
            
        unvisited_neighbors.sort(key=heuristic_key)
        
        for nb in unvisited_neighbors:
            visited.add(nb)
            path.append(nb)
            e = (min(current, nb), max(current, nb))
            available_edges.discard(e)
            
            result = dfs(nb, visited, path)
            if result is not None:
                return result
            
            # Backtrack
            path.pop()
            visited.discard(nb)
            available_edges.add(e)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(start, visited, path)

def construct():
    start_time = time.time()
    total_deadline = start_time + 80  # 80 seconds to be safe (90s limit, leave margin)
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    all_vertices = list(range(n))
    max_possible_cycles = 3 # Theoretical max for 21 vertices, 70 edges
    
    best_cycles = []
    
    # Try multiple random restarts to ensure we find the maximum set
    for attempt in range(20):
        if time.time() > total_deadline:
            break
            
        used_edges = set()
        cycles = []
        
        for i in range(max_possible_cycles):
            if time.time() > total_deadline:
                break
                
            # Check connectivity before expensive search
            if not _is_connected(all_vertices, adj, used_edges):
                break
                
            # Use a unique RNG for each attempt to vary the search
            rng = random.Random(attempt * 1000 + i)
            
            # Allocate time: give more budget to later cycles since they're harder
            # First cycle: generous, second: moderate, third: tight
            if i == 0:
                cycle_deadline = total_deadline
            elif i == 1:
                cycle_deadline = total_deadline
            else:
                cycle_deadline = total_deadline
            
            cycle = _find_ham_cycle(all_vertices, adj, used_edges, cycle_deadline, rng)
            
            if cycle is None:
                break
            
            # Add edges of this cycle to used_edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            
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
