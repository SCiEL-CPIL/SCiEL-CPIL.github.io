"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            es.add((min(u,v), max(u,v)))
        return es
    
    def validate_cycle(cycle):
        if len(cycle) < 3 or len(set(cycle)) != len(cycle):
            return False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            if (min(u,v), max(u,v)) not in edge_set:
                return False
        return True
    
    # The graph has 70 edges. Max edge-disjoint Hamiltonian cycles = floor(70/21) = 3.
    # Use iterative deepening DFS with strong pruning to find 3 disjoint H-cycles.
    
    # Precompute: for each vertex, sorted list of neighbors
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    def find_cycle_with_pruning(used_edges, start=0, max_depth=None):
        """Find Hamiltonian cycle avoiding used_edges using DFS with pruning"""
        if max_depth is None:
            max_depth = 500000
        
        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in adj_list[u]:
                e = (min(u,v), max(u,v))
                if e not in used_edges:
                    rem_adj[u].add(v)
        
        # Quick check: all vertices must have degree >= 2
        for u in range(n):
            if len(rem_adj[u]) < 2:
                return None
        
        # Convert to list for faster iteration
        rem_adj_list = [sorted(rem_adj[u]) for u in range(n)]
        
        path = [start]
        visited = [False] * n
        visited[start] = True
        path_edges = set()
        nodes_explored = 0
        
        def dfs(idx, current):
            nonlocal nodes_explored
            nodes_explored += 1
            if nodes_explored > max_depth:
                return None
            
            if idx == n - 1:
                # Check if we can close the cycle
                last = current
                e = (min(last, start), max(last, start))
                if e not in path_edges and e not in used_edges and e in edge_set:
                    return list(path)
                return None
            
            current = path[idx]
            # Try neighbors sorted by degree (MRV-like)
            neighbors = rem_adj_list[current]
            # Sort by remaining degree for better pruning
            neighbors_sorted = sorted(neighbors, key=lambda x: len(rem_adj_list[x]))
            
            for next_v in neighbors_sorted:
                if visited[next_v]:
                    continue
                e = (min(current, next_v), max(current, next_v))
                if e in path_edges or e in used_edges:
                    continue
                
                # Pruning: after adding this edge, check if any unvisited vertex
                # has lost all but one possible connection
                # (simplified: check degree in remaining graph)
                visited[next_v] = True
                path.append(next_v)
                path_edges.add(e)
                
                # Prune: check unvisited vertices have at least 2 available neighbors
                # in the subgraph induced by unvisited + {current, next_v}
                prune_needed = False
                for uv in range(n):
                    if not visited[uv]:
                        avail = 0
                        for nb in rem_adj_list[uv]:
                            # nb must be either unvisited or be next_v (the last in path)
                            if not visited[nb] or nb == next_v:
                                e2 = (min(uv, nb), max(uv, nb))
                                if e2 not in path_edges and e2 not in used_edges:
                                    avail += 1
                        if avail < 2:
                            prune_needed = True
                            break
                
                if not prune_needed:
                    result = dfs(idx + 1, next_v)
                    if result is not None:
                        return result
                
                # Backtrack
                path.pop()
                visited[next_v] = False
                path_edges.remove(e)
            
            return None
        
        return dfs(0, start)
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    found_cycles = []
    used_edges_global = set()
    
    for i in range(3):
        if time.time() > deadline - 15:
            break
        
        # Try multiple start vertices and strategies
        cycle = None
        for start_v in range(0, n, 3):  # Try several starting points
            if time.time() > deadline - 15:
                break
            cycle = find_cycle_with_pruning(used_edges_global, start=start_v, max_depth=300000)
            if cycle is not None:
                break
        
        if cycle is not None:
            cycle_edges = get_cycle_edges(cycle)
            if cycle_edges & used_edges_global:
                continue
            found_cycles.append(cycle)
            used_edges_global |= cycle_edges
        else:
            break
    
    # If we couldn't find 3, try a different strategy
    if len(found_cycles) < 3:
        # Reset and try with randomized neighbor ordering
        import random
        random.seed(42)
        found_cycles = []
        used_edges_global = set()
        
        for i in range(3):
            if time.time() > deadline - 10:
                break
            
            cycle = None
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                # Shuffle neighbor orderings
                shuffled_adj = [list(rem) for rem in adj_list]
                for lst in shuffled_adj:
                    random.shuffle(lst)
                
                # Use the shuffled adjacency
                cycle = find_cycle_with_pruning(used_edges_global, start=0, max_depth=200000)
                if cycle is not None:
                    break
            
            if cycle is not None:
                cycle_edges = get_cycle_edges(cycle)
                if cycle_edges & used_edges_global:
                    continue
                found_cycles.append(cycle)
                used_edges_global |= cycle_edges
            else:
                break
    
    # Final fallback: return at least one valid cycle
    if not found_cycles:
        # Try the simplest possible: consecutive vertices
        c = list(range(21))
        if validate_cycle(c):
            found_cycles = [c]
        else:
            # Find any triangle
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        for k in adj_list[j]:
                            if k > j and k in adj[i]:
                                found_cycles = [[i, j, k]]
                                break
                        if found_cycles:
                            break
                if found_cycles:
                    break
            if not found_cycles:
                found_cycles = [[0, 1, 2]]
    
    # Ensure all elements are Python ints and cycles are lists
    result = []
    for cycle in found_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
