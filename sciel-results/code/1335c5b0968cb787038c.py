"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def get_canonical_edge(u, v):
        if u < v:
            return (u, v)
        else:
            return (v, u)
    
    def find_hamiltonian_cycle(adj, deadline, time_limit):
        """Find a Hamiltonian cycle in the given graph using DFS with backtracking."""
        n = len(adj)
        if n < 3:
            return None
        
        # Try to find a Hamiltonian cycle starting from vertex 0
        # Use symmetry breaking: fix vertex 0 as start, and fix the first neighbor to be the smallest
        start = 0
        first_neighbor = min(adj[start]) if adj[start] else None
        if first_neighbor is None:
            return None
        
        # We'll use iterative DFS with backtracking
        # State: current vertex, path so far, visited set
        # For efficiency, use recursion with a time check
        
        visited = [False] * n
        path = []
        
        def backtrack(current, remaining):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                # Check if we can close the cycle
                if start in adj[current]:
                    return list(path) + [start]
                return None
            
            path.append(current)
            visited[current] = True
            
            # Get neighbors in a good order
            # Prefer neighbors with fewer unvisited neighbors (Warnsdorff's rule)
            neighbors = list(adj[current])
            # Sort by number of unvisited neighbors
            def sort_key(v):
                unvisited_count = sum(1 for u in adj[v] if not visited[u] and u != start)
                return unvisited_count
            
            neighbors.sort(key=sort_key)
            
            for next_v in neighbors:
                if not visited[next_v] and next_v != start:
                    result = backtrack(next_v, remaining - 1)
                    if result is not None:
                        return result
                    if time.time() > deadline:
                        break
            
            path.pop()
            visited[current] = False
            return None
        
        # Try with fixed first neighbor for symmetry breaking
        result = backtrack(first_neighbor, n - 1)
        if result is not None:
            return result
        
        # Try other first neighbors
        for first_v in adj[start]:
            if first_v == first_neighbor:
                continue
            if time.time() > deadline:
                break
            # Reset
            visited = [False] * n
            path = []
            
            def backtrack2(current, remaining):
                if time.time() > deadline:
                    return None
                
                if remaining == 0:
                    if start in adj[current]:
                        return list(path) + [start]
                    return None
                
                path.append(current)
                visited[current] = True
                
                neighbors = list(adj[current])
                def sort_key2(v):
                    unvisited_count = sum(1 for u in adj[v] if not visited[u] and u != start)
                    return unvisited_count
                
                neighbors.sort(key=sort_key2)
                
                for next_v in neighbors:
                    if not visited[next_v] and next_v != start:
                        result = backtrack2(next_v, remaining - 1)
                        if result is not None:
                            return result
                        if time.time() > deadline:
                            break
                
                path.pop()
                visited[current] = False
                return None
            
            result = backtrack2(first_v, n - 1)
            if result is not None:
                return result
        
        return None
    
    # Main algorithm: iteratively find edge-disjoint Hamiltonian cycles
    # Start with the full graph, find a cycle, remove its edges, repeat
    remaining_edges = set(edge_set)
    cycles = []
    
    deadline = time.time() + 55  # Leave 5s safety margin for 60s limit
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        if not remaining_edges:
            break
        
        # Build adjacency from remaining edges
        adj_remaining = [set() for _ in range(n)]
        for u, v in remaining_edges:
            adj_remaining[u].add(v)
            adj_remaining[v].add(u)
        
        # Check that all vertices have degree >= 2 in remaining graph
        min_degree = min(len(adj_remaining[i]) for i in range(n))
        if min_degree < 2:
            break
        
        # Try to find a Hamiltonian cycle
        cycle = find_hamiltonian_cycle(adj_remaining, deadline, 5.0)
        
        if cycle is None:
            # Try randomized restarts with different orderings
            found = False
            for restart in range(50):
                if time.time() > deadline:
                    break
                
                # Shuffle adjacency lists
                adj_shuffled = [set(adj_remaining[i]) for i in range(n)]
                for i in range(n):
                    adj_list = list(adj_shuffled[i])
                    random.shuffle(adj_list)
                    adj_shuffled[i] = set(adj_list)
                
                cycle = find_hamiltonian_cycle(adj_shuffled, deadline, 0.5)
                if cycle is not None:
                    found = True
                    break
            
            if not found:
                break
        
        # Remove the cycle's edges from remaining_edges
        cycle_edges = set()
        for i in range(len(cycle) - 1):
            e = get_canonical_edge(cycle[i], cycle[i+1])
            if e in remaining_edges:
                cycle_edges.add(e)
        
        if len(cycle_edges) != len(cycle):
            # Cycle uses an edge not in remaining, shouldn't happen
            break
        
        remaining_edges -= cycle_edges
        cycles.append(cycle)
    
    # Return at most 10 cycles, all as lists of ints
    result = []
    for c in cycles[:10]:
        # Ensure all elements are Python ints
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
