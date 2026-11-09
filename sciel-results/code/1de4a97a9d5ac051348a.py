import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe

    # Seed: Star-Path
    # Edges: (0, i) for i in 1..39, (i, i+1) for i in 1..38
    edges_set = set()
    for i in range(1, N):
        edges_set.add((0, i))
    for i in range(1, N - 1):
        edges_set.add((i, i + 1))
    
    # Convert to list of tuples for easier manipulation, but keep set for lookup
    current_edges = list(edges_set)
    
    # Helper: Check if graph with given edges is pancyclic
    def is_pancyclic(edge_list):
        # Build adjacency
        adj = {i: set() for i in range(N)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            
        # Quick checks
        for i in range(N):
            if len(adj[i]) < 2:
                return False
        
        # Check for cycles of each length k
        for k in range(3, N + 1):
            if not has_cycle_of_length(adj, k):
                return False
        return True

    def has_cycle_of_length(adj, k):
        # DFS to find cycle of length k
        # Since N=40, and k can be up to 40, this can be expensive if it fails.
        # But we start from a valid graph, so it should succeed quickly.
        
        # Optimization: If k == N, check for Hamiltonian cycle.
        # For small k, check specific structures.
        
        # General DFS
        # To avoid permutations, fix the start node to 0? 
        # No, cycle might not include 0.
        # But in our seed, all cycles include 0.
        # If we remove edges, cycles might move.
        
        # Let's try DFS from every node? Too slow.
        # Let's try DFS from 0 first. If found, return True.
        # If not, try from 1, etc.
        
        # Actually, if the graph is connected and dense enough, cycles are likely everywhere.
        
        # Simple DFS with pruning
        def dfs(node, start, path, visited):
            if len(path) == k:
                if start in adj[node]:
                    return True
                return False
            if len(path) == k - 1:
                # Next node must be start
                return start in adj[node]
            
            for neighbor in adj[node]:
                if neighbor == start:
                    # Only allowed at the end
                    if len(path) == k - 1:
                        return True
                    else:
                        continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor, start, path, visited):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False

        # Try starting from each vertex
        # To speed up, we can just try a few starts.
        # If the graph is pancyclic, it's likely "well-connected".
        for start in range(N):
            if dfs(start, start, [start], {start}):
                return True
        return False

    # Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle edges to find different removals?
        # Or just iterate.
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            edge_to_remove = current_edges[i]
            candidate = [e for j, e in enumerate(current_edges) if j != i]
            if is_pancyclic(candidate):
                current_edges = candidate
                improved = True
                break # Restart loop with new shorter list

    # Return the best found
    # Ensure edges are tuples (u, v) with u < v for consistency?
    # The task says "undirected edge pairs".
    # Let's normalize them.
    final_edges = []
    for u, v in current_edges:
        if u > v:
            u, v = v, u
        final_edges.append((u, v))
    
    # Generate cycles for the final graph
    # We need to find ONE cycle of each length 3..40 in the final graph.
    cycles = []
    adj = {i: set() for i in range(N)}
    for u, v in final_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for k in range(3, N + 1):
        cycle_found = None
        # Reuse the DFS logic to FIND the cycle
        def find_cycle(node, start, path, visited):
            if len(path) == k:
                if start in adj[node]:
                    return path[:] + [start]
                return None
            if len(path) == k - 1:
                if start in adj[node]:
                    return path[:] + [start]
                return None
            
            for neighbor in adj[node]:
                if neighbor == start:
                    if len(path) == k - 1:
                        return path[:] + [start]
                    else:
                        continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    res = find_cycle(neighbor, start, path, visited)
                    if res:
                        return res
                    path.pop()
                    visited.remove(neighbor)
            return None

        for start in range(N):
            cycle_found = find_cycle(start, start, [start], {start})
            if cycle_found:
                break
        if not cycle_found:
            # Fallback: Should not happen if is_pancyclic passed
            # Use seed cycles? No, they might not be valid in new graph.
            # But is_pancyclic ensures existence.
            raise RuntimeError("Cycle not found")
        cycles.append(cycle_found)

    return {'edges': final_edges, 'cycles': cycles}

# EVOLVE-BLOCK-END
