"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, used_edges, n, deadline, start=0):
    """Find a Hamiltonian cycle using DFS with pruning."""
    # We want a cycle that visits all n vertices exactly once
    # Start from 'start'
    
    def dfs(path, current, visited):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[current] and (min(current, start), max(current, start)) not in used_edges:
                return path + [start]
            return None
        
        # Try neighbors
        neighbors = list(adj[current])
        random.shuffle(neighbors)
        for nxt in neighbors:
            if not visited[nxt]:
                edge = (min(current, nxt), max(current, nxt))
                if edge not in used_edges:
                    visited.add(nxt)
                    path.append(nxt)
                    result = dfs(path, nxt, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nxt)
        return None
    
    visited = {start}
    path = [start]
    result = dfs(path, start, visited)
    return result

def _extract_edges_from_cycle(cycle):
    """Extract the set of edges from a cycle (list of vertices)."""
    edges = set()
    n = len(cycle)
    for i in range(n - 1):
        u, v = cycle[i], cycle[i + 1]
        edges.add((min(u, v), max(u, v)))
    # Closing edge
    u, v = cycle[-1], cycle[0]
    edges.add((min(u, v), max(u, v)))
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    adj = _build_adjacency()
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add((min(u, v), max(u, v)))
    
    used_edges = set()
    cycles = []
    
    # Try to find edge-disjoint Hamiltonian cycles
    # The graph is 6-regular, so theoretically max 3 edge-disjoint Hamiltonian cycles
    
    random.seed(42)  # For reproducibility
    
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        if len(cycles) >= 3:
            break
        
        # Try different starting vertices and random shuffling
        found = False
        for start in range(n):
            if time.time() > deadline:
                break
            
            # Shuffle neighbors for diversity
            saved_adj = {k: list(v) for k, v in adj.items()}
            for k in saved_adj:
                random.shuffle(saved_adj[k])
            
            # Use a modified DFS that respects used_edges
            def dfs(path, current, visited):
                if time.time() > deadline:
                    return None
                if len(path) == n:
                    if start in saved_adj[current]:
                        edge = (min(current, start), max(current, start))
                        if edge not in used_edges:
                            return path + [start]
                    return None
                
                for nxt in saved_adj[current]:
                    if not visited[nxt]:
                        edge = (min(current, nxt), max(current, nxt))
                        if edge not in used_edges:
                            visited.add(nxt)
                            path.append(nxt)
                            result = dfs(path, nxt, visited)
                            if result is not None:
                                return result
                            path.pop()
                            visited.remove(nxt)
                return None
            
            visited = {start}
            path = [start]
            result = dfs(path, start, visited)
            
            if result is not None:
                # Verify the cycle
                cycle_edges = _extract_edges_from_cycle(result)
                if len(cycle_edges) == n and cycle_edges.isdisjoint(used_edges):
                    # Check all edges exist in the graph
                    if cycle_edges.issubset(all_edges):
                        cycles.append(result)
                        used_edges.update(cycle_edges)
                        found = True
                        break
        
        if not found:
            # Try a few more times with different random seeds
            for retry in range(5):
                if time.time() > deadline:
                    break
                random.seed(attempt * 1000 + retry)
                
                for start in range(n):
                    if time.time() > deadline:
                        break
                    
                    saved_adj = {k: list(v) for k, v in adj.items()}
                    for k in saved_adj:
                        random.shuffle(saved_adj[k])
                    
                    def dfs2(path, current, visited):
                        if time.time() > deadline:
                            return None
                        if len(path) == n:
                            if start in saved_adj[current]:
                                edge = (min(current, start), max(current, start))
                                if edge not in used_edges:
                                    return path + [start]
                            return None
                        
                        for nxt in saved_adj[current]:
                            if not visited[nxt]:
                                edge = (min(current, nxt), max(current, nxt))
                                if edge not in used_edges:
                                    visited.add(nxt)
                                    path.append(nxt)
                                    result = dfs2(path, nxt, visited)
                                    if result is not None:
                                        return result
                                    path.pop()
                                    visited.remove(nxt)
                        return None
                    
                    visited = {start}
                    path = [start]
                    result = dfs2(path, start, visited)
                    
                    if result is not None:
                        cycle_edges = _extract_edges_from_cycle(result)
                        if len(cycle_edges) == n and cycle_edges.isdisjoint(used_edges):
                            if cycle_edges.issubset(all_edges):
                                cycles.append(result)
                                used_edges.update(cycle_edges)
                                found = True
                                break
                if found:
                    break
        
        if len(cycles) >= 3:
            break
    
    # If we couldn't find any, return at least one valid cycle (the trivial one from seed)
    if not cycles:
        # Fallback: return a single Hamiltonian cycle if we can find one
        random.seed(0)
        for start in range(n):
            saved_adj = {k: list(v) for k, v in adj.items()}
            for k in saved_adj:
                random.shuffle(saved_adj[k])
            
            def dfs3(path, current, visited):
                if time.time() > deadline:
                    return None
                if len(path) == n:
                    if start in saved_adj[current]:
                        edge = (min(current, start), max(current, start))
                        if edge not in used_edges:
                            return path + [start]
                    return None
                
                for nxt in saved_adj[current]:
                    if not visited[nxt]:
                        edge = (min(current, nxt), max(current, nxt))
                        if edge not in used_edges:
                            visited.add(nxt)
                            path.append(nxt)
                            result = dfs3(path, nxt, visited)
                            if result is not None:
                                return result
                            path.pop()
                            visited.remove(nxt)
                return None
            
            visited = {start}
            path = [start]
            result = dfs3(path, start, visited)
            if result is not None:
                cycles.append(result)
                break
    
    # If still nothing, return the original seed answer
    if not cycles:
        cycles = [list(range(n))]
    
    # Convert to proper format: each cycle is a list of distinct vertices
    # The closing edge is implicit, so we should NOT include the start vertex at the end
    # Actually, the problem says "list of distinct vertices with the closing edge implicit"
    # So a cycle [0, 1, 2, ..., 20] means edges (0,1), (1,2), ..., (20,0)
    # This means we should have n vertices in the list, not n+1
    
    final_cycles = []
    for cycle in cycles:
        # If the cycle has n+1 vertices (with the start repeated at end), trim it
        if len(cycle) == n + 1 and cycle[0] == cycle[-1]:
            final_cycles.append(cycle[:-1])
        elif len(cycle) == n:
            final_cycles.append(cycle)
        else:
            # Should not happen
            final_cycles.append(cycle)
    
    # Ensure types are correct: integers must be Python int, sequences must be list
    result = []
    for cycle in final_cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
