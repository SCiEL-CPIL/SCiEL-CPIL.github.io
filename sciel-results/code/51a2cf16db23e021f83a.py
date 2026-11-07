"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, start, deadline, max_nodes=2000000):
    """Find a Hamiltonian cycle starting from 'start' using Warnsdorff-guided DFS."""
    n = P['n']
    nodes_explored = [0]
    
    def dfs(path, visited):
        if time.time() > deadline:
            return None
        nodes_explored[0] += 1
        if nodes_explored[0] > max_nodes:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if path[0] in adj[path[-1]]:
                return path[:]
            return None
        
        current = path[-1]
        # Get unvisited neighbors
        candidates = [v for v in adj[current] if v not in visited]
        
        if not candidates:
            return None
        
        # Warnsdorff: prioritize vertices with fewer onward options
        def warnsdorff_score(v):
            return len([w for w in adj[v] if w not in visited and w != path[0]])
        
        candidates.sort(key=warnsdorff_score)
        
        for v in candidates:
            path.append(v)
            visited.add(v)
            result = dfs(path, visited)
            if result is not None:
                return result
            path.pop()
            visited.remove(v)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(path, visited)

def _find_ham_cycle_v2(adj, start, deadline, max_nodes=2000000):
    """Find a Hamiltonian cycle starting from 'start' using Warnsdorff-guided DFS with different tie-breaking."""
    n = P['n']
    nodes_explored = [0]
    
    def dfs(path, visited):
        if time.time() > deadline:
            return None
        nodes_explored[0] += 1
        if nodes_explored[0] > max_nodes:
            return None
        if len(path) == n:
            if path[0] in adj[path[-1]]:
                return path[:]
            return None
        
        current = path[-1]
        candidates = [v for v in adj[current] if v not in visited]
        
        if not candidates:
            return None
        
        # Warnsdorff with secondary sort by vertex index for diversity
        def warnsdorff_score(v):
            primary = len([w for w in adj[v] if w not in visited and w != path[0]])
            return (primary, v)
        
        candidates.sort(key=warnsdorff_score)
        
        for v in candidates:
            path.append(v)
            visited.add(v)
            result = dfs(path, visited)
            if result is not None:
                return result
            path.pop()
            visited.remove(v)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(path, visited)

def construct():
    deadline = time.time() + 80  # 80 second safety margin for 90s limit
    
    edges = list(P['edges'])
    n = P['n']
    
    # Build initial adjacency
    base_adj = _build_adj(edges)
    
    best_cycles = []
    
    # Multiple restarts with different strategies
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Build adjacency from remaining edges (fresh copy each restart)
        adj = [set(s) for s in base_adj]
        used_edges = set()
        cycles = []
        
        # Try to find up to 3 edge-disjoint Hamiltonian cycles
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Check if all vertices have degree >= 2
            valid = True
            for i in range(n):
                if len(adj[i]) < 2:
                    valid = False
                    break
            if not valid:
                break
            
            # Try multiple starting vertices and strategies
            found = False
            
            # Strategy 1: Try vertices sorted by degree (descending)
            starts_deg = sorted(range(n), key=lambda v: -len(adj[v]))
            
            # Strategy 2: Try all vertices in natural order
            starts_nat = list(range(n))
            
            # Alternate between strategies based on restart
            if restart % 2 == 0:
                starts_list = starts_deg
            else:
                starts_list = starts_nat
            
            # Use different finder based on restart
            finder = _find_ham_cycle if restart % 3 != 2 else _find_ham_cycle_v2
            
            for start in starts_list:
                if time.time() > deadline:
                    break
                cycle = finder(adj, start, deadline)
                if cycle is not None:
                    # Mark edges as used
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        key = (min(u, v), max(u, v))
                        used_edges.add(key)
                        adj[u].discard(v)
                        adj[v].discard(u)
                    cycles.append(cycle)
                    found = True
                    break
            
            if not found:
                break
        
        # Keep the best result so far
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:
                break
    
    # If we didn't find 3 cycles, try a more exhaustive search with the best partial
    if len(best_cycles) < 3 and time.time() < deadline:
        # Try to extend the best partial solution
        for restart in range(10):
            if time.time() > deadline:
                break
            
            adj = [set(s) for s in base_adj]
            # Reconstruct the partial solution's edge usage
            for cycle in best_cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            cycles = list(best_cycles)
            
            for attempt in range(3 - len(cycles)):
                if time.time() > deadline:
                    break
                
                valid = True
                for i in range(n):
                    if len(adj[i]) < 2:
                        valid = False
                        break
                if not valid:
                    break
                
                found = False
                for start in range(n):
                    if time.time() > deadline:
                        break
                    cycle = _find_ham_cycle(adj, start, deadline)
                    if cycle is not None:
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            adj[u].discard(v)
                            adj[v].discard(u)
                        cycles.append(cycle)
                        found = True
                        break
                
                if not found:
                    break
            
            if len(cycles) > len(best_cycles):
                best_cycles = cycles
                if len(best_cycles) >= 3:
                    break
    
    # Convert to proper types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    # Fallback: ensure at least one cycle
    if not result:
        # Try once more with a simple approach
        adj = _build_adj(edges)
        for start in range(n):
            cycle = _find_ham_cycle(adj, start, time.time() + 5)
            if cycle is not None:
                result.append([int(v) for v in cycle])
                break
    
    return result
# EVOLVE-BLOCK-END
