"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _get_arrival():
    """Build dict mapping (u,v) sorted tuple to one-based arrival index."""
    n = P['n']
    arrival = {}
    for i, (u, v) in enumerate(P['edge_order'], 1):
        key = (min(u, v), max(u, v))
        arrival[key] = i
    return arrival

def _find_hamilton_cycle_warnsdorff(adj, n, start, deadline, best_score_ref):
    """
    Warnsdorff-style heuristic DFS to find a Hamilton cycle.
    Uses degree-based ordering to prefer neighbors with fewer unvisited options.
    """
    visited = [False] * n
    visited[start] = True
    path = [start]
    best_cycle = [None]
    
    def dfs(u, depth):
        if time.time() > deadline:
            return False
        if depth == n:
            # Check if edge from last to start exists
            if start in adj[u]:
                best_cycle[0] = path[:]
                return True
            return False
        
        # Get unvisited neighbors, sorted by Warnsdorff heuristic
        unvisited_neighbors = [v for v in adj[u] if not visited[v]]
        if not unvisited_neighbors:
            return False
        
        # Sort by number of unvisited neighbors (Warnsdorff's rule)
        def warnsdorff_key(v):
            return sum(1 for w in adj[v] if not visited[w])
        
        unvisited_neighbors.sort(key=warnsdorff_key)
        
        for v in unvisited_neighbors:
            visited[v] = True
            path.append(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            visited[v] = False
        
        return False
    
    if dfs(start, 1):
        return best_cycle[0]
    return None

def _find_hamilton_cycle_multi_start(adj, n, deadline, best_score_ref):
    """
    Try multiple starting vertices to find a Hamilton cycle.
    """
    for start in range(n):
        if time.time() > deadline:
            break
        result = _find_hamilton_cycle_warnsdorff(adj, n, start, deadline, best_score_ref)
        if result is not None:
            return result
    return None

def _build_cycle_from_cycle_cover(cycle_cover, n, arrival, deadline):
    """
    Given a cycle cover (dict mapping each vertex to its successor),
    try to merge disjoint cycles into a single Hamiltonian cycle using 2-opt swaps.
    """
    # Identify cycles in the cycle cover
    cycles = []
    visited = [False] * n
    for v in range(n):
        if not visited[v]:
            cycle = [v]
            visited[v] = True
            curr = cycle_cover[v]
            while curr != v:
                cycle.append(curr)
                visited[curr] = True
                curr = cycle_cover[curr]
            if len(cycle) > 1:
                cycles.append(cycle)
    
    # If we already have one cycle, we're done
    if len(cycles) == 1:
        return cycles[0]
    
    # Try to merge cycles using 2-opt swaps
    # For each pair of cycles, try all possible edge swaps
    max_iterations = 1000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if len(cycles) == 1:
            break
        
        merged = False
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                if len(cycles) <= 1:
                    break
                ci = cycles[i]
                cj = cycles[j]
                
                # Try all pairs of edges to swap
                for ai in range(len(ci)):
                    for aj in range(len(cj)):
                        u1, v1 = ci[ai], ci[(ai + 1) % len(ci)]
                        u2, v2 = cj[aj], cj[(aj + 1) % len(cj)]
                        
                        # Check if cross edges exist in the original graph
                        # (We need to check if edges (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2) are available)
                        # For simplicity, we'll just try to merge by reconnecting
                        # This is a simplified approach; a full 2-opt would check edge availability
                        
                        # Create merged cycle
                        new_cycle = ci[:ai+1] + cj[aj+1:] + cj[:aj] + ci[ai+1:]
                        # Verify this is a valid cycle (all consecutive edges exist)
                        valid = True
                        for k in range(len(new_cycle)):
                            a = new_cycle[k]
                            b = new_cycle[(k + 1) % len(new_cycle)]
                            # Check if edge (a,b) exists - we'd need the full graph
                            # For now, assume it does if we're working with the cycle cover structure
                        if valid:
                            cycles = [c for idx, c in enumerate(cycles) if idx not in (i, j)]
                            cycles.append(new_cycle)
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
    
    if len(cycles) == 1:
        return cycles[0]
    return None

def construct():
    n = P['n']
    arrival = _get_arrival()
    deadline = time.time() + 85  # Leave 5s safety margin for 90s limit
    
    best_cycle = list(range(n))  # fallback: identity permutation
    best_score = max(arrival[(min(u,v), max(u,v))] for u, v in 
                     [(best_cycle[i], best_cycle[(i+1) % n]) for i in range(n)])
    
    # Try increasing prefix sizes with binary search for efficiency
    # Minimum possible is n (24 edges for a cycle on 24 vertices)
    lo, hi = n, len(P['edge_order'])
    
    # First, do a linear scan to find the first prefix where a cycle exists
    found_prefix = None
    for prefix_size in range(n, min(n + 50, hi + 1)):
        if time.time() > deadline:
            break
        
        # Build adjacency list using only edges arriving in first prefix_size positions
        adj = [set() for _ in range(n)]
        for i, (u, v) in enumerate(P['edge_order'][:prefix_size], 1):
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if all vertices have degree >= 2 (necessary for Hamilton cycle)
        if any(len(adj[v]) < 2 for v in range(n)):
            continue
        
        # Try multi-start Warnsdorff DFS
        result = _find_hamilton_cycle_multi_start(adj, n, deadline, [best_score])
        if result is not None:
            # Verify and compute score
            score = 0
            valid = True
            for i in range(n):
                u = result[i]
                v = result[(i + 1) % n]
                key = (min(u, v), max(u, v))
                if key not in arrival:
                    valid = False
                    break
                score = max(score, arrival[key])
            if valid and score < best_score:
                best_score = score
                best_cycle = result
                found_prefix = prefix_size
                # Since we're scanning linearly, this is the first valid prefix
                # We can break because larger prefixes won't give better scores
                break
    
    # If we didn't find a cycle in the initial scan, try a few more prefixes
    if found_prefix is None:
        for prefix_size in range(n + 50, min(n + 100, hi + 1)):
            if time.time() > deadline:
                break
            
            adj = [set() for _ in range(n)]
            for i, (u, v) in enumerate(P['edge_order'][:prefix_size], 1):
                adj[u].add(v)
                adj[v].add(u)
            
            if any(len(adj[v]) < 2 for v in range(n)):
                continue
            
            result = _find_hamilton_cycle_multi_start(adj, n, deadline, [best_score])
            if result is not None:
                score = 0
                valid = True
                for i in range(n):
                    u = result[i]
                    v = result[(i + 1) % n]
                    key = (min(u, v), max(u, v))
                    if key not in arrival:
                        valid = False
                        break
                    score = max(score, arrival[key])
                if valid and score < best_score:
                    best_score = score
                    best_cycle = result
                    break
    
    # Final verification
    if best_cycle is not None and len(best_cycle) == n:
        # Verify it's a valid permutation
        if sorted(best_cycle) != list(range(n)):
            best_cycle = list(range(n))
            best_score = max(arrival[(min(u,v), max(u,v))] for u, v in 
                             [(best_cycle[i], best_cycle[(i+1) % n]) for i in range(n)])
    
    return best_cycle
# EVOLVE-BLOCK-END
