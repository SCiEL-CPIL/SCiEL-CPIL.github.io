"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_bitmask(adj, n, start):
    """BFS using bitmasks for adjacency. Returns list of distances."""
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    visited = 1 << start
    
    while q:
        u = q.popleft()
        # Get unvisited neighbors
        neighbors = adj[u] & ~visited
        while neighbors:
            # Extract lowest set bit
            v_bit = neighbors & -neighbors
            v = v_bit.bit_length() - 1
            neighbors ^= v_bit
            dist[v] = dist[u] + 1
            visited |= v_bit
            q.append(v)
            
    return dist

def _calc_diameter_and_violations(adj, n):
    """
    Returns (diameter, num_violations) where violations are pairs with dist > 4.
    If graph is disconnected, diameter is -1 and violations is n*n.
    """
    max_d = 0
    violations = 0
    for start in range(n):
        dist = _bfs_bitmask(adj, n, start)
        if -1 in dist:
            return -1, n * n
        for d in dist:
            if d > max_d:
                max_d = d
            if d > 4:
                violations += 1
    return max_d, violations

def _is_triangle_free_add(adj, n, u, v):
    """Check if adding (u,v) creates a triangle."""
    # If u and v share a common neighbor, adding (u,v) creates a triangle
    common = adj[u] & adj[v]
    return common == 0

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Initialize adjacency bitmasks
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    # Precompute all possible edges not in base
    possible_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Check if edge exists in base
            if not (adj[i] & (1 << j)):
                possible_edges.append((i, j))
                
    best_solution = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Try multiple strategies
    strategies = [
        "max_violation_reduction",
        "max_diameter_reduction",
        "random"
    ]
    
    for strategy in strategies:
        # Reset adj to base
        curr_adj = adj[:]
        added_edges = []
        
        # Greedy Loop
        while time.time() < deadline:
            diam, violations = _calc_diameter_and_violations(curr_adj, n)
            if diam <= 4:
                break
                
            if violations == 0:
                break # Should not happen if diam > 4 and connected? 
                # Actually if diam > 4, there are violations.
                
            best_edge = None
            best_score = float('inf')
            
            # Shuffle for random strategy to break ties differently
            candidates = possible_edges[:]
            if strategy == "random":
                random.shuffle(candidates)
            else:
                # Sort by some heuristic? 
                # For now, just iterate. We can optimize later.
                pass
            
            for u, v in candidates:
                # Check if edge already added
                if curr_adj[u] & (1 << v):
                    continue
                
                # Triangle check
                if not _is_triangle_free_add(curr_adj, n, u, v):
                    continue
                
                # Simulate adding edge
                curr_adj[u] |= 1 << v
                curr_adj[v] |= 1 << u
                
                new_diam, new_violations = _calc_diameter_and_violations(curr_adj, n)
                
                # Remove edge
                curr_adj[u] &= ~(1 << v)
                curr_adj[v] &= ~(1 << u)
                
                if new_diam == -1:
                    continue
                    
                if strategy == "max_violation_reduction":
                    score = new_violations
                elif strategy == "max_diameter_reduction":
                    score = new_diam
                else:
                    score = new_violations + new_diam * 10 # Mixed
                
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
                    
            if best_edge is None:
                break
                
            u, v = best_edge
            curr_adj[u] |= 1 << v
            curr_adj[v] |= 1 << u
            added_edges.append([u, v])
            
        # Pruning Phase
        # Try to remove edges
        changed = True
        while changed:
            changed = False
            for i in range(len(added_edges) - 1, -1, -1):
                u, v = added_edges[i]
                # Remove
                curr_adj[u] &= ~(1 << v)
                curr_adj[v] &= ~(1 << u)
                
                diam, violations = _calc_diameter_and_violations(curr_adj, n)
                
                if diam <= 4 and violations == 0: # violations 0 implies diam<=4 if connected
                    # Keep removed
                    added_edges.pop(i)
                    changed = True
                else:
                    # Re-add
                    curr_adj[u] |= 1 << v
                    curr_adj[v] |= 1 << u
                    
        # Check final validity
        if added_edges:
            # Rebuild adj to be sure
            final_adj = adj[:]
            for u, v in added_edges:
                final_adj[u] |= 1 << v
                final_adj[v] |= 1 << u
            
            diam, violations = _calc_diameter_and_violations(final_adj, n)
            if diam <= 4:
                if len(added_edges) < best_size:
                    best_size = len(added_edges)
                    best_solution = added_edges[:]
        
        # If we found a solution with size 0, 1, etc, we can stop early?
        # Theoretical min is likely > 0.
        if best_size <= 4:
             break

    if best_solution is None:
        # Fallback: return a known valid solution if we didn't find one
        # The seed program found one. We should try to replicate or find a simple one.
        # For now, return empty if fail, but that's invalid.
        # Let's force a simple greedy without pruning if this happens.
        curr_adj = adj[:]
        added_edges = []
        while time.time() < deadline:
            diam, _ = _calc_diameter_and_violations(curr_adj, n)
            if diam <= 4:
                break
            best_edge = None
            best_score = float('inf')
            for u, v in possible_edges:
                if curr_adj[u] & (1 << v):
                    continue
                if not _is_triangle_free_add(curr_adj, n, u, v):
                    continue
                curr_adj[u] |= 1 << v
                curr_adj[v] |= 1 << u
                new_diam, _ = _calc_diameter_and_violations(curr_adj, n)
                curr_adj[u] &= ~(1 << v)
                curr_adj[v] &= ~(1 << u)
                
                if new_diam == -1: continue
                if new_diam < best_score:
                    best_score = new_diam
                    best_edge = (u, v)
            if best_edge is None: break
            u, v = best_edge
            curr_adj[u] |= 1 << v
            curr_adj[v] |= 1 << u
            added_edges.append([u, v])
            
        if added_edges:
            final_adj = adj[:]
            for u, v in added_edges:
                final_adj[u] |= 1 << v
                final_adj[v] |= 1 << u
            diam, _ = _calc_diameter_and_violations(final_adj, n)
            if diam <= 4:
                best_solution = added_edges
                best_size = len(added_edges)

    if best_solution:
        # Convert to list of lists of ints
        return [[int(u), int(v)] for u, v in best_solution]
    else:
        # Should not happen, but return empty to avoid crash
        return []

# EVOLVE-BLOCK-END
