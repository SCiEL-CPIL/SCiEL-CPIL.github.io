"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Initialize adjacency bitsets for base graph
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    # Generate candidate edges
    # Exclude base edges and self-loops
    # Also exclude edges that form a triangle with base edges?
    # As analyzed, (u,v) with v >= u+2 does not form a triangle with base edges alone.
    # But it might form a triangle with *other* selected edges.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            # Skip base edges
            if (1 << v) & adj[u]:
                continue
            # Check if (u,v) forms a triangle with base edges?
            # Triangle requires common neighbor w in base.
            # If w is in base-neighborhood of u and v, then (u,w) and (w,v) are base edges.
            # In a path, u and v share a base neighbor only if |u-v|=2.
            # e.g., u=0, v=2. w=1. (0,1) and (1,2) are base.
            # So if v == u + 2, adding (u,v) creates a triangle with base.
            if v == u + 2:
                continue
            candidates.append((u, v))
            
    # Sort candidates by heuristic: longer edges first (likely more helpful for diameter)
    # Distance in path is v - u. Larger v-u means it covers more path length.
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # Pre-compute bitmask for each candidate
    # c_mask is not needed, just u, v
    
    def check_diameter(adj_state):
        # BFS from all nodes to find max distance
        # n=24, so 24 BFS runs is fine
        max_diam = 0
        for start in range(n):
            visited = [False] * n
            dist = [-1] * n
            queue = [start]
            visited[start] = True
            dist[start] = 0
            q_idx = 0
            while q_idx < len(queue):
                u = queue[q_idx]
                q_idx += 1
                # Iterate neighbors in adj_state[u]
                neighbors_mask = adj_state[u]
                while neighbors_mask:
                    # Get lowest set bit
                    lsb = neighbors_mask & -neighbors_mask
                    v = lsb.bit_length() - 1
                    if not visited[v]:
                        visited[v] = True
                        dist[v] = dist[u] + 1
                        if dist[v] > max_diam:
                            max_diam = dist[v]
                        queue.append(v)
                    neighbors_mask ^= lsb
            # Check if graph is connected (all visited)
            if not all(visited):
                return n + 1 # Disconnected, treat as bad
        return max_diam

    def search(idx, selected, adj_state, k_remaining):
        # idx: index in candidates list
        # selected: list of selected edges
        # adj_state: current adjacency bitsets
        # k_remaining: how many more edges we can pick
        
        if k_remaining == 0:
            diam = check_diameter(adj_state)
            if diam <= 4:
                return True
            return False
        
        # Pruning: if we have picked enough edges but diameter is still > 4, 
        # and we can't pick more, it fails.
        
        # Heuristic lower bound: 
        # If current diameter is D, and we have k_remaining edges, 
        # can we reduce it to 4?
        # This is hard to estimate tightly without simulation.
        # We will rely on the small search space and early termination.
        
        # If we have reached the end of candidates
        if idx >= len(candidates):
            return False
        
        # Option 1: Skip candidate idx
        # Option 2: Take candidate idx
        
        # Try taking candidate idx first? 
        # Usually, taking "good" edges early helps find a solution faster.
        u, v = candidates[idx]
        
        # Check if adding (u,v) creates a triangle
        # Triangle if u and v share a common neighbor in adj_state
        if (adj_state[u] & adj_state[v]) == 0:
            # No triangle
            # Update state
            adj_state[u] |= 1 << v
            adj_state[v] |= 1 << u
            selected.append((u, v))
            
            if search(idx + 1, selected, adj_state, k_remaining - 1):
                return True
            
            # Backtrack
            selected.pop()
            adj_state[u] ^= 1 << v
            adj_state[v] ^= 1 << u
            
        # Try skipping candidate idx
        if search(idx + 1, selected, adj_state, k_remaining):
            return True
            
        return False

    # Try k from 1 to 10
    start_time = time.time()
    for k in range(1, 11):
        # Reset state
        adj_state = adj[:]  # Copy base adj
        selected = []
        
        # Time guard
        if time.time() - start_time > 45:
            break
            
        if search(0, selected, adj_state, k):
            # Found solution with k edges
            # Convert to list of lists
            result = [[u, v] for u, v in selected]
            # Ensure types are int
            return result
            
    # If no solution found within time or k limit, return empty or best effort
    # The task implies a solution exists.
    # If we timeout, we might return a partial or invalid result, which scores 0.
    # Better to return the best valid one found? 
    # The search returns True only if valid.
    # If we exit loop without return, we must return something.
    # Let's try a greedy fallback if B&B fails?
    # Or just return empty list? No, must be valid.
    # Given n=24, k should be small.
    
    # Fallback: Return empty (invalid) to avoid hanging, or raise error?
    # The prompt says "Any broken rule makes the whole result invalid".
    # Returning empty list is invalid if diameter > 4.
    # But if we can't find one, we can't do much.
    # However, with n=24, it should be fast.
    
    return []

# EVOLVE-BLOCK-END
