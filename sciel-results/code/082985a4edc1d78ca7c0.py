import time
import random
import sys

# Build the edge order from the seed program's data
_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24
ALL_MASK = (1 << N) - 1

# Precompute: for each edge (u,v) with u<v, its 1-based arrival index
_edge_index = {}
for i, (u, v) in enumerate(_EDGE_ORDER):
    if u > v:
        u, v = v, u
    _edge_index[(u, v)] = i + 1

def get_edge_idx(u, v):
    if u > v:
        u, v = v, u
    return _edge_index[(u, v)]

def build_adj_bitmasks(K):
    """Build adjacency bitmasks for the graph containing edges with index <= K."""
    adj = [0] * N
    for (u, v), idx in _edge_index.items():
        if idx <= K:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def has_hamilton_cycle(adj, start=0):
    """
    Check if the graph defined by adj bitmasks has a Hamilton cycle.
    Uses DFS with bitmasks and pruning.
    Returns True if a cycle exists, False otherwise.
    """
    # Pruning: Check if graph is connected (ignoring isolated vertices if any, but here we want all)
    # Actually, for Hamilton cycle, every vertex must have degree >= 2.
    for v in range(N):
        if bin(adj[v]).count('1') < 2:
            return False

    visited = [False] * N
    visited[start] = True
    path = [start]
    
    # Connectivity check on unvisited subgraph is expensive but powerful.
    # We'll use a simpler degree-based pruning first.
    
    def dfs(current, visited_mask, path_len):
        if path_len == N:
            # Check if we can close the cycle
            if (adj[current] & (1 << start)) != 0:
                return True
            return False
        
        # Pruning: Check degrees of unvisited vertices
        # If any unvisited vertex has degree < 2 in the remaining graph (unvisited + current), it's impossible.
        # Note: The "remaining graph" for unvisited vertices v is adj[v] & ~visited_mask.
        # But they also need to connect to the path. 
        # A standard pruning: for each unvisited v, deg_in_unvisited(v) must be >= 2, 
        # unless v is the last one to be visited? No, even the last one needs 2 connections in the final cycle.
        # Actually, in the partial path, the unvisited vertices must form a structure that can be embedded.
        # Simple check: if an unvisited vertex has no neighbors in unvisited set and is not adjacent to current or start (if last), it fails.
        
        # Let's stick to a more efficient check:
        # For every unvisited vertex v, the number of neighbors in (unvisited U {current}) must be at least 2.
        # Wait, if v is the *last* vertex added, it connects to current and start.
        # So, if path_len == N-1, the next vertex v must be adjacent to current AND start.
        
        candidates = adj[current] & ~visited_mask
        
        if not candidates:
            return False

        # Pruning for the last step
        if path_len == N - 1:
            # Only one candidate left? Or check if any candidate connects to start
            # Actually, we just iterate candidates. If we pick v, we check if v-start exists.
            # So we can filter candidates to those that connect to start.
            candidates &= adj[start]
            if not candidates:
                return False
            # Pick the first one (or any)
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            return True # If we reach here, the last step is valid
            
        # General pruning:
        # For each unvisited vertex v, let deg_unvisited = popcount(adj[v] & ~visited_mask).
        # If deg_unvisited < 2, return False.
        # Why? Because in the final cycle, v must have 2 neighbors. 
        # One or both might be 'current' or 'start' (if v is first/last).
        # But if v is not first or last, both neighbors must be in unvisited set? 
        # No, one neighbor could be 'current'. 
        # This pruning is tricky. A safer one:
        # If an unvisited vertex has degree 0 in the subgraph induced by unvisited vertices, 
        # it can only connect to visited vertices. It can connect to at most 2 visited vertices (current and potentially start if it's the last one).
        # If it's not the last one, it needs 2 unvisited neighbors? No.
        
        # Let's use a simpler, effective pruning:
        # If the unvisited graph is disconnected, return False.
        # We can check connectivity of the subgraph induced by ~visited_mask.
        unvisited_mask = ~visited_mask & ALL_MASK
        if unvisited_mask == 0:
             return False
        # Pick a bit from unvisited
        start_bit = unvisited_mask & -unvisited_mask
        start_v = start_bit.bit_length() - 1
        # BFS/DFS from start_v within unvisited_mask
        seen = 0
        stack = [start_v]
        while stack:
            u = stack.pop()
            if seen & (1 << u):
                continue
            seen |= (1 << u)
            neighbors = adj[u] & unvisited_mask & ~seen
            while neighbors:
                nb_bit = neighbors & -neighbors
                nb = nb_bit.bit_length() - 1
                stack.append(nb)
                neighbors &= neighbors - 1
        if seen != unvisited_mask:
            return False

        # Iterate through candidates
        # Order: try vertices with fewer unvisited neighbors first (most constrained)
        cand_list = []
        temp_cand = candidates
        while temp_cand:
            v_bit = temp_cand & -temp_cand
            v = v_bit.bit_length() - 1
            # Count unvisited neighbors for v
            unvisited_neighbors = adj[v] & unvisited_mask
            deg = bin(unvisited_neighbors).count('1')
            cand_list.append((deg, v))
            temp_cand &= temp_cand - 1
        
        cand_list.sort()
        
        for deg, v in cand_list:
            if dfs(v, visited_mask | (1 << v), path_len + 1):
                return True
        return False

    return dfs(start, 1 << start, 1)

def find_cycle(adj, start=0):
    """Find and return the Hamilton cycle as a list of vertices."""
    visited = [False] * N
    visited[start] = True
    path = [start]
    
    def dfs(current, visited_mask, path_len):
        if path_len == N:
            if (adj[current] & (1 << start)) != 0:
                return True
            return False
        
        candidates = adj[current] & ~visited_mask
        
        if not candidates:
            return False

        if path_len == N - 1:
            candidates &= adj[start]
            if not candidates:
                return False
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            path.append(v)
            return True
        
        # Connectivity check
        unvisited_mask = ~visited_mask & ALL_MASK
        start_bit = unvisited_mask & -unvisited_mask
        start_v = start_bit.bit_length() - 1
        seen = 0
        stack = [start_v]
        while stack:
            u = stack.pop()
            if seen & (1 << u):
                continue
            seen |= (1 << u)
            neighbors = adj[u] & unvisited_mask & ~seen
            while neighbors:
                nb_bit = neighbors & -neighbors
                nb = nb_bit.bit_length() - 1
                stack.append(nb)
                neighbors &= neighbors - 1
        if seen != unvisited_mask:
            return False

        cand_list = []
        temp_cand = candidates
        while temp_cand:
            v_bit = temp_cand & -temp_cand
            v = v_bit.bit_length() - 1
            unvisited_neighbors = adj[v] & unvisited_mask
            deg = bin(unvisited_neighbors).count('1')
            cand_list.append((deg, v))
            temp_cand &= temp_cand - 1
        
        cand_list.sort()
        
        for deg, v in cand_list:
            path.append(v)
            if dfs(v, visited_mask | (1 << v), path_len + 1):
                return True
            path.pop()
            
        return False

    if dfs(start, 1 << start, 1):
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Binary search for the minimal K
    low = 24  # Minimum edges for a cycle
    high = len(_EDGE_ORDER)  # 276
    
    # Quick check: if high works
    adj_high = build_adj_bitmasks(high)
    if not has_hamilton_cycle(adj_high):
        # Fallback to a greedy/random cycle if no cycle exists even in full graph (should not happen)
        # But K24 always has one.
        return list(range(N))

    best_cycle = None
    best_K = high
    
    # We can optimize by checking connectivity first in binary search
    while low < high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        adj_mid = build_adj_bitmasks(mid)
        
        # Optional: quick degree check
        valid = True
        for v in range(N):
            if bin(adj_mid[v]).count('1') < 2:
                valid = False
                break
        
        if valid:
            if has_hamilton_cycle(adj_mid):
                best_K = mid
                high = mid
            else:
                low = mid + 1
        else:
            low = mid + 1

    # Find the cycle for best_K
    if time.time() < deadline:
        adj_best = build_adj_bitmasks(best_K)
        # Try different starting points if 0 fails (though 0 should work if cycle exists)
        for s in range(N):
            cycle = find_cycle(adj_best, s)
            if cycle:
                best_cycle = cycle
                break
    
    if not best_cycle:
        # Fallback: use a known valid cycle from full graph
        adj_full = build_adj_bitmasks(high)
        for s in range(N):
            cycle = find_cycle(adj_full, s)
            if cycle:
                best_cycle = cycle
                break
    
    if not best_cycle:
        # Last resort: identity cycle (might be invalid score-wise but returns something)
        best_cycle = list(range(N))

    return best_cycle
