import time
import sys
from itertools import combinations

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def cycle_score(cycle):
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        p = get_pos(u, v)
        if p > mx:
            mx = p
    return mx

def hamilton_cycle_dp(adj_bitmasks, n, start=0):
    """
    Use DP over subsets to find a Hamilton cycle.
    adj_bitmasks[v] is a bitmask of neighbors of v.
    Returns the cycle as a list, or None.
    """
    full = (1 << n) - 1
    start_bit = 1 << start
    
    # dp[mask][v] = True if there's a path from start visiting exactly mask, ending at v
    # We use a list of dicts for memory efficiency
    # dp[mask] = set of vertices v such that there's a Hamiltonian path from start through mask ending at v
    # For n=24, we have 2^24 masks, each with up to 24 entries
    
    # Actually, let's use a more memory-efficient approach:
    # dp[mask] = bitmask of valid ending vertices
    dp = [0] * (1 << n)
    dp[start_bit] = start_bit  # Only start vertex, ending at start
    
    # prev[mask][v] = previous vertex to reconstruct path
    # This is too much memory. Instead, let's just check existence first,
    # then reconstruct.
    
    # For reconstruction, we'll store prev as a 2D array
    # prev[mask][v] = u means the path to (mask, v) came from u
    # But 2^24 * 24 integers = too much. Let's use a different approach.
    
    # Alternative: just do the DP for existence, then use backtracking with the DP table
    # to reconstruct.
    
    # Let's use a list of bytearrays or just store the predecessor differently.
    # For n=24, 2^24 = 16M entries. If each is a 24-bit mask, that's 16M * 3 bytes = 48MB. OK.
    
    # Actually, let's just do existence check first with the bitmask DP.
    # Then if it exists, do a guided backtracking to find the actual cycle.
    
    # Existence check:
    # dp[mask] = bitmask of vertices v such that there's a path from start visiting mask, ending at v
    for mask in range(1 << n):
        if dp[mask] == 0:
            continue
        # For each vertex v in dp[mask], try to extend to neighbors
        # v must be in mask
        for v in range(n):
            if not (dp[mask] >> v) & 1:
                continue
            # Try extending to neighbors of v not in mask
            avail = adj_bitmasks[v] & ~mask
            while avail:
                bit = avail & (-avail)
                w = bit.bit_length() - 1
                new_mask = mask | bit
                dp[new_mask] |= bit
                avail ^= bit
    
    # Check if full mask has any vertex v that connects back to start
    if dp[full] == 0:
        return None
    
    # Find a valid ending vertex
    end_mask = dp[full]
    # The end vertex must connect to start
    for v in range(n):
        if (end_mask >> v) & 1:
            if (adj_bitmasks[v] >> start) & 1:
                # We have a Hamilton cycle: start -> ... -> v -> start
                # Now reconstruct the path
                # We need to backtrack. Let's redo the DP storing predecessors.
                break
    else:
        return None
    
    # For reconstruction, we need predecessors. Let's do a second pass.
    # Actually, let's just do a guided backtracking using the dp table.
    # At state (mask, v), we know which (prev_mask, u) transitions led here.
    # We'll just do DFS guided by the dp table.
    
    # Reconstruct: start at (full, end_v), go backwards
    # We need to know: for state (mask, v), which u in mask\{v} has u adjacent to v and dp[mask ^ (1<<v)] has u set
    
    end_v = None
    for v in range(n):
        if (end_mask >> v) & 1 and (adj_bitmasks[v] >> start) & 1:
            end_v = v
            break
    
    if end_v is None:
        return None
    
    # Backtrack to reconstruct path
    path = [end_v]
    mask = full
    current = end_v
    
    while mask != start_bit:
        prev_mask = mask ^ (1 << current)
        # Find u in prev_mask such that u is adjacent to current and dp[prev_mask] has u
        found = False
        for u in range(n):
            if not (prev_mask >> u) & 1:
                continue
            if not (dp[prev_mask] >> u) & 1:
                continue
            if (adj_bitmasks[current] >> u) & 1:
                path.append(u)
                current = u
                mask = prev_mask
                found = True
                break
        if not found:
            return None  # Should not happen
    
    path.append(start)
    path.reverse()
    return path

def find_hamilton_cycle_adjacency(adj_list, n, start=0, deadline=None):
    """
    Find Hamilton cycle in graph given by adjacency list.
    adj_list[v] = set of neighbors of v.
    Uses DP over subsets.
    """
    # Convert to bitmasks
    adj_bitmasks = [0] * n
    for v in range(n):
        mask = 0
        for u in adj_list[v]:
            mask |= (1 << u)
        adj_bitmasks[v] = mask
    
    # Quick check
    for v in range(n):
        if adj_bitmasks[v] == 0:
            return None
    
    return hamilton_cycle_dp(adj_bitmasks, n, start)

def local_search_refine(cycle, n, deadline=None):
    """Improve a Hamilton cycle by 2-opt and segment reversal moves."""
    if deadline is not None and time.time() > deadline:
        return cycle
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        # 2-opt: reverse a segment
        for i in range(n):
            for j in range(i + 2, n):
                if deadline is not None and time.time() > deadline:
                    return best_cycle
                if j == n - 1 and i == 0:
                    continue
                
                new_cycle = best_cycle[:]
                # Reverse segment i..j
                new_cycle[i:j+1] = reversed(new_cycle[i:j+1])
                
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    improved = True
        
        # Swap non-adjacent vertices
        for i in range(n):
            for j in range(i + 2, n):
                if deadline is not None and time.time() > deadline:
                    return best_cycle
                if abs(i - j) == 1 or (i == 0 and j == n - 1):
                    continue
                
                new_cycle = best_cycle[:]
                new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    improved = True
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = N
    total_edges = n * (n - 1) // 2  # 276
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Binary search for minimum threshold T
    lo, hi = n, total_edges
    
    # First establish an upper bound
    # Try with all edges
    adj_full = [set() for _ in range(n)]
    for (u, v) in EDGE_ORDER:
        adj_full[u].add(v)
        adj_full[v].add(u)
    
    if time.time() < deadline:
        cycle = find_hamilton_cycle_adjacency(adj_full, n, 0, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_score = cycle_score(cycle)
            hi = best_score
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [set() for _ in range(n)]
        for i in range(mid):
            u, v = EDGE_ORDER[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Quick degree check
        feasible = True
        for v in range(n):
            if len(adj[v]) < 2:
                feasible = False
                break
        
        if not feasible:
            lo = mid + 1
            continue
        
        # Check connectivity
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
        if count != n:
            lo = mid + 1
            continue
        
        if time.time() > deadline:
            break
            
        cycle = find_hamilton_cycle_adjacency(adj, n, 0, deadline)
        
        if cycle is not None:
            score = cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Refine with local search
    if best_cycle is not None and time.time() < deadline:
        refined = local_search_refine(best_cycle, n, deadline)
        if refined is not None:
            rs = cycle_score(refined)
            if rs < best_score:
                best_score = rs
                best_cycle = refined
    
    # Fallback
    if best_cycle is None:
        best_cycle = list(range(n))
    
    return best_cycle
