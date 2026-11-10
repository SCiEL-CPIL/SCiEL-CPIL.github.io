<<<<<<< SEARCH
import time
import random
from functools import lru_cache

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency with bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_mask = (1 << n) - 1
    
    deadline = time.time() + 82  # 82s safety margin for 90s limit
    best_cycles = []
    random.seed(12345)
    
    # Fast Hamiltonian cycle finder using bitmask DP
    # Fix start vertex to 0 for symmetry breaking
    def find_hc(used_edges_mask, adj_mask_local):
        """Find a Hamiltonian cycle in graph with adjacency masks, avoiding used edges.
        used_edges_mask is a set of edge indices already used.
        Returns list of vertices forming the cycle, or None."""
        # Build local adjacency masks that exclude used edges
        # This is expensive, so we do it carefully
        local_adj = [0] * n
        for u in range(n):
            local_adj[u] = adj_mask_local[u]
            # Remove edges that are used
            for v in range(n):
                if (local_adj[u] >> v) & 1:
                    ek = (min(u, v), max(u, v))
                    if ek in used_edges_mask:
                        local_adj[u] &= ~(1 << v)
        
        # DP: dp[mask][last] = True if there's a path from 0 through all vertices in mask ending at last
        # mask includes vertex 0
        # Use dict for sparse DP
        from collections import defaultdict
        dp = defaultdict(set)  # mask -> set of last vertices reachable
        dp[1] = {0}  # mask with only vertex 0, last is 0
        
        for mask in range(1, 1 << n):
            if not (mask & 1):  # Must include vertex 0
                continue
            if mask in dp:
                lasts = dp[mask]
                for last in lasts:
                    # Try to extend to next vertex
                    remaining = all_mask & ~mask
                    if remaining == 0:
                        continue
                    # Get available neighbors of last not in mask
                    avail = local_adj[last] & remaining
                    while avail:
                        lsb = avail & (-avail)
                        nxt = lsb.bit_length() - 1
                        new_mask = mask | lsb
                        dp[new_mask].add(nxt)
                        avail ^= lsb
        
        # Check if we have a Hamiltonian cycle: mask = all_mask, last can connect to 0
        if all_mask in dp:
            for last in dp[all_mask]:
                if (local_adj[last] >> 0) & 1:
                    # Reconstruct path
                    # We need to backtrack, but we didn't store parent pointers
                    # Let's use a different approach: store parent
                    pass
        
        # The above doesn't reconstruct. Let me use a proper DP with parent tracking.
        # This is getting complex. Let me use a simpler iterative approach with parent tracking.
        
        return None
    
    # Better approach: Use iterative DP with parent tracking for Hamiltonian cycle
    def find_hamiltonian_cycle_fast(used_edges):
        """Find a Hamiltonian cycle avoiding used edges. Returns list of vertices or None."""
        # Build local adjacency excluding used edges
        local_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                ek = (min(u, v), max(u, v))
                if ek in used_edges:
                    continue
                if (adj_mask[u] >> v) & 1:
                    local_adj[u].add(v)
        
        # DP with parent tracking
        # dp[mask][last] = previous vertex in path, or -1 if not reachable
        # To save memory, we only need the final layer, but we need parent for reconstruction
        # Use dictionary: key = (mask, last), value = prev
        parent = {}
        # Start: mask = {0}, last = 0, no parent
        # We'll iterate over mask size
        
        from collections import defaultdict
        reachable = defaultdict(set)
        reachable[1].add(0)
        
        for mask in range(1, 1 << n):
            if not (mask & 1):
                continue
            if mask not in reachable:
                continue
            for last in reachable[mask]:
                # Try to extend
                for nxt in local_adj[last]:
                    if (mask >> nxt) & 1:
                        continue
                    new_mask = mask | (1 << nxt)
                    if new_mask not in reachable or nxt not in reachable[new_mask]:
                        reachable[new_mask].add(nxt)
                        parent[(new_mask, nxt)] = last
        
        # Check for Hamiltonian cycle
        if all_mask in reachable:
            for last in reachable[all_mask]:
                if 0 in local_adj[last]:
                    # Reconstruct
                    path = []
                    cur_mask = all_mask
                    cur_last = last
                    while cur_last != -1:
                        path.append(cur_last)
                        if cur_last == 0:
                            break
                        prev = parent.get((cur_mask, cur_last), -1)
                        if prev == -1:
                            return None
                        cur_mask ^= (1 << cur_last)
                        cur_last = prev
                    if cur_last == 0:
                        path.reverse()
                        return path
        return None
    
    # Even faster: use the fact that we can fix start=0 and use array-based DP
    def find_hc_optimized(used_edges):
        """Optimized Hamiltonian cycle finder."""
        local_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                ek = (min(u, v), max(u, v))
                if ek not in used_edges and (adj_mask[u] >> v) & 1:
                    local_adj[u].add(v)
        
        # Check degree constraints first
        for v in range(n):
            if len(local_adj[v]) < 2:
                return None
        
        # DP: dp[mask] = set of possible last vertices
        # For memory efficiency, process by mask size
        from collections import defaultdict
        dp = [None] * (1 << n)
        dp[1] = {0}
        
        # Also track parent for reconstruction
        # parent[mask][last] = prev
        # This uses too much memory. Let's try a different approach.
        
        # Use the Held-Karp style but with parent tracking via a single array
        # For each (mask, last), store the previous vertex
        # n=21, 2^21 = 2M states, each with up to 21 lasts = 44M entries max
        # That's too much. Let's use a smarter approach.
        
        # Alternative: just do DFS with memoization and early termination
        # Since n=21, and we have degree constraints, DFS with good heuristics should work
        
        visited = [False] * n
        visited[0] = True
        path = [0]
        
        def dfs(current, depth):
            if time.time() > deadline:
                return False
            if depth == n:
                # Check if we can close the cycle
                if 0 in local_adj[current]:
                    return True
                return False
            
            # Get unvisited neighbors
            unvisited = [v for v in local_adj[current] if not visited[v]]
            if not unvisited:
                return False
            
            # Heuristic: visit vertices with fewest options first
            unvisited.sort(key=lambda v: len([x for x in local_adj[v] if not visited[x]]))
            
            for nxt in unvisited:
                visited[nxt] = True
                path.append(nxt)
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None
    
    # Greedy extraction of Hamiltonian cycles
    def greedy_extract_cycles(max_cycles=10):
        used_edges = set()
        cycles = []
        
        for _ in range(max_cycles):
            if time.time() > deadline:
                break
            
            cycle = find_hc_optimized(used_edges)
            if cycle is None:
                break
            
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                ek = (min(u, v), max(u, v))
                used_edges.add(ek)
        
        return cycles
    
    # Try greedy extraction
    result = greedy_extract_cycles(10)
    
    if len(result) > len(best_cycles):
        best_cycles = result
    
    # If we found few cycles, try randomized restarts with different edge orderings
    if len(best_cycles) < 5 and time.time() < deadline - 10:
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Randomly shuffle the order in which we try to find cycles
            # by perturbing the heuristic
            used_edges = set()
            cycles = []
            
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Use a slightly randomized DFS
                cycle = _find_hc_randomized(used_edges, random)
                if cycle is None:
                    break
                
                cycles.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    ek = (min(u, v), max(u, v))
                    used_edges.add(ek)
            
            if len(cycles) > len(best_cycles):
                best_cycles = cycles
                if len(best_cycles) >= 8:
                    break
    
    def _find_hc_randomized(used_edges, rng):
        """Find HC with randomized heuristic."""
        local_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                ek = (min(u, v), max(u, v))
                if ek not in used_edges and (adj_mask[u] >> v) & 1:
                    local_adj[u].add(v)
        
        for v in range(n):
            if len(local_adj[v]) < 2:
                return None
        
        visited = [False] * n
        visited[0] = True
        path = [0]
        
        def dfs(current, depth):
            if time.time() > deadline - 5:
                return False
            if depth == n:
                if 0 in local_adj[current]:
                    return True
                return False
            
            unvisited = [v for v in local_adj[current] if not visited[v]]
            if not unvisited:
                return False
            
            # Randomized sort with some heuristic
            unvisited.sort(key=lambda v: (len([x for x in local_adj[v] if not visited[x]]), rng.random()))
            
            for nxt in unvisited:
                visited[nxt] = True
                path.append(nxt)
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None
    
    # Final result
    if not best_cycles:
        # Absolute fallback: return a single known valid cycle
        # Find any cycle in the graph
        # Use BFS/DFS to find a simple cycle
        visited = [False] * n
        visited[0] = True
        stack = [(0, [0])]
        cycle_found = None
        while stack:
            node, path = stack.pop()
            for nb in range(n):
                if (adj_mask[node] >> nb) & 1:
                    if nb == 0 and len(path) >= 3:
                        cycle_found = list(path)
                        break
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append((nb, path + [nb]))
            if cycle_found:
                break
        
        if cycle_found is None:
            # Shouldn't happen for this connected graph
            return [[0, 1, 2]]
        
        best_cycles = [cycle_found]
    
    # Ensure correct types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result


# Define _find_hc_randomized at module level if needed, but it's nested above
# Actually, let me restructure to avoid the closure issue


# Rewrite with proper structure
def construct():
=======
import time
import random
from functools import lru_cache

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def construct():
>>>>>>> REPLACE

