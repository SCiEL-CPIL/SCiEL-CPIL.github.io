import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Trivial solution: 0,1,2,...,23
    trivial = list(range(N))
    trivial_score = _cycle_score(trivial)
    
    best_cycle = trivial
    best_score = trivial_score
    
    # Binary search on K using bitmask DP (Held-Karp)
    # For a given K, we check if there's a Hamilton cycle using only edges with position <= K
    
    def hamilton_cycle_exists(K):
        """Returns (exists, path) where path is a list of vertices if a Hamilton cycle exists
        using only edges with position <= K, else (False, None)"""
        
        # Build adjacency: adj[u] is a bitmask of vertices v where edge (u,v) has position <= K
        adj = [0] * N
        for i, (u, v) in enumerate(EDGE_ORDER, 1):
            if i > K:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check that every vertex has degree at least 2 (necessary for Hamilton cycle)
        for u in range(N):
            if bin(adj[u]).count('1') < 2:
                return False, None
        
        # DP: dp[mask][v] = True if there's a path from 0 visiting exactly the vertices in mask, ending at v
        # mask includes 0 and v
        # Use bitset representation: dp[mask] is a bitmask of possible ending vertices
        
        full_mask = (1 << N) - 1
        start = 0
        start_bit = 1 << start
        
        # dp[mask] = bitmask of vertices v such that there's a path 0 -> ... -> v visiting exactly mask
        dp = [0] * (1 << N)
        dp[start_bit] = start_bit  # Path of length 1: just vertex 0, ending at 0
        
        # parent tracking for reconstruction
        # parent[mask][v] = previous vertex in the path
        # This is memory-heavy: 2^24 * 24 ints. Let's use a different approach.
        # Instead, store parent as a dict or only for the final answer.
        
        # Actually, let's track parent more carefully.
        # parent[v][mask] = previous vertex, but that's also large.
        # Alternative: store the full path by backtracking, but we need to know how we got there.
        
        # Let's use: prev[mask][v] = the vertex before v in the path that achieves mask ending at v
        # Size: 2^24 * 24 * 1 byte = ~1MB if we use bytes. Let's try.
        
        # Actually, 2^24 * 24 = 402M entries. That's too much memory.
        
        # Alternative approach: only track the existence, and if it exists, do a separate DFS/BFS to extract the path.
        # Or: use a smaller state representation.
        
        # Let me reconsider. For N=24, 2^24 = 16.7M masks. Each mask stores a 24-bit integer (bitmask of possible endpoints).
        # So dp is 16.7M integers, which is ~64MB. That's feasible.
        # For parent tracking, we could store it differently.
        
        # Strategy: 
        # 1. First pass: determine if a Hamilton cycle exists (existence only)
        # 2. Second pass: if it exists, extract the actual path using a more targeted search
        
        # Pass 1: existence check
        for mask in range(1 << N):
            if not (mask & start_bit):
                continue
            current_endpoints = dp[mask]
            if current_endpoints == 0:
                continue
            # For each endpoint v in current_endpoints, try to extend to unvisited vertices
            endpoints = current_endpoints
            unvisited = full_mask ^ mask
            while endpoints:
                v_bit = endpoints & (-endpoints)
                v = v_bit.bit_length() - 1
                endpoints ^= v_bit
                # Try extending from v to any unvisited neighbor
                neighbors = adj[v] & unvisited
                while neighbors:
                    w_bit = neighbors & (-neighbors)
                    w = w_bit.bit_length() - 1
                    neighbors ^= w_bit
                    new_mask = mask | w_bit
                    dp[new_mask] |= w_bit
        
        # Check if full_mask is reachable and can close back to start
        if dp[full_mask] == 0:
            return False, None
        # Check if any endpoint can connect back to start
        for v_bit in bin(dp[full_mask]).split('1'):
            pass
        # Better: check if (dp[full_mask] & start_bit) is set, or if any endpoint v has edge (v, start) with pos <= K
        endpoint_mask = dp[full_mask]
        # An endpoint v can close the cycle if edge (v, start) exists in our prefix
        closable = 0
        for v in range(N):
            if endpoint_mask & (1 << v):
                # Check if edge (v, start) is in prefix
                if _edge_idx(v, start) <= K:
                    closable |= (1 << v)
        
        if closable == 0:
            return False, None
        
        # Existence confirmed. Now extract a path.
        # We'll do a backward search from (full_mask, some closable endpoint) back to (start_bit, start)
        
        # Pick a closable endpoint
        endpoint = (closable & (-closable)).bit_length() - 1
        
        # Reconstruct path by backtracking
        # We need to know for each (mask, v), which previous vertex led to this state.
        # This requires storing parent information. Let's do a second DP that stores parents.
        
        # Actually, let's do it differently. We'll store the parent as we go.
        # parent[mask][v] = previous vertex u such that dp[mask ^ (1<<v)] had u as endpoint and edge (u,v) exists.
        
        # Memory concern: 2^24 * 24 bytes = ~1GB. Too much.
        
        # Alternative: iterative deepening or use the fact that we just need ONE path.
        # Let's do a BFS/DFS from the end state back to start, using the dp array as a guide.
        
        # Actually, the cleanest approach for extraction: 
        # Do the DP again, but this time store for each (mask, v) the parent u.
        # Use a compressed representation: parent is a 5-bit value (0-23) per (mask, v) pair.
        # That's 2^24 * 24 * 1 byte = ~1GB. Still too much.
        
        # Let's try a different strategy for extraction:
        # We know dp[full_mask] has some endpoint e that can close to start.
        # We want to find a path 0 -> v1 -> v2 -> ... -> v23 = e such that all edges are in prefix.
        # This is equivalent to finding a Hamilton path from 0 to e in the graph induced by prefix edges.
        
        # We can do this with a separate DP or even a DFS with memoization.
        # But the state space is the same.
        
        # Practical approach: use the dp array we already computed, and backtrack greedily.
        # For mask = full_mask, endpoint = e:
        #   Find a vertex u in (mask without e) such that:
        #     - edge (u, e) is in prefix
        #     - dp[mask without e] has u as an endpoint
        #   If found, recurse on (mask without e, u)
        
        def backtrack(mask, end):
            """Reconstruct path from start to end visiting exactly the vertices in mask."""
            if mask == start_bit:
                return [start]
            
            prev_mask = mask ^ (1 << end)
            # Find a vertex u in prev_mask such that:
            #   1. edge (u, end) is in prefix (position <= K)
            #   2. dp[prev_mask] has u as an endpoint
            candidates = dp[prev_mask]
            # Also restrict to vertices u where edge (u, end) exists in prefix
            candidates &= adj[end]
            # And u must be in prev_mask
            candidates &= prev_mask
            
            while candidates:
                u_bit = candidates & (-candidates)
                u = u_bit.bit_length() - 1
                candidates ^= u_bit
                # Verify that dp[prev_mask] indeed has u
                if dp[prev_mask] & u_bit:
                    # Check edge (u, end) is in prefix
                    if _edge_idx(u, end) <= K:
                        sub_path = backtrack(prev_mask, u)
                        if sub_path is not None:
                            sub_path.append(end)
                            return sub_path
            return None
        
        path = backtrack(full_mask, endpoint)
        if path is None:
            return False, None
        
        # The path is 0 -> ... -> endpoint, and we close with edge (endpoint, start)
        # Verify all edges are in prefix
        for i in range(len(path) - 1):
            if _edge_idx(path[i], path[i+1]) > K:
                return False, None
        if _edge_idx(path[-1], start) > K:
            return False, None
        
        # The Hamilton cycle is path[0], path[1], ..., path[-1], and back to path[0]
        # As a permutation: path itself (with implicit closing edge)
        return True, path
    
    # Binary search on K
    lo, hi = 24, best_score - 1
    
    while lo <= hi and time.time() < deadline - 3.0:
        mid = (lo + hi) // 2
        exists, path = hamilton_cycle_exists(mid)
        
        if exists:
            if path is not None:
                sc = _cycle_score(path)
                if sc < best_score:
                    best_score = sc
                    best_cycle = list(path)
                hi = mid - 1
            else:
                # Shouldn't happen, but handle gracefully
                hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle

