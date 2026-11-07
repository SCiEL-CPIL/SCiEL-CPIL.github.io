P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
import time
import sys

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
            
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        size = 1 << n
        
        is_clique = [False] * size
        is_indep = [False] * size
        is_clique[0] = True
        is_indep[0] = True
        
        for mask in range(1, size):
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            
            if is_clique[rest] and (rest & ~adj[v]) == 0:
                is_clique[mask] = True
            if is_indep[rest] and (rest & adj[v]) == 0:
                is_indep[mask] = True
                
        valid = [is_clique[m] or is_indep[m] for m in range(size)]
        
        dp = [float('inf')] * size
        dp[0] = 0
        
        # Optimize by checking subsets in a smarter way?
        # Standard subset DP is O(3^n).
        # 3^14 is small enough.
        
        start_time = time.time()
        deadline = 85 # 90s limit, keep margin
        
        for mask in range(1, size):
            if time.time() > deadline:
                break
            best = float('inf')
            s = mask
            while s:
                if valid[s]:
                    val = 1 + dp[mask ^ s]
                    if val < best:
                        best = val
                        if best == 1:
                            break
                s = (s - 1) & mask
            dp[mask] = best
            
        full_mask = size - 1
        min_parts = dp[full_mask]
        
        # Reconstruct
        parts = []
        remaining = full_mask
        while remaining:
            if time.time() > deadline:
                break
            found = False
            s = remaining
            while s:
                if valid[s] and dp[remaining ^ s] == dp[remaining] - 1:
                    part = []
                    for v in range(n):
                        if s & (1 << v):
                            part.append(v)
                    parts.append(part)
                    remaining ^= s
                    found = True
                    break
                s = (s - 1) & remaining
            if not found:
                break
                
        # Fallback
        all_vertices = set()
        for p in parts:
            for v in p:
                all_vertices.add(v)
        if all_vertices != set(range(n)) or len(parts) == 0:
            parts = [[i] for i in range(n)]
            
        return parts
