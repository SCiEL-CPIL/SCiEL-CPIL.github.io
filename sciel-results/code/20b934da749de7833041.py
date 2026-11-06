"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    # inv[x] is inverse of x mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # Precompute all non-empty subsets
    # We store: (size, residue, terms_list)
    # We also need a way to map back to terms for the final result.
    # We can store the mask and have a mask_to_terms dictionary.
    
    mask_to_terms = {}
    # subset_data: list of (size, residue, mask)
    subset_data = []
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % 101
        size = len(terms)
        mask_to_terms[mask] = terms
        subset_data.append((size, s, mask))
        
    # Sort by size to easily filter for K
    subset_data.sort(key=lambda x: x[0])
    
    def hopcroft_karp(adj, n_left=101):
        """
        adj: list of lists. adj[r] contains mask IDs.
        Returns matchL array where matchL[r] is the matched mask, or -1.
        """
        matchL = [-1] * n_left
        matchR = [-1] * 1024  # masks are 1..1023
        
        def bfs():
            dist = [-1] * n_left
            queue = deque()
            for r in range(n_left):
                if matchL[r] == -1:
                    dist[r] = 0
                    queue.append(r)
            found = False
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    m = matchR[mask]
                    if m == -1:
                        found = True
                    elif dist[m] == -1:
                        dist[m] = dist[r] + 1
                        queue.append(m)
            return found, dist
        
        def dfs(r, dist, iter_ptr):
            for i in range(iter_ptr[r], len(adj[r])):
                iter_ptr[r] = i + 1
                mask = adj[r][i]
                m = matchR[mask]
                if m == -1 or (dist[m] == dist[r] + 1 and dfs(m, dist, iter_ptr)):
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
            dist[r] = -1
            return False
        
        matching_size = 0
        while True:
            if time.time() > deadline - 2:
                break
            found, dist = bfs()
            if not found:
                break
            iter_ptr = [0] * n_left
            for r in range(n_left):
                if matchL[r] == -1:
                    if dfs(r, dist, iter_ptr):
                        matching_size += 1
        return matchL, matching_size

    def can_match(K):
        """
        Check if perfect matching exists with all subset sizes <= K.
        Returns matchL if successful, else None.
        """
        # Build adjacency list
        adj = [[] for _ in range(101)]
        for size, residue, mask in subset_data:
            if size > K:
                break
            adj[residue].append(mask)
            
        matchL, matching_size = hopcroft_karp(adj)
        if matching_size == 101:
            return matchL
        return None

    # Binary search for minimum K
    lo, hi = 1, 10
    best_K = 10
    best_matchL = None
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        matchL = can_match(mid)
        if matchL is not None:
            best_K = mid
            best_matchL = matchL
            hi = mid - 1
        else:
            lo = mid + 1
            
    # If binary search didn't find a match (should not happen, K=10 always works), fallback
    if best_matchL is None:
        # Try K=10 explicitly again or just use any valid assignment
        # Since K=10 includes all subsets, a matching must exist if the problem is solvable.
        # Let's force a match with K=10.
        adj = [[] for _ in range(101)]
        for size, residue, mask in subset_data:
            adj[residue].append(mask)
        matchL, _ = hopcroft_karp(adj)
        best_matchL = matchL

    # Build result
    result = [None] * 101
    for r in range(101):
        mask = best_matchL[r]
        if mask != -1:
            result[r] = mask_to_terms[mask]
        else:
            # Fallback: find any subset for this residue
            # This should not happen if matching was perfect
            found = False
            for size, residue, mask in subset_data:
                if residue == r:
                    result[r] = mask_to_terms[mask]
                    found = True
                    break
            if not found:
                result[r] = [1] # Arbitrary fallback

    return result
# EVOLVE-BLOCK-END
