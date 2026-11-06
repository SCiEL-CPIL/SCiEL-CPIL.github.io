"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    # Also precompute terms for each mask for fast lookup
    mask_to_terms = {}
    residue_to_masks = [[] for _ in range(101)]
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x - 1] for x in terms) % 101
        mask_to_terms[mask] = terms
        residue_to_masks[residue].append((size, mask))
    
    # Sort each residue's list by size (smaller first)
    for r in range(101):
        residue_to_masks[r].sort()
    
    def hopcroft_karp(adj, n_left=101):
        """Run Hopcroft-Karp. adj[r] is list of right-node IDs (mask values).
        Returns matchL array where matchL[r] = matched mask, or -1 if unmatched."""
        matchL = [-1] * n_left
        # matchR: use a list indexed by mask (masks are 1..1023)
        matchR = [-1] * 1024  # matchR[mask] = residue or -1
        
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
            if time.time() > deadline - 3:
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
        """Check if perfect matching exists with all subset sizes <= K."""
        adj = [[] for _ in range(101)]
        for r in range(101):
            for size, mask in residue_to_masks[r]:
                if size <= K:
                    adj[r].append(mask)
                else:
                    break
        _, matching_size = hopcroft_karp(adj)
        return matching_size == 101
    
    # Binary search for minimum K
    lo, hi = 1, 10
    best_K = 10
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        if can_match(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Construct the actual assignment using best_K
    adj = [[] for _ in range(101)]
    for r in range(101):
        for size, mask in residue_to_masks[r]:
            if size <= best_K:
                adj[r].append(mask)
            else:
                break
    
    matchL, matching_size = hopcroft_karp(adj)
    
    # Build result
    result = [None] * 101
    for r in range(101):
        mask = matchL[r]
        if mask is not None and mask != -1:
            result[r] = mask_to_terms[mask]
        else:
            # Fallback: use any subset for this residue
            if residue_to_masks[r]:
                size, mask = residue_to_masks[r][0]
                result[r] = mask_to_terms[mask]
            else:
                result[r] = [1]
    
    # Verify all residues are covered
    for r in range(101):
        if result[r] is None or len(result[r]) == 0:
            result[r] = [1]
    
    return result
# EVOLVE-BLOCK-END
