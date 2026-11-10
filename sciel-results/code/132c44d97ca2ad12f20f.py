"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict, deque
import math

def construct():
    p = 101
    # Precompute modular inverses for denominators 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Generate all non-empty subsets of {1..10} and their inverse sums
    # subset represented as bitmask, size = popcount
    all_subsets = []  # list of (mask, size, residue)
    
    for mask in range(1, 1 << 10):
        size = 0
        residue = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                size += 1
                residue = (residue + inv[x]) % p
        all_subsets.append((mask, size, residue))
        
    # Sort subsets by size to easily access subsets of size <= k
    all_subsets.sort(key=lambda x: x[1])
    
    # Precompute prefix counts: count_subsets_leq[k] = number of subsets with size <= k
    # Since max size is 10, we can just keep a running count or use binary search
    # But since we iterate k from 1 to 10, we can just find the cutoff index
    
    # Hopcroft-Karp implementation
    def hopcroft_karp(n_left, n_right, adj):
        """
        n_left: number of left nodes (0..n_left-1)
        n_right: number of right nodes (0..n_right-1)
        adj: list of lists, adj[i] = list of right nodes connected to left node i
        Returns (matching_size, pair_u, pair_v)
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u, pair_v

    # Try k from 1 to 10
    # We need to find the smallest k such that there are at least 101 subsets of size <= k
    # and a perfect matching exists.
    
    # Precompute for each k, the list of subset indices with size <= k
    # Since all_subsets is sorted by size, we can just take a prefix.
    # Let's find the index where size > k for each k.
    # Actually, we can just iterate k and expand the right side.
    
    # For efficiency, we'll build the adjacency list for each k.
    # adj[r] = list of right-node indices (which are just indices in the prefix of all_subsets)
    
    for k in range(1, 11):
        # Find how many subsets have size <= k
        # Since all_subsets is sorted by size, we can binary search or just count.
        # Max size is 10, so linear scan is fine.
        n_right = 0
        for i, (mask, size, residue) in enumerate(all_subsets):
            if size <= k:
                n_right = i + 1
            else:
                break
        
        if n_right < 101:
            continue
            
        # Build adjacency list for the first n_right subsets
        adj = [[] for _ in range(p)]
        for i in range(n_right):
            mask, size, residue = all_subsets[i]
            adj[residue].append(i)
            
        # Check if every residue has at least one subset
        valid = True
        for r in range(p):
            if not adj[r]:
                valid = False
                break
        if not valid:
            continue
            
        # Run Hopcroft-Karp
        matching_size, pair_u, pair_v = hopcroft_karp(p, n_right, adj)
        
        if matching_size == p:
            # Construct result
            result = []
            for r in range(p):
                right_idx = pair_u[r]
                mask = all_subsets[right_idx][0]
                terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
                result.append(terms)
            return result
            
    # Fallback (should not happen)
    representations = {}
    for mask, size, residue in all_subsets:
        if residue not in representations:
            terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            representations[residue] = terms
    return [representations[r] for r in range(p)]
# EVOLVE-BLOCK-END
