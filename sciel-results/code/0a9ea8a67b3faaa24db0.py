import time
from itertools import combinations
from collections import deque

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Inverses of 1..10 mod 101
    invs = [0] * 11
    for i in range(1, 11):
        invs[i] = pow(i, -1, 101)
    
    # Precompute all non-empty subsets: list of (size, residue, terms_tuple)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = tuple(x for x in range(1, 11) if (mask >> (x - 1)) & 1)
        r = 0
        for x in terms:
            r += invs[x]
        r %= 101
        all_subsets.append((len(terms), r, terms))
    
    # Group subsets by size: by_size[k] = list of (residue, terms)
    by_size = [[] for _ in range(11)]
    for size, r, terms in all_subsets:
        by_size[size].append((r, terms))
    
    # Hopcroft-Karp: left side = residues 0..100, right side = subset indices
    # For a given K, build adjacency: residue r -> list of subset indices with size <= K and residue == r
    # We need a perfect matching (all 101 residues matched)
    
    def hopcroft_karp(n_left, adj):
        """
        n_left: number of left nodes (residues)
        adj: list of lists, adj[i] = list of right node indices adjacent to left node i
        Returns matching: match_left[i] = right node matched to left node i, or -1
        """
        INF = float('inf')
        match_left = [-1] * n_left
        match_right = [-1] * len(adj[0]) if adj else []
        
        # Actually, right side size varies. Let's track it differently.
        # Right nodes are subset indices. We'll use a dict or track max right index.
        # Better: pass right_size as parameter or infer.
        
        # Let me restructure: adj is list of sets/lists of right node ids
        # match_left[i] = right_id or -1
        # match_right[j] = left_id or -1
        
        n_right = 0
        for neighbors in adj:
            for nb in neighbors:
                if nb >= n_right:
                    n_right = nb + 1
        
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        
        def bfs():
            """Returns True if there's an augmenting path"""
            dist = [INF] * n_left
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == INF:
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found, dist
        
        def dfs(u, dist):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v], dist)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = INF
            return False
        
        matching_size = 0
        while True:
            found, dist = bfs()
            if not found:
                break
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u, dist):
                        matching_size += 1
        return matching_size, match_left
    
    # Binary search on K
    # For each K, build adjacency: for each residue r, list of subset indices (in the filtered set) that have residue r and size <= K
    
    # Precompute: for each K, which subsets are available
    # subsets_by_residue_and_size: residue -> size -> list of (subset_index, terms)
    # Actually, let's just build adj for each K on the fly
    
    # To make this efficient, precompute for each subset its size and residue
    # subset_info[i] = (size, residue) for subset i
    subset_info = [(s, r) for s, r, t in all_subsets]
    n_subsets = len(all_subsets)
    
    # For binary search on K in [1, 10]:
    # Check if perfect matching exists with subsets of size <= K
    
    def can_cover(K):
        """Returns (bool, match_left) where match_left[r] = subset_index for residue r, or None if not possible"""
        # Build adjacency: for each residue r (0..100), list of subset indices with size <= K and residue == r
        adj = [[] for _ in range(101)]
        for idx in range(n_subsets):
            s, r = subset_info[idx]
            if s <= K:
                adj[r].append(idx)
        
        # Check if any residue has no subsets
        for r in range(101):
            if not adj[r]:
                return False, None
        
        matching_size, match_left = hopcroft_karp(101, adj)
        if matching_size == 101:
            return True, match_left
        return False, None
    
    # Try K from 1 upward
    best_K = 10
    best_match = None
    
    for K in range(1, 11):
        if time.time() > deadline - 2:
            break
        possible, match_left = can_cover(K)
        if possible:
            best_K = K
            best_match = match_left
            break
    
    # Construct result
    if best_match is not None:
        result = []
        for r in range(101):
            idx = best_match[r]
            result.append(list(all_subsets[idx][2]))
        return result
    else:
        # Fallback: greedy assignment (should not happen)
        coverage = {}
        for K in range(1, 11):
            for s, r, terms in all_subsets:
                if s == K and r not in coverage:
                    coverage[r] = terms
            if len(coverage) == 101:
                break
        result = []
        for r in range(101):
            if r in coverage:
                result.append(list(coverage[r]))
            else:
                # Find any subset
                found = False
                for idx in range(n_subsets):
                    if subset_info[idx][1] == r:
                        result.append(list(all_subsets[idx][2]))
                        found = True
                        break
                if not found:
                    result.append([1])
        return result
