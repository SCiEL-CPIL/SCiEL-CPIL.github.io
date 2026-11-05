import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    P = 101
    # Precompute inverses
    inv = {x: pow(x, -1, P) for x in range(1, 11)}
    
    # Generate all non-empty subsets of {1..10}, compute their residue and size
    # subsets: list of (residue, size, subset_list)
    subsets = []
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        subsets.append((r, sz, subset))
    
    # Group subsets by size
    subsets_by_size = defaultdict(list)
    for r, sz, subset in subsets:
        subsets_by_size[sz].append((r, subset))
    
    # For a given max_size k, build bipartite graph:
    # Left: residues 0..100
    # Right: subset indices (only those with size <= k)
    # Edge: residue r connected to subset i if subset i has residue r and size <= k
    # We want a perfect matching (all 101 residues matched)
    
    # Precompute: for each max_size k, which subsets are available
    # Since there are only 10 possible sizes, we can precompute adjacency lists
    
    # For each size k, list of (residue, subset_index_in_all_subsets)
    # Actually, let's just use the subsets list directly
    
    # Hopcroft-Karp for bipartite matching
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[left_node] = list of right_nodes
        n_left: number of left nodes (0..n_left-1)
        n_right: number of right nodes (0..n_right-1)
        Returns: matching as dict left->right, or None if no perfect matching
        """
        # matchL[i] = matched right node for left node i, or -1
        # matchR[j] = matched left node for right node j, or -1
        matchL = [-1] * n_left
        matchR = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if matchL[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if matchR[v] == -1:
                        found = True
                    elif dist[matchR[v]] == float('inf'):
                        dist[matchR[v]] = dist[u] + 1
                        queue.append(matchR[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if matchR[v] == -1 or (dist[matchR[v]] == dist[u] + 1 and dfs(matchR[v])):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for u in range(n_left):
                if matchL[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        if matching_size == n_left:
            return matchL
        return None
    
    # Try max_size from 1 to 10
    best_result = None
    for max_size in range(1, 11):
        if time.time() > deadline:
            break
        
        # Build adjacency: for each residue, which subsets (by index) can cover it with size <= max_size
        # We need to index subsets properly
        # Let's create a list of subset indices grouped by residue, filtered by size
        
        # adj[residue] = list of subset indices
        adj = [[] for _ in range(P)]
        right_nodes = []  # list of (residue, subset) for all available subsets
        
        for idx, (r, sz, subset) in enumerate(subsets):
            if sz <= max_size:
                adj[r].append(idx)
        
        # The right side is just the subset indices that are available
        # n_right = number of available subsets
        n_right = sum(len(adj[r]) for r in range(P))
        # Actually, each subset index is unique, so n_right = count of subsets with size <= max_size
        n_right = sum(1 for (r, sz, subset) in subsets if sz <= max_size)
        
        # Hopcroft-Karp: left = residues (0..100), right = subset indices
        # But subset indices go up to 1022, so we need to remap
        # Let's create a mapping from subset index to right node index
        right_map = {}
        right_idx = 0
        for idx, (r, sz, subset) in enumerate(subsets):
            if sz <= max_size:
                right_map[idx] = right_idx
                right_idx += 1
        
        # Rebuild adj with right node indices
        adj_mapped = [[] for _ in range(P)]
        for r in range(P):
            for idx in adj[r]:
                adj_mapped[r].append(right_map[idx])
        
        n_right = right_idx
        
        # Run Hopcroft-Karp
        matchL = hopcroft_karp(adj_mapped, P, n_right)
        
        if matchL is not None:
            # Perfect matching found! Build the result
            result = [None] * P
            for r in range(P):
                subset_idx = matchL[r]
                # Find the actual subset
                # We need to reverse the right_map
                # Let's just iterate subsets to find which one has this right index
                # Better: build reverse map
                pass
            
            # Build reverse map
            rev_right_map = {}
            for idx, ridx in right_map.items():
                rev_right_map[ridx] = idx
            
            for r in range(P):
                ridx = matchL[r]
                subset_idx = rev_right_map[ridx]
                result[r] = subsets[subset_idx][2]
            
            best_result = result
            break
    
    if best_result is None:
        # Fallback: use the greedy minimum-size approach
        best = [None] * P
        for mask in range(1, 1 << 10):
            subset = []
            s = 0
            for x in range(1, 11):
                if mask >> (x - 1) & 1:
                    subset.append(x)
                    s += inv[x]
            r = s % P
            sz = len(subset)
            if best[r] is None or sz < best[r][0]:
                best[r] = (sz, subset)
        # Ensure all residues covered
        for r in range(P):
            if best[r] is None:
                for mask in range(1, 1 << 10):
                    subset = []
                    s = 0
                    for x in range(1, 11):
                        if mask >> (x - 1) & 1:
                            subset.append(x)
                            s += inv[x]
                    if s % P == r:
                        best[r] = (len(subset), subset)
                        break
                if best[r] is None:
                    best[r] = (1, [1])
        best_result = [best[r][1] for r in range(P)]
    
    return best_result
# EVOLVE-BLOCK-END
