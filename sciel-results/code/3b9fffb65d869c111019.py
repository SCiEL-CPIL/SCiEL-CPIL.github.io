import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    P = 101
    # Precompute inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, P - 2, P)
    
    # Generate all non-empty subsets of {1..10}, compute their residue and size
    # Store as: (residue, size, subset_list)
    subsets = []
    subset_residues = []  # residue for each mask
    subset_sizes = []     # size for each mask
    subset_lists = []     # list of denominators for each mask
    
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        sz = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
                sz += 1
        r = s % P
        subsets.append((r, sz, subset))
        subset_residues.append(r)
        subset_sizes.append(sz)
        subset_lists.append(subset)
    
    # Precompute: for each residue, list of (size, subset_idx) sorted by size
    subsets_by_residue = defaultdict(list)
    for idx in range(len(subsets)):
        r = subset_residues[idx]
        sz = subset_sizes[idx]
        subsets_by_residue[r].append((sz, idx))
    
    # Sort each residue's list by size
    for r in subsets_by_residue:
        subsets_by_residue[r].sort(key=lambda x: x[0])
    
    # Hopcroft-Karp for bipartite matching with current-edge optimization
    def hopcroft_karp(adj, n_left, n_right):
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
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if matchR[v] == -1:
                        found = True
                    elif dist[matchR[v]] == -1:
                        dist[matchR[v]] = dist[u] + 1
                        queue.append(matchR[v])
            return found
        
        def dfs(u, cur):
            for i in range(cur[u], len(adj[u])):
                cur[u] = i
                v = adj[u][i]
                if matchR[v] == -1 or (dist[matchR[v]] == dist[u] + 1 and dfs(matchR[v], cur)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching_size = 0
        while bfs():
            cur = [0] * n_left
            for u in range(n_left):
                if matchL[u] == -1:
                    if dfs(u, cur):
                        matching_size += 1
        
        if matching_size == n_left:
            return matchL
        return None
    
    # Try max_size from 1 to 10
    best_result = None
    for max_size in range(1, 11):
        if time.time() > deadline:
            break
        
        # Build adjacency: left nodes are residues 0..100, right nodes are subset indices
        # Only include subsets with size <= max_size
        adj = [[] for _ in range(P)]
        right_indices = []  # list of subset indices that are valid
        subset_to_right = {}  # subset_idx -> right_idx
        
        for r in range(P):
            for sz, idx in subsets_by_residue.get(r, []):
                if sz > max_size:
                    break  # sorted by size, so we can break
                if idx not in subset_to_right:
                    subset_to_right[idx] = len(right_indices)
                    right_indices.append(idx)
                adj[r].append(subset_to_right[idx])
        
        n_right = len(right_indices)
        
        if n_right < P:
            # Not enough right nodes, skip
            continue
        
        # Run Hopcroft-Karp
        matchL = hopcroft_karp(adj, P, n_right)
        
        if matchL is not None:
            # Perfect matching found! Build the result
            result = [None] * P
            for r in range(P):
                ridx = matchL[r]
                subset_idx = right_indices[ridx]
                result[r] = subset_lists[subset_idx]
            
            best_result = result
            break
    
    if best_result is None:
        # Fallback: greedy approach
        best = [None] * P
        for idx in range(len(subsets)):
            r = subset_residues[idx]
            sz = subset_sizes[idx]
            if best[r] is None or sz < best[r][0]:
                best[r] = (sz, idx)
        for r in range(P):
            if best[r] is None:
                best[r] = (1, 0)
        best_result = [subset_lists[best[r][1]] for r in range(P)]
    
    return best_result
# EVOLVE-BLOCK-END
