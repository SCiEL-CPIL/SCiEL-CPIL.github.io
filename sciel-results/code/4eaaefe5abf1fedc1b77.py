def construct():
    import time
    P = 101
    N = 10
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s margin
    
    # Precompute inverses
    invs = [0] * (N + 1)
    for i in range(1, N + 1):
        invs[i] = pow(i, -1, P)
    
    # Generate all non-empty subsets with their sum and size
    subsets = []  # list of (sum_mod_P, size, subset_list)
    for mask in range(1, 1 << N):
        s = 0
        size = 0
        subset = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                s = (s + invs[i]) % P
                size += 1
                subset.append(i)
        subsets.append((s, size, subset))
    
    # For each residue, store list of (size, subset_index)
    # subsets_by_residue[r] = list of (size, idx)
    subsets_by_residue = [[] for _ in range(P)]
    for idx, (s, size, subset) in enumerate(subsets):
        subsets_by_residue[s].append((size, idx))
    
    # Sort each residue's options by size
    for r in range(P):
        subsets_by_residue[r].sort()
    
    # Check if a perfect matching exists using only subsets of size <= k
    def can_match(k):
        # Build bipartite graph: residues -> subsets of size <= k
        # Use Hopcroft-Karp
        # Left: residues 0..P-1, Right: subset indices 0..len(subsets)-1
        # But we only connect to subsets with size <= k
        
        # Adjacency: for each residue, list of subset indices with size <= k
        adj = [[] for _ in range(P)]
        for r in range(P):
            for size, idx in subsets_by_residue[r]:
                if size <= k:
                    adj[r].append(idx)
                else:
                    break  # sorted by size
        
        # Check if all residues have at least one option
        for r in range(P):
            if not adj[r]:
                return False, None
        
        # Hopcroft-Karp
        from collections import deque
        
        # match_left[r] = subset index matched to residue r, or -1
        # match_right[idx] = residue matched to subset idx, or -1
        match_left = [-1] * P
        match_right = [-1] * len(subsets)
        dist = [0] * P
        
        def bfs():
            # BFS to find shortest augmenting paths
            q = deque()
            for r in range(P):
                if match_left[r] == -1:
                    dist[r] = 0
                    q.append(r)
                else:
                    dist[r] = float('inf')
            
            found = False
            while q:
                r = q.popleft()
                for idx in adj[r]:
                    mr = match_right[idx]
                    if mr == -1:
                        found = True
                    elif dist[mr] == float('inf'):
                        dist[mr] = dist[r] + 1
                        q.append(mr)
            return found
        
        def dfs(r):
            for idx in adj[r]:
                mr = match_right[idx]
                if mr == -1 or (dist[mr] == dist[r] + 1 and dfs(mr)):
                    match_left[r] = idx
                    match_right[idx] = r
                    return True
            dist[r] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for r in range(P):
                if match_left[r] == -1:
                    if dfs(r):
                        matching_size += 1
        
        if matching_size < P:
            return False, None
        
        # Reconstruct assignment
        assignment = [None] * P
        for r in range(P):
            idx = match_left[r]
            assignment[r] = subsets[idx][2]  # the subset list
        
        return True, assignment
    
    # Binary search on k
    best_k = N + 1
    best_assignment = None
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        found, assignment = can_match(k)
        if found:
            best_k = k
            best_assignment = assignment
            break
    
    # If we found an assignment, return it
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: use the greedy approach (smallest subset per residue)
    result = []
    for r in range(P):
        if subsets_by_residue[r]:
            # Take the smallest subset
            _, idx = subsets_by_residue[r][0]
            result.append(subsets[idx][2])
        else:
            # Should not happen, but handle gracefully
            result.append([1])
    
    return result
