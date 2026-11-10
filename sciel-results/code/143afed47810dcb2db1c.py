import time
from collections import deque

def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0
    
    P = 101
    N = 10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
    
    # Precompute all non-empty subsets: mask -> (length, sum_mod, denoms_list)
    num_subsets = (1 << N) - 1
    subset_sum = [0] * (1 << N)
    subset_len = [0] * (1 << N)
    subset_denoms = [None] * (1 << N)
    
    for mask in range(1, 1 << N):
        length = 0
        s = 0
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                denoms.append(d)
                s = (s + inv[d]) % P
                length += 1
        subset_sum[mask] = s
        subset_len[mask] = length
        subset_denoms[mask] = denoms
    
    # For each residue, collect masks that achieve it, sorted by length
    masks_by_residue = [[] for _ in range(P)]
    for mask in range(1, 1 << N):
        masks_by_residue[subset_sum[mask]].append(mask)
    for r in range(P):
        masks_by_residue[r].sort(key=lambda m: subset_len[m])
    
    def find_matching(k):
        """Find perfect matching using only subsets of size <= k via Hopcroft-Karp."""
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        # Build adjacency: residue r -> list of mask indices (0-based) with len <= k
        adj = [[] for _ in range(P)]
        for r in range(P):
            for mask in masks_by_residue[r]:
                if subset_len[mask] <= k:
                    adj[r].append(mask - 1)  # 0-based index
        
        pair_u = [-1] * P
        pair_v = [-1] * num_subsets
        dist = [0] * P
        INF = float('inf')
        
        def bfs():
            queue = deque()
            for u in range(P):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == INF:
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False
        
        matching = 0
        while bfs():
            for u in range(P):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        if matching != P:
            return None
        
        assignment = [None] * P
        for u in range(P):
            v = pair_u[u]
            mask = v + 1
            assignment[u] = subset_denoms[mask]
        
        return assignment
    
    # Try k=1,2,...,N and find the smallest k with a perfect matching
    best_assignment = None
    for k in range(1, N + 1):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        result = find_matching(k)
        if result is not None:
            best_assignment = result
            break
    
    if best_assignment is None:
        # Fallback: for each residue, pick the shortest subset that achieves it
        best_assignment = [None] * P
        for r in range(P):
            if masks_by_residue[r]:
                best_mask = masks_by_residue[r][0]
                best_assignment[r] = subset_denoms[best_mask]
            else:
                # This should never happen since all residues are covered by some subset
                best_assignment[r] = [1]
    
    return best_assignment
