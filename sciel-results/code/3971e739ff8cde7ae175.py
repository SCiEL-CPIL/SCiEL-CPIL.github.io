import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    p = 101
    denominators = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denominators]
    
    # Precompute all non-empty subsets: (sum_mod_p, size, denom_list)
    all_subsets = []
    for mask in range(1, 1 << 10):
        denom_list = []
        s = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                denom_list.append(denominators[i])
                s = (s + invs[i]) % p
                size += 1
        all_subsets.append((s, size, denom_list))
        
    # Hopcroft-Karp implementation
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] contains right nodes connected to left node i
        n_left: number of left nodes (residues)
        n_right: number of right nodes (subsets)
        Returns matching size and pair_u (assignment for left nodes)
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
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
        return matching, pair_u

    n_left = 101
    
    # We know K must be at least 3.
    # Subsets of size 1: 10
    # Subsets of size 2: 45
    # Total <= 2: 55 < 101. Impossible.
    # So we start checking from K=3.
    
    best_result = None
    
    for K in range(3, 11):
        if time.time() > deadline:
            break
            
        # Filter subsets with size <= K
        valid_subsets = [(s, sz, dl) for s, sz, dl in all_subsets if sz <= K]
        
        # If we don't have enough subsets to cover all residues, skip
        if len(valid_subsets) < n_left:
            continue
            
        # Build adjacency list for residues (left) to subsets (right)
        # Left nodes: 0..100 (residues)
        # Right nodes: 0..len(valid_subsets)-1
        
        # Map residue to list of subset indices
        residue_to_subsets = [[] for _ in range(n_left)]
        for idx, (s, sz, dl) in enumerate(valid_subsets):
            residue_to_subsets[s].append(idx)
            
        adj = residue_to_subsets
        n_right = len(valid_subsets)
        
        matching_size, pair_u = hopcroft_karp(adj, n_left, n_right)
        
        if matching_size == n_left:
            # Perfect matching found for this K
            # Reconstruct the result
            result = []
            for r in range(n_left):
                idx = pair_u[r]
                # valid_subsets[idx] is (s, sz, dl)
                result.append(valid_subsets[idx][2])
            best_result = result
            break
            
    if best_result is None:
        # Fallback: This should not happen as K=10 includes all subsets.
        # But just in case, return a valid (though not minimal) construction.
        valid_subsets = all_subsets
        residue_to_subsets = [[] for _ in range(n_left)]
        for idx, (s, sz, dl) in enumerate(valid_subsets):
            residue_to_subsets[s].append(idx)
            
        adj = residue_to_subsets
        n_right = len(valid_subsets)
        matching_size, pair_u = hopcroft_karp(adj, n_left, n_right)
        
        if matching_size == n_left:
            result = []
            for r in range(n_left):
                idx = pair_u[r]
                result.append(valid_subsets[idx][2])
            return result
        else:
            # Extreme fallback
            return [[1] for _ in range(101)]

    return best_result
