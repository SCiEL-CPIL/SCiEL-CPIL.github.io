import time
from collections import defaultdict

def construct():
    start_time = time.time()
    
    # Precompute inverses of 1..10 modulo 101
    p = 101
    invs = [pow(x, -1, p) for x in range(1, 11)]
    denominators = list(range(1, 11))
    
    # Precompute all non-empty subsets: (sum_mod_p, size, denom_list)
    # We store them in a list for easy iteration
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
        
    # Sort subsets by size to facilitate binary search or linear scan
    # We will try K=1, K=2, etc.
    
    # Hopcroft-Karp implementation
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] contains right nodes connected to left node i
        n_left: number of left nodes
        n_right: number of right nodes
        Returns matching size
        """
        from collections import deque
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

    # We want to minimize the maximum size K.
    # Try K = 1, 2, ...
    # For each K, build the bipartite graph and check for perfect matching.
    
    # To be safe with time, we can just iterate K from 1 to 10.
    # The number of subsets is small, so this is fast.
    
    best_K = 10
    best_matching = None
    
    for K in range(1, 11):
        # Filter subsets with size <= K
        valid_subsets = [(s, sz, dl) for s, sz, dl in all_subsets if sz <= K]
        
        # Build adjacency list for residues (left) to subsets (right)
        # Left nodes: 0..100 (residues)
        # Right nodes: 0..len(valid_subsets)-1 (indices in valid_subsets)
        
        # Map residue to list of subset indices
        residue_to_subsets = defaultdict(list)
        for idx, (s, sz, dl) in enumerate(valid_subsets):
            residue_to_subsets[s].append(idx)
            
        adj = [[] for _ in range(101)]
        for r in range(101):
            adj[r] = residue_to_subsets[r]
            
        n_left = 101
        n_right = len(valid_subsets)
        
        if n_right < n_left:
            continue
            
        matching_size, pair_u = hopcroft_karp(adj, n_left, n_right)
        
        if matching_size == n_left:
            best_K = K
            # Reconstruct the assignment
            # pair_u[r] is the index in valid_subsets for residue r
            result = []
            for r in range(101):
                idx = pair_u[r]
                result.append(valid_subsets[idx][2]) # denom_list
            best_matching = result
            break
            
    if best_matching is None:
        # Fallback: should not happen, but just in case
        # Use greedy or any valid assignment
        # Since K=10 includes all subsets, and there are 1023 > 101, 
        # a matching must exist.
        # Let's force K=10 logic again if it failed due to bug, 
        # or just return a valid construction.
        # Actually, if no matching found, we can just pick any subset for each residue.
        # But Hopcroft-Karp should find it.
        # Let's try a simple greedy fallback for safety.
        result = []
        used = [False] * len(all_subsets)
        # Sort subsets by size
        sorted_subsets = sorted(all_subsets, key=lambda x: x[1])
        for r in range(101):
            found = False
            for idx, (s, sz, dl) in enumerate(sorted_subsets):
                if s == r and not used[idx]:
                    result.append(dl)
                    used[idx] = True
                    found = True
                    break
            if not found:
                # This shouldn't happen
                result.append(denominators)
        best_matching = result

    return best_matching
