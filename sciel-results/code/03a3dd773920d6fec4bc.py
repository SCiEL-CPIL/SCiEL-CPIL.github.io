import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each subset, compute its residue
    # subsets_by_residue[r] = list of subsets (as tuples) that sum to r
    subsets_by_residue = [[] for _ in range(101)]
    
    # Generate all non-empty subsets and their residues
    all_subsets = []
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            subsets_by_residue[residue].append(combo)
            all_subsets.append((combo, residue))
    
    # For each k, check if there's a perfect matching using subsets of size <= k
    # Use Hopcroft-Karp for bipartite matching
    
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] = list of right nodes adjacent to left node i
        n_left: number of left nodes
        n_right: number of right nodes
        Returns: matching as dict left_node -> right_node, or None if no perfect matching
        """
        import collections
        
        pair_u = [-1] * n_left   # pair_u[u] = v matched to u
        pair_v = [-1] * n_right  # pair_v[v] = u matched to v
        dist = [0] * n_left
        
        def bfs():
            queue = collections.deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        if matching == n_left:
            return pair_u
        return None
    
    # For each k from 1 to 10, try to find perfect matching
    for k in range(1, 11):
        if time.time() > deadline:
            break
        
        # Build adjacency: for each residue, which subsets of size <= k map to it
        # Left side: 101 residues (0..100)
        # Right side: all subsets of size <= k
        
        # Collect all subsets of size <= k
        subsets_k = []
        for s in range(1, k + 1):
            for combo in combinations(range(1, 11), s):
                subsets_k.append(combo)
        
        n_right = len(subsets_k)
        
        # Build adjacency list for left nodes (residues)
        # adj[r] = list of indices of subsets that sum to residue r
        adj = [[] for _ in range(101)]
        for idx, combo in enumerate(subsets_k):
            r = sum(inv[x] for x in combo) % 101
            adj[r].append(idx)
        
        # Check if every residue has at least one subset
        if any(len(adj[r]) == 0 for r in range(101)):
            continue
        
        # Run Hopcroft-Karp
        matching = hopcroft_karp(adj, 101, n_right)
        
        if matching is not None:
            # Build result
            result = []
            for r in range(101):
                subset_idx = matching[r]
                result.append(list(subsets_k[subset_idx]))
            
            # Verify correctness
            for r in range(101):
                row = result[r]
                assert len(row) > 0, f"Row {r} is empty"
                assert len(set(row)) == len(row), f"Row {r} has duplicates"
                assert all(1 <= x <= 10 for x in row), f"Row {r} has out-of-range values"
                computed = sum(inv[x] for x in row) % 101
                assert computed == r, f"Row {r} sums to {computed}, expected {r}"
            
            return result
    
    # Fallback: use greedy approach (should not be reached)
    residues_by_size = [dict() for _ in range(11)]
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            if residue not in residues_by_size[s]:
                residues_by_size[s][residue] = list(combo)
    
    result = []
    for r in range(101):
        for s in range(1, 11):
            if r in residues_by_size[s]:
                result.append(residues_by_size[s][r])
                break
        else:
            result.append(list(range(1, 11)))
    
    return result
