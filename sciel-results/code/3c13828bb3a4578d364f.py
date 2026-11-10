import time
from itertools import combinations
import collections

def construct():
    deadline = time.time() + 80
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all subsets and their residues, sorted by size
    # subsets_all[i] = (combo, residue)
    subsets_all = []
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            subsets_all.append((combo, residue))
            
    # subsets_all is already sorted by size because we iterate s from 1 to 10
    
    def hopcroft_karp(adj, n_left, n_right):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
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

    # Count how many subsets are of each size
    # We need to know the index in subsets_all where size k ends
    # Since subsets_all is sorted by size, we can just use a pointer or precomputed counts
    # Let's precompute the end index for each k
    end_idx_for_k = {}
    idx = 0
    for s in range(1, 11):
        count = len(list(combinations(range(1, 11), s))) # This is inefficient, better to just accumulate
        # Actually, we can just iterate subsets_all and track size
        pass

    # Simpler: just build adjacency for each k using the precomputed list
    # But we need to know which subsets are of size <= k
    # Since subsets_all is sorted by size, the first N_k subsets are of size <= k
    # We can precompute N_k
    N_k = [0] * 11
    current_idx = 0
    for s in range(1, 11):
        for _ in combinations(range(1, 11), s):
            current_idx += 1
        N_k[s] = current_idx

    for k in range(1, 11):
        if time.time() > deadline:
            break
        
        n_right = N_k[k]
        
        # Build adjacency list for left nodes (residues 0..100)
        adj = [[] for _ in range(101)]
        for idx in range(n_right):
            combo, residue = subsets_all[idx]
            adj[residue].append(idx)
            
        # Check if every residue has at least one subset
        if any(len(adj[r]) == 0 for r in range(101)):
            continue
        
        # Run Hopcroft-Karp
        matching = hopcroft_karp(adj, 101, n_right)
        
        if matching is not None:
            result = []
            for r in range(101):
                subset_idx = matching[r]
                combo, _ = subsets_all[subset_idx]
                result.append(list(combo))
            
            # Verify correctness
            for r in range(101):
                row = result[r]
                assert len(row) > 0, f"Row {r} is empty"
                assert len(set(row)) == len(row), f"Row {r} has duplicates"
                assert all(1 <= x <= 10 for x in row), f"Row {r} has out-of-range values"
                computed = sum(inv[x] for x in row) % 101
                assert computed == r, f"Row {r} sums to {computed}, expected {r}"
            
            return result
    
    # Fallback
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
