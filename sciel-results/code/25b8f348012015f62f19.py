import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask representation of adjacency
    # adj[v] is a bitmask of neighbors of v
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Complement adjacency
    comp_adj = [0] * n
    for v in range(n):
        # All bits set except self and adj[v]
        full_mask = (1 << n) - 1
        comp_adj[v] = full_mask ^ (1 << v) ^ adj[v]

    def bron_kerbosch(R, P_set, X, adj_list, result):
        if not P_set and not X:
            result.append(R)
            return
        # Choose pivot u from P_set U X to minimize branching
        # Simple heuristic: pick first one
        pivot_candidates = P_set | X
        if pivot_candidates:
            # Find an index
            u = -1
            temp = pivot_candidates
            while temp:
                lsb = temp & -temp
                u = (lsb.bit_length() - 1)
                break
            # Branch on vertices in P_set \ N(u)
            candidates = P_set & ~adj_list[u]
        else:
            candidates = P_set
            
        # Iterate through candidates
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            new_R = R | (1 << v)
            new_P = P_set & adj_list[v]
            new_X = X & adj_list[v]
            bron_kerbosch(new_R, new_P, new_X, adj_list, result)
            P_set ^= lsb
            X |= lsb

    # Find maximal cliques in G
    max_cliques_G = []
    bron_kerbosch(0, (1 << n) - 1, 0, adj, max_cliques_G)
    
    # Find maximal cliques in G_bar (which are maximal independent sets in G)
    max_independent_G = []
    bron_kerbosch(0, (1 << n) - 1, 0, comp_adj, max_independent_G)
    
    # Combine candidate sets
    candidates = []
    for mask in max_cliques_G:
        candidates.append(mask)
    for mask in max_independent_G:
        candidates.append(mask)
        
    # Remove duplicates (in case a set is both a clique and independent set, e.g., single vertex or empty, though maximal usually prevents duplicates unless graph is special)
    candidates = list(set(candidates))
    
    # Sort candidates by size (popcount) descending
    def popcount(x):
        return bin(x).count('1')
        
    candidates.sort(key=popcount, reverse=True)
    
    # Precompute max_valid_size for all subsets
    # max_valid_size[mask] = max size of a candidate set contained in mask
    max_valid_size = [0] * (1 << n)
    # For each candidate, update all its supersets
    for c in candidates:
        size_c = popcount(c)
        # Iterate through all supersets of c
        # Supersets of c are c | s where s is a subset of (~c)
        # More efficiently:
        # Let remaining_bits = full_mask ^ c
        # Iterate submasks of remaining_bits
        sub = (1 << n) - 1 ^ c
        # We need to iterate all subsets of (full_mask ^ c)
        # Standard submask iteration:
        s = sub
        while True:
            sup = c | s
            if size_c > max_valid_size[sup]:
                max_valid_size[sup] = size_c
            if s == 0:
                break
            s = (s - 1) & sub

    # DFS to find minimum number of parts
    best = n + 1
    full_mask = (1 << n) - 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe within 60s
    
    # Memoization or just global best?
    # Global best is sufficient with good pruning.
    
    def dfs(remaining, current_count):
        nonlocal best
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if current_count < best:
                best = current_count
            return
            
        # Pruning: Lower bound
        # LB = ceil(popcount(remaining) / max_valid_size[remaining])
        # Note: max_valid_size[remaining] is the largest valid set *contained* in remaining.
        # If remaining is non-empty, max_valid_size[remaining] >= 1.
        lb = math.ceil(popcount(remaining) / max_valid_size[remaining])
        if current_count + lb >= best:
            return
            
        # Try each candidate
        for c in candidates:
            if (c & remaining) == c:
                # c is a subset of remaining
                dfs(remaining & ~c, current_count + 1)
                if time.time() > deadline:
                    return

    dfs(full_mask, 0)
    
    # Reconstruct the partition for the best count?
    # The task asks for the partition, not just the number.
    # We need to store the path that led to 'best'.
    
    # Let's modify DFS to store the solution
    best_solution = None
    
    def dfs_with_path(remaining, current_count, path):
        nonlocal best, best_solution
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if current_count < best:
                best = current_count
                best_solution = path[:]
            return
            
        lb = math.ceil(popcount(remaining) / max_valid_size[remaining])
        if current_count + lb >= best:
            return
            
        for c in candidates:
            if (c & remaining) == c:
                path.append(c)
                dfs_with_path(remaining & ~c, current_count + 1, path)
                path.pop()
                if time.time() > deadline:
                    return

    dfs_with_path(full_mask, 0, [])
    
    if best_solution is None:
        # Fallback: greedy or singletons
        # This shouldn't happen as n singletons is always a valid solution
        best_solution = [1 << i for i in range(n)]
        
    # Convert masks back to lists of vertices
    partition = []
    for mask in best_solution:
        part = []
        for v in range(n):
            if mask & (1 << v):
                part.append(v)
        if part: # Ensure non-empty
            partition.append(part)
            
    # Verify partition validity and format
    # Types must be list of lists of ints
    # Check if every part is clique or independent
    # This is guaranteed by construction from maximal cliques/independent sets
    
    return partition

# EVOLVE-BLOCK-END
