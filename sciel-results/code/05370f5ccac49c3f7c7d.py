"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute all maximal cliques and independent sets
    all_cliques = []
    all_independent = []
    
    # For each subset, check if it's a clique or independent set
    # We'll generate maximal ones using a simple approach for n=14
    # A set S is a clique if for all u,v in S, (1<<v) & adj[u] != 0
    # A set S is independent if for all u,v in S, (1<<v) & adj[u] == 0
    
    # Let's enumerate all subsets (2^14 = 16384, very fast)
    for mask in range(1, 1 << n):
        # Check if it's a clique
        is_clique = True
        is_indep = True
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if (adj[u] >> v) & 1:
                    is_indep = False
                else:
                    is_clique = False
            if not is_clique and not is_indep:
                break
        
        if is_clique:
            all_cliques.append(mask)
        if is_indep:
            all_independent.append(mask)
    
    # Combine all valid parts
    all_parts = list(set(all_cliques + all_independent))
    # Sort by size descending to try larger parts first
    all_parts.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    # Memoization for subproblems
    memo = {}
    memo[0] = 0  # Empty set needs 0 parts
    
    # We want to find the minimum number of parts
    # Use DFS with branch and bound
    def search(remaining, best, parts_list):
        if time.time() > deadline:
            return False
        if remaining == 0:
            return True
        if remaining in memo:
            if memo[remaining] >= best:
                return False
            else:
                # We already know we can solve it in memo[remaining] parts
                # But we need to reconstruct, so we still need to search
                pass
        
        # Try each valid part that is a subset of remaining
        # To be efficient, only try maximal parts or parts that are "good"
        # For simplicity, try all subsets that are valid parts
        found = False
        for part_mask in all_parts:
            if part_mask & remaining != part_mask:
                continue
            new_remaining = remaining ^ part_mask
            if new_remaining == remaining:
                continue
            
            # Pruning: if we've already seen this state with a better or equal bound
            if new_remaining in memo and memo[new_remaining] + 1 >= best:
                continue
            
            # Recursive search
            if search(new_remaining, best - 1, parts_list):
                parts_list.append([v for v in range(n) if (part_mask >> v) & 1])
                memo[remaining] = best
                return True
        
        memo[remaining] = best
        return False
    
    # Start with an upper bound from greedy
    # Simple greedy to get initial bound
    greedy_parts = []
    used = 0
    for v in range(n):
        if (used >> v) & 1:
            continue
        # Find a maximal clique or independent set containing v
        best_part = 1 << v
        # Try to extend
        for mask in all_parts:
            if (mask >> v) & 1 and (mask & ~used) == mask and bin(mask).count('1') > bin(best_part).count('1'):
                best_part = mask
        greedy_parts.append([x for x in range(n) if (best_part >> x) & 1])
        used |= best_part
    
    upper_bound = len(greedy_parts)
    
    # Now search for optimal
    best_solution = None
    parts_list = []
    if search((1 << n) - 1, upper_bound, parts_list):
        best_solution = parts_list
    
    if best_solution is None:
        # Fallback to greedy
        return [list(p) for p in greedy_parts]
    
    # Convert to list of lists of ints
    result = []
    for part in best_solution:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
