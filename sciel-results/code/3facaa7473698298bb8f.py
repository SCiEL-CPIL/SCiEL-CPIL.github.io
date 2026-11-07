"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj_mask = [0] * n
    for e in P['edges']:
        u, v = e[0], e[1]
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deadline = time.time() + 55  # safety margin
    
    # Precompute clique and independent set checks using bitmasks
    # A subset S (bitmask) is a clique if for every v in S, S is subset of adj_mask[v] | (1<<v)
    # A subset S is independent if for every v in S, S is subset of ~(adj_mask[v]) & full_mask | (1<<v)
    full_mask = (1 << n) - 1
    
    # Memoization cache: (subset, k) -> bool
    # We'll use a dict for memoization
    memo = {}
    
    # Precompute all maximal cliques and independent sets? No, that's expensive.
    # Instead, use recursive search with memoization on (remaining_set, k_remaining)
    
    def is_clique(subset):
        """Check if subset is a clique using bitmask operations"""
        if subset == 0:
            return True
        # Get lowest set bit
        lsb = subset & (-subset)
        v = lsb.bit_length() - 1
        # Check if all other vertices in subset are adjacent to v
        others = subset ^ lsb
        # v is adjacent to all in others if (adj_mask[v] & others) == others
        if (adj_mask[v] & others) != others:
            return False
        # Recurse on others
        return is_clique(others)
    
    def is_independent(subset):
        """Check if subset is an independent set using bitmask operations"""
        if subset == 0:
            return True
        lsb = subset & (-subset)
        v = lsb.bit_length() - 1
        others = subset ^ lsb
        # v is non-adjacent to all in others if (adj_mask[v] & others) == 0
        if (adj_mask[v] & others) != 0:
            return False
        return is_independent(others)
    
    # Use iterative deepening: try k=1, 2, ...
    # For each k, use backtracking with memoization
    
    def solve_subset(remaining, k):
        """Can we partition 'remaining' into at most k parts (each clique or independent)?"""
        if remaining == 0:
            return True, []
        if k == 0:
            return False, None
        
        key = (remaining, k)
        if key in memo:
            return memo[key]
        
        # Try to find a part (clique or independent set) from remaining
        # We need to enumerate subsets of remaining that are cliques or independent sets
        # To reduce search, we can fix one vertex and enumerate subsets containing it
        
        # Pick a vertex from remaining (lowest bit for determinism)
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        
        # Enumerate all subsets of remaining that contain v and are cliques
        # Then check if the rest can be partitioned into k-1 parts
        
        # To enumerate subsets containing v: iterate over submasks of (remaining ^ lsb)
        # But this is 2^(|remaining|-1) which is too many for large remaining
        
        # Better approach: use recursive enumeration with pruning
        # A subset S containing v is a clique if S ⊆ {v} ∪ N(v) ∩ remaining
        # A subset S containing v is independent if S ⊆ {v} ∪ (V \ N(v)) ∩ remaining
        
        # Let's compute the candidate vertices for clique containing v:
        clique_cands = remaining & (adj_mask[v] | lsb)
        # And for independent set containing v:
        indep_cands = remaining & (~(adj_mask[v]) & full_mask)
        # Note: ~(adj_mask[v]) & full_mask gives all non-neighbors of v (including v itself)
        
        # Try all subsets of clique_cands that contain v
        # And all subsets of indep_cands that contain v
        
        # To avoid duplicates, we'll enumerate submasks of (clique_cands ^ lsb) and add lsb
        # Same for indep_cands
        
        # Optimization: try larger subsets first (greedy)
        
        # Enumerate subsets of clique_cands containing v
        rest_clique = clique_cands ^ lsb
        sub = rest_clique
        while True:
            part = sub | lsb
            # Check if part is a clique
            if is_clique(part):
                rest = remaining ^ part
                if rest == 0:
                    if k >= 1:
                        memo[key] = (True, [list(range(n))])  # placeholder
                        # Need to reconstruct, so let's use a different approach
                        pass
                elif k >= 2:
                    # Check if rest can be partitioned into k-1 parts
                    ok, _ = solve_subset(rest, k-1)
                    if ok:
                        memo[key] = (True, None)
                        return (True, None)
            if sub == 0:
                break
            sub = (sub - 1) & rest_clique
        
        # Enumerate subsets of indep_cands containing v
        rest_indep = indep_cands ^ lsb
        sub = rest_indep
        while True:
            part = sub | lsb
            if is_independent(part):
                rest = remaining ^ part
                if rest == 0:
                    if k >= 1:
                        memo[key] = (True, None)
                        return (True, None)
                elif k >= 2:
                    ok, _ = solve_subset(rest, k-1)
                    if ok:
                        memo[key] = (True, None)
                        return (True, None)
            if sub == 0:
                break
            sub = (sub - 1) & rest_indep
        
        memo[key] = (False, None)
        return (False, None)
    
    # The above approach has issues with reconstruction. Let's use a simpler backtracking
    # that directly builds the partition.
    
    best_parts = None
    best_count = n + 1
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: -bin(adj_mask[v]).count('1'))
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if k >= best_count:
            break
        
        parts = [[] for _ in range(k)]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # Check all parts are non-empty
                for p in parts:
                    if not p:
                        return False
                return True
            
            v = order[idx]
            
            # Try adding v to each part
            for pi in range(k):
                # Check if v can be added to parts[pi] as clique or independent
                # We need to track the type of each part
                # Actually, we should determine the type when we first add a vertex
                
                # Let's use a different approach: for each part, track if it's clique, independent, or empty
                pass
            
            return False
        
        # The above is getting complicated. Let's use a cleaner bitmask-based approach.
        
        # For a given k, we want to find a partition of all vertices into k parts.
        # Each part is either a clique or an independent set.
        
        # Use recursive search: pick the first unassigned vertex, try to assign it
        # to a new part (as clique or independent) or to an existing part.
        
        # But this is essentially the same as the original approach, just with bitmask operations.
        
        # Let's try a different strategy: enumerate all possible parts (cliques and independent sets)
        # and then use set cover / exact cover.
        
        # For n=14, the number of cliques and independent sets is manageable.
        
        # Generate all maximal cliques and maximal independent sets
        # Then find the minimum set cover.
        
        pass
    
    # Fallback: use the original backtracking approach with bitmask optimizations
    # Let's rewrite the original approach with bitmask-based clique/independent checks
    
    is_adj = [[False]*n for _ in range(n)]
    for e in P['edges']:
        is_adj[e[0]][e[1]] = True
        is_adj[e[1]][e[0]] = True
    
    order = sorted(range(n), key=lambda v: -len([u for u in range(n) if is_adj[v][u]]))
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if k >= best_count:
            break
        
        parts = [[] for _ in range(k)]
        part_types = [None] * k  # None, 'clique', or 'independent'
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # All vertices assigned
                for pi in range(k):
                    if not parts[pi]:
                        return False
                return True
            
            v = order[idx]
            
            for pi in range(k):
                t = part_types[pi]
                
                if t is None:
                    # Empty part, can be either clique or independent
                    parts[pi].append(v)
                    part_types[pi] = 'clique'
                    if backtrack(idx + 1):
                        return True
                    part_types[pi] = 'independent'
                    if backtrack(idx + 1):
                        return True
                    parts[pi].pop()
                    part_types[pi] = None
                elif t == 'clique':
                    # Check if v can be added to clique
                    can = True
                    for u in parts[pi]:
                        if not is_adj[u][v]:
                            can = False
                            break
                    if can:
                        parts[pi].append(v)
                        if backtrack(idx + 1):
                            return True
                        parts[pi].pop()
                elif t == 'independent':
                    # Check if v can be added to independent set
                    can = True
                    for u in parts[pi]:
                        if is_adj[u][v]:
                            can = False
                            break
                    if can:
                        parts[pi].append(v)
                        if backtrack(idx + 1):
                            return True
                        parts[pi].pop()
            
            return False
        
        if backtrack(0):
            result = []
            for p in parts:
                if p:
                    result.append(list(p))
            if result and len(result) < best_count:
                best_parts = result
                best_count = len(result)
            break
    
    if best_parts is None:
        best_parts = [[v] for v in range(n)]
    
    final = []
    for p in best_parts:
        if p:
            final.append(list(p))
    
    return final

# EVOLVE-BLOCK-END
