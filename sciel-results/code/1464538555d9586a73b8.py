"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 85.0
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute: for each vertex, its neighbors and non-neighbors (excluding itself)
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = (all_mask ^ (1 << v)) & ~adj[v]
    
    # A set S is a clique if every vertex in S is adjacent to every other
    # A set S is independent if every vertex in S is non-adjacent to every other
    
    def is_clique(mask):
        if mask == 0:
            return True
        # Pick lowest set bit
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        # All vertices in rest must be adjacent to v
        return (adj[v] & rest) == rest and is_clique(rest)
    
    def is_independent(mask):
        if mask == 0:
            return True
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        # All vertices in rest must be non-adjacent to v
        return (non_adj[v] & rest) == rest and is_independent(rest)
    
    # Branch and bound: try to cover all vertices with k parts, each part is a clique or independent set
    # We'll try k=1,2,3,... until we find a valid partition
    
    def find_partition(k):
        """Try to partition all vertices into k parts, each a clique or independent set."""
        # Use backtracking with memoization on (uncovered_mask, remaining_parts)
        # For speed, we'll use a simple recursive search with pruning
        
        # Lower bound: we need at least max(ceil(n/k_max_part_size)) but that's complex
        # Simpler: if k=1, check if whole thing is clique or independent
        if k == 1:
            if is_clique(all_mask) or is_independent(all_mask):
                if is_clique(all_mask):
                    return [list(range(n))]
                else:
                    return [list(range(n))]
            return None
        
        # Try to assign vertices greedily first to get a quick solution, then optimize
        # Actually, let's do a proper backtracking search
        
        # Strategy: pick the first uncovered vertex, try all maximal homogeneous sets containing it
        # This could be expensive, so let's use a smarter approach
        
        # First, let's try a greedy construction to get an upper bound
        def greedy_construct():
            parts = []
            uncovered = list(range(n))
            while uncovered:
                v = uncovered[0]
                # Try to make the largest possible clique or independent set starting with v
                # Greedily add vertices
                part = [v]
                uncovered.remove(v)
                # Try adding more vertices
                for u in list(uncovered):
                    # Check if part + [u] is still homogeneous
                    trial = part + [u]
                    mask = 0
                    for x in trial:
                        mask |= (1 << x)
                    if is_clique(mask) or is_independent(mask):
                        part.append(u)
                        uncovered.remove(u)
                parts.append(part)
            return parts
        
        # Get greedy solution as starting point
        greedy = greedy_construct()
        if len(greedy) <= k:
            return greedy
        
        # Now try to improve: backtracking search
        # We need exactly k parts (or fewer, but we're trying k so <=k)
        # Use recursive backtracking
        
        best = None
        def backtrack(uncovered_mask, parts_so_far, remaining):
            nonlocal best
            if best is not None:
                return
            if time.time() > deadline:
                return
            
            if uncovered_mask == 0:
                if len(parts_so_far) <= k:
                    best = [list(p) for p in parts_so_far]
                return
            
            if remaining == 0:
                return
            
            # Pruning: if remaining parts can't cover uncovered vertices
            # Each part can have at most n vertices, so if remaining * max_possible < popcount, prune
            # But that's loose. Better: if remaining < 1 and uncovered != 0, prune (already handled)
            
            # Pick the lowest set bit vertex to branch on
            v = (uncovered_mask & -uncovered_mask).bit_length() - 1
            v_bit = 1 << v
            
            # Try all subsets of uncovered_mask that contain v and are homogeneous (clique or independent)
            # We'll enumerate by trying to add vertices to form a homogeneous set containing v
            
            # Build candidate sets containing v from uncovered_mask
            # Start with just {v}
            candidates = []
            
            # Try cliques containing v
            # We can build up from v
            def extend_clique(current_mask, available):
                """Try to extend current_mask by adding vertices from available to form a clique"""
                if current_mask == 0:
                    return
                # Get lowest bit of current_mask
                cv = (current_mask & -current_mask).bit_length() - 1
                # Vertices in available that are adjacent to all in current_mask
                # Actually, we need vertices that are adjacent to all vertices in current_mask
                compatible = available
                cm = current_mask
                while cm:
                    u = (cm & -cm).bit_length() - 1
                    compatible &= adj[u]
                    cm ^= (1 << u)
                
                if compatible & available:
                    # Can extend
                    comp_bits = compatible & available
                    while comp_bits:
                        ub = comp_bits & -comp_bits
                        u = ub.bit_length() - 1
                        new_mask = current_mask | ub
                        candidates.append(new_mask)
                        extend_clique(new_mask, available ^ ub)
                        comp_bits ^= ub
                
            def extend_independent(current_mask, available):
                """Try to extend current_mask by adding vertices from available to form an independent set"""
                if current_mask == 0:
                    return
                cv = (current_mask & -current_mask).bit_length() - 1
                compatible = available
                cm = current_mask
                while cm:
                    u = (cm & -cm).bit_length() - 1
                    compatible &= non_adj[u]
                    cm ^= (1 << u)
                
                if compatible & available:
                    comp_bits = compatible & available
                    while comp_bits:
                        ub = comp_bits & -comp_bits
                        u = ub.bit_length() - 1
                        new_mask = current_mask | ub
                        candidates.append(new_mask)
                        extend_independent(new_mask, available ^ ub)
                        comp_bits ^= ub
            
            # This enumeration can be exponential. For n=14 it should be manageable with pruning.
            # Let's try a simpler approach: iterate over all subsets of uncovered containing v
            # and check if they're homogeneous.
            
            # Actually for n=14, there are at most 2^14 = 16384 subsets, but we only care about those containing v.
            # That's 2^13 = 8192 subsets at most. But we can prune more.
            
            # Let's enumerate subsets of (uncovered_mask ^ v_bit) and add v_bit
            rest = uncovered_mask ^ v_bit
            
            # Enumerate subsets of rest
            sub = rest
            # We want to try larger sets first to cover more vertices
            # Actually, let's try all subsets
            
            # For efficiency, let's try to build up from {v}
            # We'll do a BFS/DFS to build homogeneous sets containing v
            
            # Simpler: just try all 2^(popcount(rest)) subsets
            # popcount(rest) <= 13, so at most 8192 subsets
            # For each, check if {v} union subset is clique or independent
            
            # To be faster, let's sort candidates by size (larger first) to find solutions quicker
            
            # Generate all subsets of rest
            # Use bit manipulation
            # Actually, let's just iterate
            subset = rest
            # We'll collect all valid homogeneous sets containing v
            valid_sets = []
            
            # Iterate over all subsets of rest
            # This is 2^13 = 8192 iterations max, each with a clique/independent check
            # The check is O(popcount) so total is about 8192 * 14 = ~115k operations per level
            # With depth up to 14, this could be slow but n=14 is small
            
            # Optimization: precompute which subsets are cliques/independent? 
            # Too many subsets. Let's just compute on the fly.
            
            # For speed, let's try a different approach:
            # Just pick one vertex and assign it to a new part, trying to maximize the part size greedily
            # then backtrack if needed
            
            # Given the time constraint, let's use a simpler but effective heuristic:
            # 1. Greedy to get initial solution
            # 2. Local search: try to merge or swap vertices between parts to reduce count
            
            # Actually, let's just do the full backtracking but with good pruning.
            # For n=14, the search space should be manageable.
            
            # Let's try: for the current vertex v, try all possible homogeneous sets S containing v
            # where S is a subset of uncovered_mask
            # We'll enumerate by growing S from {v}
            
            # Start with S = {v}
            current = v_bit
            # Try to add vertices to S
            # We'll use a stack-based DFS to enumerate homogeneous sets containing v
            
            # To keep it simple and fast for n=14, let's just try:
            # - S = {v}
            # - S = {v, u} for each u in uncovered
            # - S = {v, u, w} for each pair
            # etc., but only if homogeneous
            
            # Let's limit the size to keep it manageable, but try all sizes
            
            # Actually, let's just do the full subset enumeration. 8192 subsets is fine.
            
            sub = rest
            while True:
                S = sub | v_bit
                if is_clique(S) or is_independent(S):
                    valid_sets.append(S)
                if sub == 0:
                    break
                # Next subset: standard bit trick
                # sub = (sub - 1) & rest
                sub = (sub - 1) & rest
            
            # Sort valid_sets by size (larger first) to cover more vertices
            valid_sets.sort(key=lambda m: m.bit_count(), reverse=True)
            
            for S in valid_sets:
                new_uncovered = uncovered_mask ^ S
                parts_so_far.append(S)
                backtrack(new_uncovered, parts_so_far, remaining - 1)
                parts_so_far.pop()
                if best is not None:
                    return
            
        backtrack(all_mask, [], k)
        return best
    
    # Try k = 1, 2, 3, ... until we find a valid partition
    for k in range(1, n + 1):
        if time.time() > deadline - 5.0:
            break
        result = find_partition(k)
        if result is not None:
            # Verify the result
            # Each part should be a clique or independent set
            # All vertices should be covered exactly once
            seen = [False] * n
            for part in result:
                for v in part:
                    if seen[v]:
                        # Duplicate, invalid
                        continue
                    seen[v] = True
                # Check homogeneity
                mask = 0
                for v in part:
                    mask |= (1 << v)
                if not (is_clique(mask) or is_independent(mask)):
                    # Invalid part
                    continue
            # If we got here without continuing, result is valid
            # Convert to proper format
            return [[int(v) for v in part] for part in result]
    
    # Fallback: return greedy solution
    def greedy_construct():
        parts = []
        uncovered = list(range(n))
        while uncovered:
            v = uncovered[0]
            part = [v]
            uncovered.remove(v)
            for u in list(uncovered):
                trial = part + [u]
                mask = 0
                for x in trial:
                    mask |= (1 << x)
                if is_clique(mask) or is_independent(mask):
                    part.append(u)
                    uncovered.remove(u)
            parts.append(part)
        return parts
    
    return greedy_construct()

# EVOLVE-BLOCK-END
