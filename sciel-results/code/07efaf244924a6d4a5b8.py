"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute: for any subset, is it a clique? independent set?
    # Represent subsets as bitmasks
    all_vertices_mask = (1 << n) - 1
    
    # Precompute clique and independent set properties for all subsets
    # A subset is a clique if every pair is an edge
    # A subset is independent if no pair is an edge
    
    def is_clique(mask):
        if mask == 0:
            return True
        # Check all pairs
        bits = []
        m = mask
        while m:
            lb = m & (-m)
            bits.append(lb.bit_length() - 1)
            m ^= lb
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                if bits[j] not in adj[bits[i]]:
                    return False
        return True
    
    def is_independent(mask):
        if mask == 0:
            return True
        bits = []
        m = mask
        while m:
            lb = m & (-m)
            bits.append(lb.bit_length() - 1)
            m ^= lb
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                if bits[j] in adj[bits[i]]:
                    return False
        return True
    
    # Precompute for all 2^14 = 16384 subsets
    clique = [False] * (1 << n)
    indep = [False] * (1 << n)
    for mask in range(1 << n):
        if mask == 0:
            clique[mask] = True
            indep[mask] = True
        else:
            # Use DP: a set is clique if removing any vertex leaves a clique AND that vertex connects to all others
            # Simpler: just check directly for small n
            clique[mask] = is_clique(mask)
            indep[mask] = is_independent(mask)
    
    # Also precompute: can a set S be covered by k cliques or independent sets?
    # For the search, we'll use branch and bound
    
    # Lower bound: max of (chromatic number of G, chromatic number of complement)
    # Actually, the cochromatic number is at least the max of chi(G) and chi(complement G)
    # But also at least the maximum of (omega(G), alpha(G)) and related bounds
    
    # Let's compute a simple lower bound
    # The partition must cover all vertices. Each part is either clique or independent.
    # So we need at least ceil(n / max_clique_size) parts if all cliques, but that's not quite right.
    
    # Better lower bound: 
    # For any partition into k parts, each part is either a clique or independent set.
    # Consider the complement: independent sets in G are cliques in complement.
    # So the problem is symmetric.
    
    # Simple LB: the size of a maximum clique gives that we need at least 1 part for it,
    # but that's weak. Let's use: LB = max(ceil(n / alpha_max), ceil(n / omega_max)) where
    # alpha_max is max independent set size, omega_max is max clique size.
    
    # Actually, a better approach: try k from 1 upward, and check if a valid k-partition exists.
    
    # For checking if a k-partition exists, use backtracking:
    # Assign each vertex to one of k parts, where each part must be all-clique or all-independent.
    # But we don't know in advance which parts are cliques and which are independent.
    
    # Alternative: enumerate all maximal cliques and maximal independent sets, then find a set cover.
    
    # Let's collect all maximal cliques and maximal independent sets
    # Then we need to find a minimum set cover where each set is either a clique or independent set.
    
    # For n=14, we can enumerate all subsets and check which are maximal cliques/independent sets.
    
    # Collect all homogeneous sets (cliques or independent sets) that are "useful"
    # Actually, for the search, let's use a different approach:
    
    # Backtracking search:
    # State: which vertices are unassigned, and for each part so far, whether it's clique or independent
    # We build parts one by one.
    
    # Better: assign vertices one by one to existing or new parts.
    
    # Let me use a cleaner backtracking:
    # We have k slots. For each slot, we track: set of vertices assigned, and type (clique/indep/undecided)
    # A slot is valid if it's a clique or independent set.
    
    # To make this efficient, I'll use the following strategy:
    # - Try k from a lower bound upward
    # - Use symmetry breaking: parts are ordered, so we only assign vertex v to part i if 
    #   either i is the first unused part, or part i-1 is non-empty (to avoid permuting identical empty parts)
    
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s margin
    
    def solve(k):
        """Check if a valid k-partition exists. Returns the partition if found."""
        # parts[i] = (set_of_vertices, type) where type is 0=clique, 1=independent, -1=undecided
        # We'll use lists for efficiency
        parts = []  # list of [set, type]
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        def backtrack(idx, parts):
            if time.time() > deadline:
                return None
            
            if idx == n:
                # Check all parts are valid
                for part_set, part_type in parts:
                    if not part_set:
                        continue
                    m = 0
                    for v in part_set:
                        m |= (1 << v)
                    if part_type == 0 and not clique[m]:
                        return None
                    if part_type == 1 and not indep[m]:
                        return None
                    if part_type == -1:
                        # Should have been resolved
                        if not (clique[m] or indep[m]):
                            return None
                return [list(p[0]) for p in parts if p[0]]
            
            v = order[idx]
            
            # Try assigning v to each existing part
            for i, (part_set, part_type) in enumerate(parts):
                # Symmetry breaking: only try part i if i==0 or parts[i-1] is non-empty
                # Actually, we need to be more careful. Let's just try all and rely on the search.
                
                new_set = set(part_set)
                new_set.add(v)
                m = 0
                for u in new_set:
                    m |= (1 << u)
                
                # Determine if this assignment is valid
                if part_type == 0:
                    # Must remain a clique
                    if not clique[m]:
                        continue
                elif part_type == 1:
                    # Must remain independent
                    if not indep[m]:
                        continue
                else:
                    # Undecided: must be either clique or independent
                    if not (clique[m] or indep[m]):
                        continue
                    # Determine the type
                    if clique[m] and indep[m]:
                        # Can be either; try both
                        # Try as clique first
                        result = backtrack(idx + 1, parts[:i] + [[new_set, 0]] + parts[i+1:])
                        if result is not None:
                            return result
                        # Try as independent
                        result = backtrack(idx + 1, parts[:i] + [[new_set, 1]] + parts[i+1:])
                        if result is not None:
                            return result
                        continue
                    elif clique[m]:
                        new_type = 0
                    else:
                        new_type = 1
                
                new_parts = parts[:i] + [[new_set, part_type if part_type != -1 else (0 if clique[m] else 1)]] + parts[i+1:]
                result = backtrack(idx + 1, new_parts)
                if result is not None:
                    return result
            
            # Try creating a new part (if we haven't used all k parts yet)
            if len(parts) < k:
                new_set = {v}
                m = 1 << v
                # A single vertex is both a clique and independent
                # Try as clique
                new_parts = parts + [[new_set, 0]]
                result = backtrack(idx + 1, new_parts)
                if result is not None:
                    return result
                # Try as independent
                new_parts = parts + [[new_set, 1]]
                result = backtrack(idx + 1, new_parts)
                if result is not None:
                    return result
            
            return None
        
        # Initial state: no parts, or start with one part containing the first vertex
        # Let's start with an empty list of parts
        result = backtrack(0, [])
        return result
    
    # Compute a lower bound
    # The minimum number of parts is at least 1
    # Also, consider: if the graph is complete, we need 1 part (it's a clique)
    # If the graph is empty, we need 1 part (it's independent)
    
    # A better LB: the cochromatic number is at least the maximum over all subsets S of
    # ceil(|S| / max{alpha(S), omega(S)})... this is getting complicated.
    
    # Let's just try k from 1 upward
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        result = solve(k)
        if result is not None:
            # Convert to list of lists of ints
            final_parts = []
            for part in result:
                final_parts.append([int(v) for v in part])
            return final_parts
    
    # Fallback: greedy assignment (should not reach here for n=14)
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            m = 0
            for u in trial:
                m |= (1 << u)
            if clique[m] or indep[m]:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    
    return [[int(v) for v in part] for part in parts]

# EVOLVE-BLOCK-END
