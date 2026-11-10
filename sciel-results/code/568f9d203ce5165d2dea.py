"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute all maximal cliques and maximal independent sets using bitmasks
    # A set of vertices is a clique if for every pair, they are adjacent
    # An independent set is a clique in the complement graph
    
    # Let's use a simple approach: generate all subsets and check if clique/independent
    # But n=14, so 2^14 = 16384 subsets, which is manageable
    
    # For each subset, check if it's a clique
    # A subset S is a clique if for every v in S, S \ {v} is a subset of adj[v]
    
    all_cliques = []
    all_independent_sets = []
    
    for mask in range(1, 1 << n):
        # Check if clique
        is_clique = True
        is_independent = True
        vertices = []
        for v in range(n):
            if mask & (1 << v):
                vertices.append(v)
        
        if len(vertices) <= 1:
            all_cliques.append(frozenset(vertices))
            all_independent_sets.append(frozenset(vertices))
            continue
        
        # Check clique
        for v in vertices:
            for w in vertices:
                if v != w and not (adj[v] & (1 << w)):
                    is_clique = False
                    break
            if not is_clique:
                break
        
        # Check independent set
        if is_independent:
            for v in vertices:
                for w in vertices:
                    if v != w and (adj[v] & (1 << w)):
                        is_independent = False
                        break
                if not is_independent:
                    break
        
        if is_clique:
            all_cliques.append(frozenset(vertices))
        if is_independent:
            all_independent_sets.append(frozenset(vertices))
    
    # Now we want to partition vertices into minimum number of parts,
    # where each part is a clique or independent set.
    
    # Use backtracking with memoization. State: (remaining_vertices_mask, current_parts_mask)
    # But parts can be any clique or independent set, so we need to track which parts we've used.
    
    # Alternative: since n=14 is small, we can try all possible numbers of parts k=1,2,...
    # and check if we can cover all vertices with k parts where each part is a clique or independent set.
    
    # For a given k, we need to assign each vertex to one of k parts, such that each part is a clique or independent set.
    
    # Let's use a different approach: generate all valid parts (cliques and independent sets),
    # then use set cover / exact cover with minimum number of sets.
    
    # Collect all valid parts (non-empty cliques and non-empty independent sets)
    valid_parts = set()
    for c in all_cliques:
        valid_parts.add(c)
    for s in all_independent_sets:
        valid_parts.add(s)
    
    # Convert to list for indexing
    part_list = list(valid_parts)
    part_to_mask = [0] * len(part_list)
    for i, part in enumerate(part_list):
        mask = 0
        for v in part:
            mask |= (1 << v)
        part_to_mask[i] = mask
    
    # We want to find the minimum number of parts that cover all n vertices exactly once.
    # This is an exact cover problem.
    
    # Use backtracking: pick a vertex, try all parts containing that vertex,
    # remove those vertices from remaining, recurse.
    
    full_mask = (1 << n) - 1
    
    # Sort vertices by degree (descending) to prioritize high-degree vertices
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # For each vertex, precompute which parts contain it
    parts_containing_vertex = [[] for _ in range(n)]
    for i, mask in enumerate(part_to_mask):
        for v in range(n):
            if mask & (1 << v):
                parts_containing_vertex[v].append(i)
    
    # Backtracking search
    best = [n + 1]  # best number of parts found
    
    def backtrack(remaining_mask, num_parts, used_parts_mask):
        if remaining_mask == 0:
            if num_parts < best[0]:
                best[0] = num_parts
                best_parts = used_parts_mask.copy()
            return
        
        # Pruning: if num_parts >= best[0], stop
        if num_parts >= best[0]:
            return
        
        # Pick the vertex with highest degree among remaining
        min_v = -1
        min_deg = -1
        for v in vertex_order:
            if remaining_mask & (1 << v):
                if degrees[v] > min_deg:
                    min_deg = degrees[v]
                    min_v = v
                break
        
        if min_v == -1:
            return
        
        # Try all parts containing min_v
        for part_idx in parts_containing_vertex[min_v]:
            part_mask = part_to_mask[part_idx]
            # Part must be subset of remaining
            if part_mask & ~remaining_mask:
                continue
            # Try adding this part
            new_remaining = remaining_mask & ~part_mask
            backtrack(new_remaining, num_parts + 1, used_parts_mask | (1 << part_idx))
    
    # Since the number of parts can be large, and part_list can be large,
    # let's try a simpler greedy + local search approach first, then refine.
    
    # Actually, let's just do a direct backtracking without tracking used_parts_mask,
    # since we only care about the count.
    
    best_count = [n + 1]
    best_solution = [None]
    
    def backtrack2(remaining_mask, num_parts, current_parts):
        if remaining_mask == 0:
            if num_parts < best_count[0]:
                best_count[0] = num_parts
                best_solution[0] = current_parts.copy()
            return
        
        if num_parts >= best_count[0]:
            return
        
        # Pick vertex with highest degree in remaining
        min_v = -1
        min_deg = -1
        for v in vertex_order:
            if remaining_mask & (1 << v):
                if degrees[v] > min_deg:
                    min_deg = degrees[v]
                    min_v = v
                break
        
        if min_v == -1:
            return
        
        # Try all parts containing min_v that are subsets of remaining
        for part_idx in parts_containing_vertex[min_v]:
            part_mask = part_to_mask[part_idx]
            if part_mask & ~remaining_mask:
                continue
            new_remaining = remaining_mask & ~part_mask
            current_parts.append(part_list[part_idx])
            backtrack2(new_remaining, num_parts + 1, current_parts)
            current_parts.pop()
    
    backtrack2(full_mask, 0, [])
    
    if best_solution[0] is not None:
        # Convert frozensets to lists
        result = [list(part) for part in best_solution[0]]
        return result
    else:
        # Fallback: single vertex per part
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
