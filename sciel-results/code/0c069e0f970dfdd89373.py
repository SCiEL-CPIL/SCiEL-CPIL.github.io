"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all maximal cliques
    def find_maximal_cliques():
        cliques = []
        # Use Bron-Kerbosch with pivot
        def bron_kerbosch(R, P_set, X):
            if not P_set and not X:
                cliques.append(set(R))
                return
            # Choose pivot u from P_set ∪ X to minimize branching
            candidates = list(P_set) + list(X)
            if not candidates:
                return
            # Pick pivot with max degree in P_set
            pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
            for v in list(P_set - adj[pivot]):
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X & adj[v]
                bron_kerbosch(new_R, new_P, new_X)
                P_set.discard(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return cliques
    
    # Find all maximal independent sets (same as maximal cliques in complement)
    def find_maximal_independent_sets():
        # Complement adjacency
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and v not in adj[u]:
                    comp_adj[u].add(v)
        
        isets = []
        def bron_kerbosch_comp(R, P_set, X):
            if not P_set and not X:
                isets.append(set(R))
                return
            candidates = list(P_set) + list(X)
            if not candidates:
                return
            pivot = max(candidates, key=lambda u: len(comp_adj[u] & P_set))
            for v in list(P_set - comp_adj[pivot]):
                new_R = R | {v}
                new_P = P_set & comp_adj[v]
                new_X = X & comp_adj[v]
                bron_kerbosch_comp(new_R, new_P, new_X)
                P_set.discard(v)
                X.add(v)
        
        bron_kerbosch_comp(set(), set(range(n)), set())
        return isets
    
    max_cliques = find_maximal_cliques()
    max_indep_sets = find_maximal_independent_sets()
    
    # Candidate sets: all maximal cliques and maximal independent sets
    candidate_sets = []
    seen = set()
    for cs in max_cliques:
        cs_frozenset = frozenset(cs)
        if cs_frozenset not in seen:
            seen.add(cs_frozenset)
            candidate_sets.append(cs)
    for iset in max_indep_sets:
        iset_frozenset = frozenset(iset)
        if iset_frozenset not in seen:
            seen.add(iset_frozenset)
            candidate_sets.append(iset)
    
    # For each candidate set, compute its bitmask
    candidate_masks = []
    for cs in candidate_sets:
        mask = 0
        for v in cs:
            mask |= (1 << v)
        candidate_masks.append(mask)
    
    all_vertices_mask = (1 << n) - 1
    
    # Branch and bound for minimum set cover
    best_parts = None
    best_count = n + 1
    
    # Sort candidates by size descending for better pruning
    candidate_indices = sorted(range(len(candidate_masks)), key=lambda i: -len(candidate_sets[i]))
    
    def solve_set_cover(remaining_mask, chosen_indices, depth, current_count):
        nonlocal best_parts, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if current_count >= best_count:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_parts = [list(candidate_sets[i]) for i in chosen_indices]
            return
        
        # Lower bound: find minimum number of sets needed to cover remaining
        # Greedy lower bound: repeatedly pick the set that covers the most uncovered vertices
        lb_remaining = remaining_mask
        lb_count = 0
        temp_mask = remaining_mask
        while temp_mask != 0:
            best_cover = 0
            for i in candidate_indices:
                cover = candidate_masks[i] & temp_mask
                if cover != 0 and bin(cover).count('1') > best_cover:
                    best_cover = bin(cover).count('1')
            if best_cover == 0:
                # Can't cover, this is a dead end
                return
            lb_count += 1
            # Remove the best covered vertices (approximation)
            # Actually, for a valid lower bound, we need to be careful
            # Let's use a simpler lower bound: ceiling of (uncovered vertices) / (max set size)
            break
        
        # Better lower bound: for each uncovered vertex, it must be in some set
        # Count how many distinct "requirements" exist
        uncovered_vertices = [v for v in range(n) if (remaining_mask >> v) & 1]
        if not uncovered_vertices:
            return
        
        # Simple lower bound: at least 1 if there are uncovered vertices
        if current_count + 1 >= best_count:
            return
        
        # Try each candidate set that covers at least one uncovered vertex
        # Order by how many uncovered vertices it covers (descending)
        options = []
        for i in candidate_indices:
            cover = candidate_masks[i] & remaining_mask
            if cover != 0:
                options.append((bin(cover).count('1'), i))
        
        options.sort(reverse=True)
        
        for _, i in options:
            if time.time() > deadline:
                return
            if current_count + 1 >= best_count:
                break
            new_remaining = remaining_mask & ~candidate_masks[i]
            chosen_indices.append(i)
            solve_set_cover(new_remaining, chosen_indices, depth + 1, current_count + 1)
            chosen_indices.pop()
    
    solve_set_cover(all_vertices_mask, [], 0, 0)
    
    if best_parts is None:
        # Fallback: use the greedy approach from the original
        parts = []
        es = {tuple(e) for e in edges}
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for idx, u in enumerate(trial) for w in trial[idx+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Validate and return
    result = []
    for part in best_parts:
        part_list = list(part)
        result.append(part_list)
    
    return result

# EVOLVE-BLOCK-END
