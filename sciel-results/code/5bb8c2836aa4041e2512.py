"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    def bron_kerbosch(R, P_set, X, max_sets, adj_set):
        if time.time() > deadline:
            return
        if not P_set and not X:
            if R:
                max_sets.append(frozenset(R))
            return
        pivot_set = P_set | X
        if pivot_set:
            pivot = max(pivot_set, key=lambda v: len(P_set & adj_set[v]))
            candidates = P_set - adj_set[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj_set[v], X & adj_set[v], max_sets, adj_set)
            P_set.discard(v)
            X.add(v)
    
    all_cliques = []
    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset(), all_cliques, adj)
    
    all_indep = []
    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset(), all_indep, comp_adj)
    
    candidates = []
    seen = set()
    for cs in all_cliques + all_indep:
        if cs not in seen:
            seen.add(cs)
            candidates.append(cs)
    
    # Remove dominated candidates
    filtered = []
    for i, cs in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i != j and cs < other:
                dominated = True
                break
        if not dominated:
            filtered.append(cs)
    
    candidates = filtered
    
    # Greedy set cover as fallback and initial solution
    uncovered = set(range(n))
    greedy_result = []
    while uncovered:
        best_set = None
        best_count = -1
        for cs in candidates:
            coverage = len(cs & uncovered)
            if coverage > best_count:
                best_count = coverage
                best_set = cs
        if best_set is None:
            break
        greedy_result.append(sorted(best_set))
        uncovered -= best_set
    
    best_count = len(greedy_result)
    best_parts = [list(p) for p in greedy_result]
    
    # Branch and bound on set cover
    def search_set_cover(uncovered, current_sets, depth):
        nonlocal best_parts, best_count, deadline
        if time.time() > deadline:
            return False
        if len(current_sets) >= best_count:
            return False
        if not uncovered:
            if len(current_sets) < best_count:
                best_parts = [list(p) for p in current_sets]
                best_count = len(current_sets)
            return True
        
        # Lower bound: at least ceil(len(uncovered) / max_candidate_size)
        max_size = max(len(cs) for cs in candidates if cs & uncovered)
        if max_size == 0:
            return False
        import math
        lb = math.ceil(len(uncovered) / max_size)
        if len(current_sets) + lb >= best_count:
            return False
        
        # Choose vertex with fewest candidate sets covering it
        v = min(uncovered, key=lambda v: sum(1 for cs in candidates if v in cs))
        
        # Try all candidates containing v
        for cs in candidates:
            if v in cs:
                new_uncovered = uncovered - cs
                if search_set_cover(frozenset(new_uncovered), current_sets + [list(cs)], depth + 1):
                    return True
        
        return False
    
    search_set_cover(frozenset(range(n)), [], 0)
    
    final_result = [[int(v) for v in part] for part in best_parts]
    return final_result
# EVOLVE-BLOCK-END
