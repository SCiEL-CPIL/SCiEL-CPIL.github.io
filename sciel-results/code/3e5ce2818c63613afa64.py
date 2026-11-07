import random
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges
    orig_edge_set = set()
    for u, v in edges:
        if u == v: continue
        if u > v: u, v = v, u
        orig_edge_set.add((u, v))
        
    adj = [set() for _ in range(n)]
    for u, v in orig_edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Generate Candidate Bicliques
    # We will generate candidates by expanding from each edge with randomization.
    candidates = []
    candidate_sets = [] # List of sets of edges
    
    def get_edges(L, R):
        e_set = set()
        for u in L:
            for v in R:
                if u == v: continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in orig_edge_set:
                    e_set.add((a, b))
        return e_set

    def expand_biclique(L, R, seed):
        """Greedy expansion with random tie-breaking based on seed"""
        L = set(L)
        R = set(R)
        random.seed(seed)
        
        improved = True
        while improved:
            improved = False
            # Try adding to L
            pot_L = [v for v in range(n) if v not in L and adj[v].issuperset(R)]
            random.shuffle(pot_L)
            if pot_L:
                v = pot_L[0]
                L.add(v)
                improved = True
            # Try adding to R
            pot_R = [v for v in range(n) if v not in R and adj[v].issuperset(L)]
            random.shuffle(pot_R)
            if pot_R:
                v = pot_R[0]
                R.add(v)
                improved = True
        return L, R

    # Generate candidates
    seen_bicliques = set()
    for u, v in orig_edge_set:
        for seed in range(20):
            L, R = expand_biclique({u}, {v}, seed)
            # Canonical form for deduplication
            key = (frozenset(L), frozenset(R))
            # Also try swapping L and R? No, biclique is defined by L, R.
            # But [L, R] and [R, L] represent the same set of edges.
            # We just need the set of edges to be unique or the pair to be unique.
            # For exact cover, we just need the set of edges.
            e_set = get_edges(L, R)
            if e_set:
                e_key = frozenset(e_set)
                if e_key not in seen_bicliques:
                    seen_bicliques.add(e_key)
                    candidates.append((L, R))
                    candidate_sets.append(e_set)
                    
    # Also add single edge bicliques as fallback
    for u, v in orig_edge_set:
        e_set = {(u, v) if u < v else (v, u)}
        e_key = frozenset(e_set)
        if e_key not in seen_bicliques:
            seen_bicliques.add(e_key)
            L = {u}
            R = {v}
            candidates.append((L, R))
            candidate_sets.append(e_set)
            
    # 2. Solve Exact Cover
    # Universe: all edges
    universe = list(orig_edge_set)
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(universe)}
    
    # Matrix: rows are candidates, columns are edges
    # M[row][col] = 1 if candidate covers edge
    num_candidates = len(candidates)
    num_edges = len(universe)
    
    # We want to find a subset of rows that covers each column exactly once.
    # This is Exact Cover.
    
    # Use Algorithm X (Dancing Links) or simple backtracking.
    # Given small size, simple backtracking with heuristics might suffice.
    
    # Precompute for each edge, the list of candidates covering it
    edge_to_candidates = [[] for _ in range(num_edges)]
    for i, e_set in enumerate(candidate_sets):
        for e in e_set:
            idx = edge_to_idx[e]
            edge_to_candidates[idx].append(i)
            
    # Sort candidates by size (larger first) to help pruning?
    # Or just use the heuristic in the solver.
    
    best_solution = None
    best_size = num_candidates + 1
    
    # Heuristic: Pick the edge with fewest available candidates
    # State: used_candidates (set), covered_edges (set)
    
    import time
    start_time = time.time()
    
    def solve_exact_cover(available_candidates, covered_edges, current_solution):
        nonlocal best_solution, best_size
        if time.time() - start_time > 50: # Timeout guard
            return False
        
        if len(covered_edges) == num_edges:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = current_solution[:]
            return True
            
        # If current solution size + lower bound >= best_size, prune
        # Lower bound: 1 if any edges left
        if len(current_solution) >= best_size:
            return False
            
        # Find edge with fewest candidates
        min_count = float('inf')
        min_edge_idx = -1
        for e_idx in range(num_edges):
            if e_idx in covered_edges:
                continue
            # Count available candidates for this edge
            count = 0
            for c in edge_to_candidates[e_idx]:
                if c in available_candidates:
                    count += 1
            if count == 0:
                return False # Dead end
            if count < min_count:
                min_count = count
                min_edge_idx = e_idx
                
        if min_edge_idx == -1:
            return False # No uncovered edges? Should be caught above
            
        # Try each candidate for min_edge_idx
        # Order candidates by size (larger first) to find good solutions faster?
        # Or smaller? Larger covers more, might lead to smaller count.
        cands = []
        for c in edge_to_candidates[min_edge_idx]:
            if c in available_candidates:
                cands.append(c)
                
        # Sort by size of candidate (descending)
        cands.sort(key=lambda c: len(candidate_sets[c]), reverse=True)
        
        for c in cands:
            # Check if candidate c is compatible with current covered_edges
            # i.e., c does not overlap with covered_edges
            if candidate_sets[c].isdisjoint(covered_edges):
                new_covered = covered_edges.union(candidate_sets[c])
                new_avail = available_candidates - {c}
                # Also, we should remove any other candidates that overlap with c?
                # No, we just remove c from available.
                # The disjunction check ensures we don't pick overlapping ones later.
                # But wait, if we pick c, we cannot pick any candidate that overlaps with c.
                # So we should remove all candidates that overlap with c from available_candidates.
                new_avail = set()
                for c2 in available_candidates:
                    if c2 != c and candidate_sets[c2].isdisjoint(candidate_sets[c]):
                        new_avail.add(c2)
                        
                current_solution.append(c)
                if solve_exact_cover(new_avail, new_covered, current_solution):
                    return True
                current_solution.pop()
                
        return False
        
    initial_avail = set(range(num_candidates))
    solve_exact_cover(initial_avail, set(), [])
    
    if best_solution is None:
        # Fallback: Greedy
        # This shouldn't happen if we have single-edge bicliques
        return []
        
    # Reconstruct result
    result = []
    for c in best_solution:
        L, R = candidates[c]
        result.append([list(L), list(R)])
        
    return result

