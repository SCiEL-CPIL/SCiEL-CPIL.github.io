"""Rewritten program: branch-and-bound maximum independent set on the bad-triple hypergraph."""
# EVOLVE-BLOCK-START
def construct():
    from itertools import combinations
    from math import lcm
    import sys
    sys.setrecursionlimit(10000)

    n = 128
    # Precompute lcm for all pairs
    # lcm[i][j] for 0 <= i < j < n, where i corresponds to number i+1
    # We'll store as a dict or 2D list. Since n=128, 128*128 = 16384 entries, fine.
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[i][j] = lcm(i + 1, j + 1)
            lcm_table[j][i] = lcm_table[i][j]

    # Find all bad triples (i, j, k) with i < j < k such that
    # lcm(i+1, j+1) == lcm(i+1, k+1) == lcm(j+1, k+1)
    bad_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            l_ij = lcm_table[i][j]
            for k in range(j + 1, n):
                if lcm_table[i][k] == l_ij and lcm_table[j][k] == l_ij:
                    bad_triples.append((i, j, k))

    # Build adjacency structure for the hypergraph:
    # For each vertex v, store the set of other vertices u such that (v, u, w) is a bad triple for some w.
    # Actually, for branch-and-bound on independent set in a 3-uniform hypergraph, we need:
    # A set S is independent if no hyperedge is fully contained in S.
    # 
    # Strategy: branch-and-bound where we decide for each vertex whether to include or exclude it.
    # State: (vertex_index, current_selected_set, current_excluded_set)
    # But we can use bitsets for efficiency.
    #
    # Let's use a different approach: 
    # We process vertices in order. At each step, we decide to include or exclude the current vertex.
    # If we include vertex v, we must exclude all vertices that form a bad triple with v and any other included vertex.
    #
    # More precisely, maintain:
    # - included: set of vertices already included
    # - available: set of vertices that can still be included (not excluded, not yet processed)
    # - For each vertex v in available, we need to know if including v would create a bad triple with two already-included vertices.
    #
    # Actually, let me think more carefully. The constraint is: no three distinct elements whose three pairwise LCMs are equal.
    # So a triple (a,b,c) is bad iff lcm(a,b)==lcm(a,c)==lcm(b,c).
    #
    # When building the set incrementally, if we have already selected a set S, and we want to add x, we need to check
    # that there do not exist a, b in S such that (a, b, x) is a bad triple.
    #
    # So for each candidate x, the "conflict" with S is: does there exist a pair (a,b) in S such that (a,b,x) is bad?
    #
    # Let's precompute for each triple (i,j,k), and for each vertex, which pairs conflict.
    #
    # For branch-and-bound, let's use the following:
    # - Process vertices in some order (say, natural order 0 to n-1).
    # - State: (idx, selected_bitset)
    # - At step idx, we consider vertex idx. We can either include it or not.
    # - If we include it, we check that it doesn't form a bad triple with any pair in selected.
    # - If including it is valid, we branch: include or exclude.
    # - We use upper bound: len(selected) + (n - idx) as a naive bound, but we can do better.
    #
    # For a tighter bound, we can note that if we include vertex idx, it may force exclusion of some future vertices.
    # But for simplicity and speed, let's start with a greedy upper bound: 
    # upper_bound = len(selected) + (number of remaining vertices that don't conflict with selected in a way that prevents all from being added)
    #
    # Actually, a simpler upper bound: len(selected) + (n - idx). This is weak but simple.
    #
    # Let's try a better approach: use a greedy coloring or packing bound.
    # For now, let's implement a basic branch-and-bound with memoization on (idx, selected_bitset) but selected_bitset can be large.
    #
    # Alternative: use the fact that the hypergraph is sparse. Let's count bad_triples.
    
    num_bad = len(bad_triples)
    
    # For each vertex v, precompute the set of pairs (a,b) such that (a,b,v) is a bad triple with a<b<v or v<a<b or a<v<b.
    # Actually, let's precompute for each vertex v, the set of other vertices u such that there exists w with (min(u,v,w), ..., max) being a bad triple.
    # This is the "conflict graph" in a sense, but it's not a graph—it's a hypergraph.
    
    # Let's precompute: for each vertex v, a set of pairs (a, b) with a < b, a != v, b != v, such that (a, b, v) is a bad triple (after sorting).
    # Then, when we have selected set S, and we want to add v, we need to check: for any pair (a,b) in conflict_pairs[v], are both a and b in S?
    
    conflict_pairs = [set() for _ in range(n)]
    for (i, j, k) in bad_triples:
        # i < j < k
        # For vertex k: conflict pair (i, j)
        conflict_pairs[k].add((i, j))
        # For vertex j: conflict pair (i, k)
        conflict_pairs[j].add((i, k))
        # For vertex i: conflict pair (j, k)
        conflict_pairs[i].add((j, k))

    # Now, for branch-and-bound:
    # We'll use a recursive function that takes:
    #   idx: next vertex to consider
    #   selected: frozenset or bitset of selected vertices
    #   best: current best solution size (we track globally)
    #
    # To use bitsets efficiently, let's represent selected as an integer bitmask.
    # n=128, so we need 128 bits. Python ints can handle this.
    
    # Precompute for each vertex v, a bitmask of vertices that "conflict" with v in the sense that...
    # Actually, the conflict is pairwise: v conflicts with the pair (a,b) if (a,b,v) is bad.
    # So when adding v, we need to check if any conflict pair of v is fully contained in selected.
    
    # Let's precompute for each v, the list of conflict pairs as a list of (a, b).
    conflict_pairs_list = [list(s) for s in conflict_pairs]
    
    # Global best
    best_size = 0
    best_set = None
    
    # Upper bound function: given idx and selected_mask, what's the maximum additional vertices we can add?
    # Simple: n - idx. But we can do slightly better by checking which remaining vertices are "available".
    # A vertex u >= idx is available if it doesn't form a bad triple with any pair in selected.
    # But checking this for all u is expensive. Let's just use n - idx for now.
    
    def can_add(v, selected_mask):
        """Check if vertex v can be added to the set represented by selected_mask."""
        for (a, b) in conflict_pairs_list[v]:
            # Check if both a and b are in selected_mask
            if (selected_mask >> a) & 1 and (selected_mask >> b) & 1:
                return False
        return True
    
    # Branch and bound
    # Order: process vertices in natural order. We can also try a different order for better pruning.
    # Let's try processing in natural order first.
    
    def solve(idx, selected_mask, current_size):
        nonlocal best_size, best_set
        
        # Upper bound
        if current_size + (n - idx) <= best_size:
            return
        
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return
        
        v = idx
        # Option 1: exclude v
        solve(idx + 1, selected_mask, current_size)
        
        # Option 2: include v, if possible
        if can_add(v, selected_mask):
            new_mask = selected_mask | (1 << v)
            solve(idx + 1, new_mask, current_size + 1)
    
    # Start the search
    solve(0, 0, 0)
    
    # Convert best_set bitmask to list of numbers
    if best_set is None:
        # Should not happen, but fallback
        return [1]
    
    result = []
    for i in range(n):
        if (best_set >> i) & 1:
            result.append(i + 1)
    
    return result
# EVOLVE-BLOCK-END
