"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    from itertools import combinations
    from functools import lru_cache
    import sys
    sys.setrecursionlimit(10000)
    
    n = 128
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Build hyperedges: sets of 3 distinct elements {a,b,c} such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # This is equivalent to: the three pairwise LCMs are all equal.
    # Note: For three numbers, if lcm(a,b) = lcm(a,c) = lcm(b,c) = L,
    # then each pair's LCM is L.
    
    # Let's think about when this happens.
    # If a, b, c are all divisors of L, and for each pair, the LCM is exactly L.
    # This means that no single number contains all the prime power factors of L,
    # but any two together do.
    
    # Actually, a simpler characterization: 
    # lcm(a,b) = lcm(a,c) = lcm(b,c) iff for each prime p, the max exponent among a,b,c
    # is achieved by at least two of them (so that no single element alone achieves the max,
    # but any pair does). Wait, that's not quite right either.
    
    # Let me just brute-force check all triples. There are C(128,3) = 341,376 triples, which is fine.
    
    # For efficiency, let's precompute for each pair (i,j) the LCM, and then for each triple check.
    # But actually, we can be smarter. Let's just build the hypergraph.
    
    # To make backtracking efficient, we need:
    # 1. List of all hyperedges (forbidden triples)
    # 2. For each vertex, the set of hyperedges it belongs to
    # 3. For each pair of vertices, the hyperedges they share (if any)
    
    # Let's index vertices 0..127 corresponding to values 1..128
    
    # First, find all forbidden triples
    forbidden_triples = set()
    # For efficiency, precompute lcm for all pairs
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            lcm_table[i][j] = lcm(i+1, j+1)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all triples where the three pairwise LCMs are equal
    for a in range(n):
        for b in range(a+1, n):
            lab = lcm_table[a][b]
            for c in range(b+1, n):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    forbidden_triples.add((a, b, c))
    
    # Build adjacency structures for backtracking
    # For each vertex, list of hyperedges containing it
    vertex_to_edges = [[] for _ in range(n)]
    edge_index = 0
    edge_list = []
    for tri in forbidden_triples:
        edge_index += 1
        edge_list.append(tri)
        for v in tri:
            vertex_to_edges[v].append(edge_index - 1)
    
    # For each pair of vertices, list of hyperedges containing both
    # This is useful for forward checking
    pair_to_edges = {}
    for ei, tri in enumerate(edge_list):
        a, b, c = tri
        for x, y in combinations(tri, 2):
            key = (x, y) if x < y else (y, x)
            if key not in pair_to_edges:
                pair_to_edges[key] = []
            pair_to_edges[key].append(ei)
    
    # Backtracking with branch and bound
    # We want to select a maximum subset S of vertices such that no hyperedge is fully contained in S.
    # This is the maximum independent set in a 3-uniform hypergraph, or equivalently,
    # the maximum set with no complete hyperedge.
    
    # Strategy: try to include or exclude each vertex, using pruning.
    # Better: use a recursive approach where we decide on vertices in some order.
    
    # Let's use a different approach: 
    # We'll use a recursive backtracking that processes vertices in order.
    # At each step, we decide whether to include the current vertex or not.
    # We maintain the set of "active" hyperedges (those that have at least one vertex still undecided).
    # A hyperedge becomes "violated" if all its vertices are included.
    
    # Actually, let's think about this differently. The constraint is: for every forbidden triple (a,b,c),
    # we cannot have all three in S. So at least one of a, b, c must be excluded.
    
    # This is a 3-SAT-like problem. Let's use a simple backtracking with good ordering.
    
    # Key insight: if we process vertices in increasing order, and maintain which vertices are already selected,
    # then when considering vertex v, we can check if adding v would complete any forbidden triple with two already-selected vertices.
    
    # Let's use this approach:
    # - Process vertices in some order (we'll try a few orderings)
    # - Maintain selected set
    # - For each new vertex v, check if there exist a, b in selected such that (a,b,v) is a forbidden triple
    # - If yes, we must exclude v (or we could backtrack and try different choices, but greedy might work well)
    
    # Actually, for maximum cardinality, we need to be more careful. Let's use proper backtracking with pruning.
    
    # Let me try a simpler approach first: greedy with good ordering, then try to improve.
    
    # First, let's try a greedy approach: process vertices in some order, include if possible.
    # The order matters. A good heuristic: include vertices that are in fewer forbidden triples first? 
    # Or the opposite? Let's think...
    
    # If a vertex is in many forbidden triples, it's "dangerous" - including it forces us to exclude many others.
    # So we might want to exclude dangerous vertices and include safe ones.
    # Alternatively, we might want to include dangerous vertices early so that the constraints are satisfied.
    
    # Let's try: sort vertices by number of forbidden triples they're in, descending.
    # Try to include each vertex if it doesn't create a violation.
    
    # Count how many forbidden triples each vertex is in
    vertex_degree = [len(vertex_to_edges[i]) for i in range(n)]
    
    # Try different orderings
    best_solution = []
    best_size = 0
    
    # Ordering 1: by degree descending
    order1 = sorted(range(n), key=lambda x: -vertex_degree[x])
    
    # Ordering 2: by degree ascending  
    order2 = sorted(range(n), key=lambda x: vertex_degree[x])
    
    # Ordering 3: natural order
    order3 = list(range(n))
    
    # Ordering 4: by value (which is the same as natural order here since vertex i = value i+1)
    
    def greedy_solve(order):
        selected = []
        selected_set = set()
        # For fast checking, for each pair in selected, we need to know if adding v creates a forbidden triple
        # Precompute: for each pair (a,b), the list of c such that (a,b,c) is forbidden
        # We already have pair_to_edges, but we need the third element
        
        # Let's build: for each pair (a,b), the set of c values that form a forbidden triple with a,b
        pair_to_third = {}
        for a, b, c in forbidden_triples:
            for x, y in combinations([a,b,c], 2):
                key = (x, y) if x < y else (y, x)

                for z in [a,b,c]:
                    if z != x and z != y:
                        third = z
                        break
                if key not in pair_to_third:
                    pair_to_third[key] = set()
                pair_to_third[key].add(third)
        
        for v in order:
            # Check if adding v creates any forbidden triple with two already-selected vertices
            can_add = True
            for a in selected:
                key = (a, v) if a < v else (v, a)
                if key in pair_to_third:
                    for c in pair_to_third[key]:
                        if c in selected_set:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                selected.append(v)
                selected_set.add(v)
        
        return selected
    
    # Try greedy with different orderings
    for order in [order1, order2, order3]:
        sol = greedy_solve(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
    
    # Now try to improve with local search / backtracking
    # Let's try a more sophisticated backtracking approach
    
    # Actually, let's first see how good the greedy is. The theoretical maximum might be much higher.
    
    # Let's try a proper backtracking with branch and bound.
    # We'll use the following approach:
    # - Order vertices by some heuristic
    # - At each step, try to include the vertex, then exclude it
    # - Prune if the upper bound (current size + remaining vertices) <= best_size
    
    # For efficiency, we need good upper bounds. A simple one: current_size + number of remaining vertices.
    # A better one: current_size + maximum independent set of the remaining hypergraph, which is hard.
    # Let's start with the simple bound and see.
    
    # Let's use a recursive backtracking. To avoid recursion depth issues, we'll use iterative or careful recursion.
    
    # Given the size (128 vertices), and the fact that the hypergraph is sparse (we need to check how many forbidden triples there are),
    # let's first check the number of forbidden triples.
    
    num_forbidden = len(forbidden_triples)
    
    # If num_forbidden is small, backtracking should be feasible.
    
    # Let's try backtracking with the following strategy:
    # - Process vertices in order of decreasing degree (most constrained first)
    # - Use forward checking: when we include a vertex, mark all hyperedges containing it as "partially satisfied"
    # - When we exclude a vertex, remove it from consideration
    # - Prune if upper bound <= best
    
    # Actually, let's just try a simple recursive backtracking and see if it runs within time limits.
    
    # To make it faster, let's use bitmasks or sets efficiently.
    
    # Let me try a different approach: since the problem is small enough, let's use a MILP-like approach with Python's constraints,
    # or just a well-optimized backtracking.
    
    # Let's try backtracking with the following:
    # - Vertices ordered by degree descending
    # - State: current selected set, index in order
    # - Pruning: if len(selected) + (n - index) <= best_size, prune
    
    # This might still be too slow. Let's add more pruning:
    # - If including vertex v would create a violation, we must exclude it (no choice)
    # - This is a form of unit propagation
    
    # Let's implement this.
    
    # Precompute for each vertex v and each pair of previously-processed vertices, whether including v creates a violation.
    # This depends on the order, so let's do it dynamically.
    
    # Given time constraints, let's try the greedy solutions and also a simple local search.
    
    # Local search: start from greedy solution, try to add vertices that were excluded, by removing conflicting vertices.
    
    # Actually, let's just return the best greedy solution for now, and hope it's good.
    # But the task says to maximize, so let's try harder.
    
    # Let's try a proper backtracking with aggressive pruning.
    
    # Reorder vertices by degree descending
    bt_order = order1  # degree descending
    
    # Map from original index to position in bt_order
    pos_in_order = [0] * n
    for i, v in enumerate(bt_order):
        pos_in_order[v] = i
    
    # For backtracking, we need to quickly check if adding v to selected creates a forbidden triple.
    # We'll maintain selected as a set, and for each v, we can check against all pairs in selected.
    # But this is O(k^2) per vertex where k is the size of selected, which could be slow.
    
    # Alternative: maintain for each pair (a,b) in selected, the set of v that would create a violation.
    # When we add a new vertex w to selected, we update the conflict sets for all pairs involving w.
    
    # Let's use this approach:
    # conflict[v] = set of vertices u such that there exists w in selected with (u, w, v) being a forbidden triple
    # Actually, it's: for each v not in selected, conflict[v] = set of u in selected such that (u, v, w) is forbidden for some w in selected
    # No, that's not right either.
    
    # Let me think again. When we consider adding v to selected, we need to check if there exist a, b in selected
    # such that (a, b, v) is a forbidden triple. So we need: for each v, the set of pairs (a,b) in selected that with v form a forbidden triple.
    
    # Let's maintain: for each v, a counter or set of how many "witness" pairs exist in selected.
    
    # Actually, the simplest is: for each v, precompute the set of pairs (a,b) such that (a,b,v) is forbidden.
    # Then when checking if v can be added, we check if any such pair has both a and b in selected.
    
    # Let's build: for each v, list of (a,b) pairs such that {a,b,v} is a forbidden triple, with a < b.
    v_to_pairs = [[] for _ in range(n)]
    for a, b, c in forbidden_triples:
        for v in [a, b, c]:
            other1, other2 = [x for x in [a,b,c] if x != v]
            pair = (min(other1, other2), max(other1, other2))
            v_to_pairs[v].append(pair)
    
    # Now, for backtracking:
    # State: selected (set), index in bt_order
    # When considering vertex v = bt_order[index]:
    #   Check if v can be added: for each (a,b) in v_to_pairs[v], if a in selected and b in selected, then no.
    #   If can add, branch: include v, then exclude v.
    #   If cannot add, only exclude v.
    
    # This is still potentially exponential, but with pruning it might work.
    
    # Let's implement with memoization or just hope it's fast enough.
    
    # Given the 60-second time limit, let's try it with a timeout.
    
    import time
    start_time = time.time()
    timeout = 50  # seconds
    
    best_bt_solution = list(best_solution)
    best_bt_size = best_size
    
    # Use iterative deepening or just plain backtracking with pruning
    # Let's use plain backtracking
    
    def backtrack(selected, index, selected_set):
        nonlocal best_bt_solution, best_bt_size, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
        
        # Pruning
        if len(selected) + (n - index) <= best_bt_size:
            return
        
        if index == n:
            if len(selected) > best_bt_size:
                best_bt_size = len(selected)
                best_bt_solution = list(selected)
            return
        
        v = bt_order[index]
        
        # Check if v can be added
        can_add = True
        for a, b in v_to_pairs[v]:
            if a in selected_set and b in selected_set:
                can_add = False
                break
        
        # Branch 1: include v (if possible)
        if can_add:
            selected.append(v)
            selected_set.add(v)
            backtrack(selected, index + 1, selected_set)
            selected.pop()
            selected_set.remove(v)
        
        # Branch 2: exclude v
        backtrack(selected, index + 1, selected_set)
    
    # Run backtracking
    backtrack([], 0, set())
    
    if len(best_bt_solution) > len(best_solution):
        best_solution = best_bt_solution
    
    # Convert back to values (vertex i = value i+1)
    result = sorted([v + 1 for v in best_solution])
    
    return result
# EVOLVE-BLOCK-END
