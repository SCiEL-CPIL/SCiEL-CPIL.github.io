"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    from itertools import combinations
    import time
    
    n = 128
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute LCM table
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[i][j] = lcm(i + 1, j + 1)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all forbidden triples: {a,b,c} where lcm(a,b)==lcm(a,c)==lcm(b,c)
    forbidden_triples = []
    for a in range(n):
        for b in range(a + 1, n):
            lab = lcm_table[a][b]
            for c in range(b + 1, n):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    forbidden_triples.append((a, b, c))
    
    # Build conflict structure using bitmasks for speed
    # For each vertex v, conflict_mask[v] is a bitmask of vertices u such that
    # there exists w where {u, w, v} is a forbidden triple.
    # Actually, we need: for each v, the set of pairs (a,b) such that {a,b,v} is forbidden.
    # Better approach: for each vertex v, precompute which pairs of other vertices conflict with v.
    
    # Let's use: for each vertex v, conflict_pairs[v] = list of (a, b) where a < b, a != v, b != v,
    # and {a, b, v} is a forbidden triple.
    
    conflict_pairs = [[] for _ in range(n)]
    for a, b, c in forbidden_triples:
        for v in (a, b, c):
            others = [x for x in (a, b, c) if x != v]
            pair = (min(others), max(others))
            conflict_pairs[v].append(pair)
    
    # Degree: number of forbidden triples each vertex is in
    degree = [len(conflict_pairs[i]) for i in range(n)]
    
    # Greedy initialization with multiple orderings
    def greedy_solve(order):
        selected = []
        selected_set = set()
        for v in order:
            can_add = True
            for a, b in conflict_pairs[v]:
                if a in selected_set and b in selected_set:
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                selected_set.add(v)
        return selected
    
    # Try several orderings for greedy
    best_solution = []
    best_size = 0
    
    orderings = [
        sorted(range(n), key=lambda x: -degree[x]),  # high degree first
        sorted(range(n), key=lambda x: degree[x]),    # low degree first
        list(range(n)),                                # natural
        sorted(range(n), key=lambda x: (degree[x], x)),  # degree then index
        sorted(range(n), key=lambda x: (-degree[x], x)), # -degree then index
    ]
    
    for order in orderings:
        sol = greedy_solve(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
    
    # Backtracking with branch and bound and forward checking
    # Use bitmask representation for selected set for fast membership checks
    
    # Order vertices for backtracking: most constrained first (highest degree)
    bt_order = sorted(range(n), key=lambda x: -degree[x])
    
    # Map: position in bt_order -> vertex
    # vertex -> position in bt_order
    pos_map = {v: i for i, v in enumerate(bt_order)}
    
    # Precompute for each vertex v, the conflict pairs sorted by position in bt_order
    # This helps with forward checking
    
    # For backtracking, we maintain:
    # - selected_mask: bitmask of selected vertices
    # - current_size: number of selected vertices
    # - index: current position in bt_order
    
    # Forward checking: when we select vertex v, any vertex u that has a conflict pair (a,b)
    # where a and b are both already selected (or one is v and the other is selected) becomes blocked.
    
    # Actually, let's use a simpler but effective approach:
    # Maintain blocked_mask: bitmask of vertices that cannot be selected because
    # they would form a forbidden triple with two already-selected vertices.
    
    # When we select vertex v:
    #   For each conflict pair (a, b) of v:
    #     If a is selected and b is not blocked, block b
    #     If b is selected and a is not blocked, block a
    #     If neither is selected, no immediate blocking (but when one gets selected later, the other gets blocked)
    
    # This is a form of arc consistency for the 3-uniform hypergraph.
    
    # Let's implement:
    # State: index, selected_mask, blocked_mask, current_size
    # At each step, pick the next vertex from bt_order[index:] that is not blocked.
    # Branch: include or exclude.
    
    # For efficiency, we'll use a recursive approach with careful state management.
    
    start_time = time.time()
    timeout = 55  # seconds, leave margin
    
    best_bt_size = best_size
    best_bt_solution = list(best_solution)
    
    # Use a stack-based approach to avoid deep recursion issues
    # State: (index, selected_mask, blocked_mask, current_size)
    
    # We'll process vertices in bt_order order.
    # At each step, we look at bt_order[index].
    # If it's blocked, we skip it (must exclude).
    # If it's not blocked, we try include then exclude.
    
    # Upper bound: current_size + number of unblocked vertices from index to end
    # We can precompute suffix counts, but blocked_mask changes dynamically.
    
    # Let's use a simple but effective bound: current_size + (n - index)
    # And add: if current_size + (n - index) <= best_bt_size, prune.
    
    # To make this faster, let's also try to find a good initial solution.
    
    # Enhanced greedy: after initial greedy, try to add blocked vertices by removing conflicting ones.
    # This is local search.
    
    # Let's implement the backtracking:
    
    def backtrack(index, selected_mask, blocked_mask, current_size):
        nonlocal best_bt_size, best_bt_solution, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
        
        # Pruning: simple bound
        if current_size + (n - index) <= best_bt_size:
            return
        
        if index == n:
            if current_size > best_bt_size:
                best_bt_size = current_size
                # Reconstruct solution
                sol = []
                for i in range(n):
                    if (selected_mask >> i) & 1:
                        sol.append(i)
                best_bt_solution = sol
            return
        
        v = bt_order[index]
        
        # If v is blocked, we must exclude it
        if (blocked_mask >> v) & 1:
            backtrack(index + 1, selected_mask, blocked_mask, current_size)
            return
        
        # Try including v
        new_selected_mask = selected_mask | (1 << v)
        new_blocked_mask = blocked_mask
        
        # Update blocked_mask: for each conflict pair (a, b) of v,
        # if a is selected, block b; if b is selected, block a
        for a, b in conflict_pairs[v]:
            if (selected_mask >> a) & 1:
                # a is selected, so b is now blocked
                new_blocked_mask |= (1 << b)
            if (selected_mask >> b) & 1:
                # b is selected, so a is now blocked
                new_blocked_mask |= (1 << a)
        
        backtrack(index + 1, new_selected_mask, new_blocked_mask, current_size + 1)
        
        # Try excluding v
        backtrack(index + 1, selected_mask, blocked_mask, current_size)
    
    # Run backtracking
    backtrack(0, 0, 0, 0)
    
    if best_bt_size > best_size:
        best_solution = best_bt_solution
        best_size = best_bt_size
    
    # Local search: try to improve by swapping vertices
    # For each vertex not in solution, try adding it and removing conflicting vertices
    selected_set = set(best_solution)
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for v in range(n):
            if v in selected_set:
                continue
            # Check if v can be added
            can_add = True
            for a, b in conflict_pairs[v]:
                if a in selected_set and b in selected_set:
                    can_add = False
                    break
            if can_add:
                selected_set.add(v)
                improved = True
                break
    
    # Try to remove and re-add to find better solutions
    # Simple local search: try removing one vertex and adding up to two others
    for _ in range(100):  # limited iterations
        if time.time() - start_time > timeout:
            break
        best_local = list(selected_set)
        best_local_size = len(selected_set)
        
        for v in list(selected_set):
            # Remove v
            test_set = set(selected_set)
            test_set.remove(v)
            
            # Try to add vertices
            for u in range(n):
                if u in test_set:
                    continue
                can_add = True
                for a, b in conflict_pairs[u]:
                    if a in test_set and b in test_set:
                        can_add = False
                        break
                if can_add:
                    test_set.add(u)
            
            if len(test_set) > best_local_size:
                best_local_size = len(test_set)
                best_local = list(test_set)
        
        if best_local_size > len(selected_set):
            selected_set = set(best_local)
            improved = True
        else:
            break
    
    best_solution = list(selected_set)
    best_size = len(best_solution)
    
    # Convert to values (vertex i = value i+1)
    result = sorted([v + 1 for v in best_solution])
    
    return result
# EVOLVE-BLOCK-END
