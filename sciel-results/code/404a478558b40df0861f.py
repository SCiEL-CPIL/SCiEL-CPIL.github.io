import time
import sys

def construct():
    deadline = time.time() + 55
    
    # 1. Generate triples
    triples = []
    # x < y
    for x in range(1, 181):
        for y in range(x+1, 181):
            max_k = 180 // (y * (x + y))
            for k in range(1, max_k + 1):
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a <= 180 and b <= 180 and c <= 180:
                    # Check distinctness
                    if len({a, b, c}) == 3:
                        triples.append((a, b, c))
    
    # Remove duplicate triples if any (though generation logic should prevent most)
    # Convert to set of tuples for O(1) lookup? Or keep list.
    # For B&B, we need to know conflicts quickly.
    
    # 2. Identify V
    V = set()
    for t in triples:
        V.update(t)
    
    # 3. Elements not in V are always included
    safe_elements = [i for i in range(1, 181) if i not in V]
    
    if not V:
        return safe_elements
    
    v_list = sorted(V)
    v_index = {v: i for i, v in enumerate(v_list)}
    n_vars = len(v_list)
    
    # Build conflict structure
    # For each variable i, list of triples it is in.
    # Better: For each pair (i, j), if they are in a triple with k, then selecting i and j forces k to be excluded?
    # No, selecting i and j is allowed ONLY IF k is not selected.
    # So, if we select i, and we later select j, we must exclude k.
    
    # Let's use a B&B that maintains:
    # - selected: set of indices
    # - excluded: set of indices
    # - unassigned: set of indices
    
    # To speed up, we precompute for each i, the list of (j, k) pairs such that (i, j, k) is a triple.
    # Actually, for each i, we want to know: if i is selected, which other elements become "dangerous"?
    # If i is selected, and j is selected, then k must be excluded.
    
    # Precomputation:
    # conflicts[i] = list of (j, k) where {i, j, k} is a triple and j != i, k != i.
    # Note: if i is selected, and j is in selected, then k must be excluded.
    
    conflicts = [[] for _ in range(n_vars)]
    for t in triples:
        a, b, c = t
        ia, ib, ic = v_index[a], v_index[b], v_index[c]
        # Add to ia's list
        conflicts[ia].append((ib, ic))
        # Add to ib's list
        conflicts[ib].append((ia, ic))
        # Add to ic's list
        conflicts[ic].append((ia, ib))
    
    # Sort variables by degree (number of conflicts) to pick branching variable
    degrees = [len(conflicts[i]) for i in range(n_vars)]
    var_order = sorted(range(n_vars), key=lambda x: degrees[x], reverse=True)
    
    best_set = set()
    best_size = 0
    
    # Initial greedy heuristic to get a good lower bound
    # Sort by degree ascending? Or descending?
    # Usually, picking low degree first allows more freedom.
    # Let's try a simple greedy: pick variable with lowest degree among unassigned, add it, remove its conflicts from consideration?
    # No, that's for MIS in graphs. Here it's hypergraph.
    # Greedy: 
    # 1. Start with all unassigned.
    # 2. Pick v with min degree.
    # 3. Select v.
    # 4. For any triple (v, u, w), we cannot select both u and w. This is a constraint, not a removal.
    # This greedy is tricky. Let's just use a random or simple heuristic for initial bound.
    # Or just start B&B.
    
    # To make B&B fast, we need a good upper bound.
    # Upper bound = len(selected) + len(unassigned)
    
    # We will use a recursive function.
    # To avoid recursion depth issues, we can use an explicit stack or ensure N is small enough.
    # N is at most 180, but likely much less.
    
    # Let's implement a simple B&B.
    
    def solve_bnb(selected, excluded, unassigned):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning
        if len(selected) + len(unassigned) <= best_size:
            return
        
        if not unassigned:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
            return
        
        # Pick a variable to branch on
        # Heuristic: Pick variable in unassigned that is involved in the most "active" conflicts?
        # Or just pick the first one in var_order that is in unassigned.
        # To be more effective, pick the one with highest degree in the *remaining* graph?
        # That's expensive. Let's stick to global degree order.
        
        branch_var = -1
        for v in var_order:
            if v in unassigned:
                branch_var = v
                break
        
        if branch_var == -1:
            return
            
        # Option 1: Select branch_var
        # Check if it's safe to select?
        # It's safe if it doesn't form a triple with two already selected vars.
        # If it does, we can't select it.
        can_select = True
        forced_exclusions = set()
        
        for (j, k) in conflicts[branch_var]:
            if j in selected and k in selected:
                can_select = False
                break
            elif j in selected:
                forced_exclusions.add(k)
            elif k in selected:
                forced_exclusions.add(j)
        
        if can_select:
            new_selected = selected | {branch_var}
            new_excluded = excluded | forced_exclusions
            # Also, branch_var itself is removed from unassigned
            new_unassigned = unassigned - {branch_var}
            # Remove forced exclusions from unassigned
            new_unassigned -= forced_exclusions
            
            solve_bnb(new_selected, new_excluded, new_unassigned)
        
        # Option 2: Exclude branch_var
        new_excluded = excluded | {branch_var}
        new_unassigned = unassigned - {branch_var}
        
        solve_bnb(selected, new_excluded, new_unassigned)

    # Initialize
    initial_unassigned = set(range(n_vars))
    initial_selected = set()
    initial_excluded = set()
    
    # Run B&B
    solve_bnb(initial_selected, initial_excluded, initial_unassigned)
    
    # Map back to values
    selected_v = [v_list[i] for i in best_set]
    result = safe_elements + selected_v
    return result
