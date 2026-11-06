"""Rewritten program: maximize subset of [1,180] with no distinct triple a,b,c s.t. 1/a=1/b+1/c."""
# EVOLVE-BLOCK-START
def construct():
    N = 180
    
    # 1. Identify all forbidden triples
    forbidden = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    forbidden.add(frozenset((a, b, c)))
    
    # 2. Core insight: No three distinct odd numbers form a triple.
    #    Proof: 1/a = 1/b + 1/c => c = ab/(b-a). If a,b odd, b-a even, ab odd => c not integer.
    #    Thus, all odd numbers can be safely included.
    odds = set(x for x in range(1, N + 1) if x % 2 != 0)
    evens = set(x for x in range(1, N + 1) if x % 2 == 0)
    
    # 3. Pre-compute conflicts for evens.
    #    We want to greedily add evens to the set of odds.
    #    A triple is invalid if all 3 elements are in the set.
    #    Since all odds are in the set, we must avoid adding evens that form a triple with:
    #      a) Two existing evens (Triple: E, E, E)
    #      b) One existing even and one odd (Triple: O, E, E)
    
    #    For each even e, we can determine which other evens e' it "conflicts" with.
    #    Conflict definition: If e and e' are both in the set, do they form a triple with some x?
    #    If x is odd: x is always in the set. So if (x, e, e') is a triple, we cannot have both e and e'.
    #    If x is even: x might or might not be in the set. So this is a 3-way constraint.
    
    #    To simplify greedy selection, we prioritize avoiding 2-way conflicts (with odds).
    #    We build a graph on evens where an edge (e, e') exists if there is an odd o such that {o, e, e'} is a forbidden triple.
    
    conflict_graph = {e: set() for e in evens}
    for tri in forbidden:
        elems = list(tri)
        # Count odds and evens in the triple
        num_odds = sum(1 for x in elems if x % 2 != 0)
        num_evens = sum(1 for x in elems if x % 2 == 0)
        
        if num_odds == 1 and num_evens == 2:
            # Type: Odd, Even, Even
            # The two evens cannot both be selected.
            e1 = [x for x in elems if x % 2 == 0][0]
            e2 = [x for x in elems if x % 2 == 0][1]
            conflict_graph[e1].add(e2)
            conflict_graph[e2].add(e1)
            
        # Note: Triples with 3 evens or 1 odd 2 evens are handled.
        # Triples with 3 odds or 2 odds 1 even are impossible.
        
    # 4. Greedy selection of evens
    #    Sort evens by degree in conflict_graph (ascending).
    #    Prefer evens with fewer 2-way conflicts.
    sorted_evens = sorted(evens, key=lambda e: len(conflict_graph[e]))
    
    selected_evens = set()
    current_set = odds.copy()
    
    for e in sorted_evens:
        # Check if adding e to current_set creates any forbidden triple.
        # We only need to check triples involving e.
        # Triple types involving e:
        # - (o, e, e') where o is odd, e' is even.
        #   This is covered by conflict_graph if e' is in selected_evens.
        #   But we also need to ensure e doesn't form a triple with an odd o and an even e' that is NOT in selected_evens but might be?
        #   No, we are building the set incrementally. We only care about the current state.
        #   If e' is not in selected_evens, it's not in current_set, so the triple is not formed.
        # - (e1, e2, e) where e1, e2 are evens.
        #   We need to check if there exist e1, e2 in selected_evens such that {e, e1, e2} is a forbidden triple.
        
        valid = True
        
        # Check 2-way conflicts (with odds)
        # If e has a conflict with e' in selected_evens, invalid.
        for e_prime in selected_evens:
            if e_prime in conflict_graph[e]:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 3-way conflicts (with evens)
        # Need to check if any pair in selected_evens forms a triple with e.
        # To be efficient, we can pre-compute for each e, the set of pairs (e1, e2) that form a triple with e.
        # Or just iterate over selected_evens pairs? That's O(K^2). K is at most 90. 90^2 = 8100.
        # Total evens 90. 90 * 8100 = 729,000 checks. Fast enough.
        
        # Optimization: Pre-compute "even_triplets" for each e.
        # Let's just do the check on the fly for simplicity, as N is small.
        
        # Actually, we can pre-compute a set of frozensets for evens.
        # Let's refine:
        # For each e, find all triples in `forbidden` containing e.
        # Filter those where the other two are evens.
        
        # To avoid O(K^2) in the loop, let's pre-compute a dictionary:
        # even_conflicts[e] = set of frozensets({e1, e2}) such that {e, e1, e2} is a forbidden triple.
        
        pass # Placeholder, we will implement this in the code below
        
    # Re-implementing the loop with pre-computation for efficiency
    
    even_pairs_conflict = {e: set() for e in evens}
    for tri in forbidden:
        elems = list(tri)
        num_evens = sum(1 for x in elems if x % 2 == 0)
        if num_evens == 3:
            e1, e2, e3 = [x for x in elems if x % 2 == 0]
            # Add pair (e2, e3) to conflicts of e1, etc.
            even_pairs_conflict[e1].add(frozenset((e2, e3)))
            even_pairs_conflict[e2].add(frozenset((e1, e3)))
            even_pairs_conflict[e3].add(frozenset((e1, e2)))
            
    # Now the greedy loop
    selected_evens = set()
    
    for e in sorted_evens:
        valid = True
        
        # Check 2-way conflicts (Odd, Even, Even)
        for e_prime in selected_evens:
            if e_prime in conflict_graph[e]:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 3-way conflicts (Even, Even, Even)
        # Check if any pair in selected_evens is in even_pairs_conflict[e]
        # To do this efficiently, we can check if any e_prime in selected_evens
        # forms a valid pair with some other e_second in selected_evens.
        # Simple check:
        for pair in even_pairs_conflict[e]:
            # pair is a frozenset of two evens
            if pair.issubset(selected_evens):
                valid = False
                break
        
        if valid:
            selected_evens.add(e)
            
    final_set = odds | selected_evens
    return sorted(final_set)
# EVOLVE-BLOCK-END
