import itertools

def construct():
    # B is fixed to [65, 128]
    B = set(range(65, 129))
    
    # We want to find max A subset of [1, 64]
    # such that no a,b,c in A have a+b, a+c, b+c in A U B
    
    # Optimization: 
    # If a+b > 128, it's not in S.
    # If 65 <= a+b <= 128, it IS in S (since B is full).
    # If a+b <= 64, it is in S iff x_{a+b} is 1.
    
    # Let's use a greedy approach with local search or just check a few promising sets.
    # Promising set: Odds in [1, 64] plus some evens.
    
    # Let's try to build A greedily.
    # Start with empty set, add elements that don't create conflicts.
    # Or start with a large set and remove conflicts.
    
    # Let's try starting with all odds in [1, 64] (size 32) and see if we can keep it.
    # Odds sum to even.
    # If A is all odds, A+A is evens.
    # Evens are not in A.
    # So a+b in S iff a+b in B.
    # So we need to ensure no a,b,c in A (odds) such that a+b, a+c, b+c are all in B.
    # a+b, a+c, b+c are even.
    # B contains evens 66, 68, ..., 128.
    # So we need to avoid triangles in the graph G on odds where edge (a,b) exists if a+b in B.
    # B evens: 66..128.
    # Min sum of distinct odds: 1+3=4.
    # Max sum: 63+61=124.
    
    # Let's check if the set of all odds in [1, 64] works.
    # A = {1, 3, ..., 63}.
    # Check if there exist a,b,c in A such that a+b, a+c, b+c in B.
    # This is equivalent to finding a triangle in G.
    # G is defined on 32 vertices.
    # Edge (a,b) if a+b in {66, 68, ..., 128}.
    
    # Let's test a specific triangle:
    # a=31, b=33, c=35.
    # 31+33=64 (not in B).
    # 31+35=66 (in B).
    # 33+35=68 (in B).
    # Not a triangle because 31+33 not in B.
    
    # a=33, b=35, c=37.
    # 33+35=68 (in B).
    # 33+37=70 (in B).
    # 35+37=72 (in B).
    # Triangle!
    # So A = all odds is NOT valid.
    
    # We must remove vertices to break all triangles.
    # This is the Minimum Vertex Cover of the hypergraph of triangles?
    # Or just find a large independent set in the "triangle conflict" graph?
    # Actually, we want to keep as many as possible.
    
    # Let's use a simple backtracking or greedy removal.
    # Start with A = odds in [1, 64].
    # Find a triangle, remove one vertex, repeat.
    
    A_candidates = set(range(1, 65, 2))
    
    def has_conflict(A):
        # Check if any a,b,c in A have sums in A U B
        # S = A U B
        S = A | B
        # To be efficient, only check triples where sums are likely in S
        # a,b,c distinct
        items = list(A)
        n = len(items)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = items[i], items[j], items[k]
                    if a+b in S and a+c in S and b+c in S:
                        return (a, b, c)
        return None
    
    # Greedy removal
    # This might be slow if we re-check all triples.
    # But 32 items, 5984 triples. Fast enough.
    
    A = set(A_candidates)
    changed = True
    while changed:
        changed = False
        conflict = has_conflict(A)
        if conflict:
            # Remove one of them. Which one?
            # Remove the one that is in the most conflicts?
            # Or just remove the first one.
            # Let's try removing the median one or random.
            # For determinism, remove the largest?
            a, b, c = conflict
            # Remove c
            A.remove(c)
            changed = True
            
    # Now try to add back evens?
    # Evens in [1, 64].
    # Adding an even e to A.
    # Check if it creates new conflicts.
    for e in range(2, 65, 2):
        if e in A:
            continue
        new_A = A | {e}
        if has_conflict(new_A) is None:
            A = new_A

    # Final check
    if has_conflict(A) is not None:
        # Should not happen
        pass

    return list(A) + list(B)
