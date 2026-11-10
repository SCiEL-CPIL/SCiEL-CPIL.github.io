import itertools

def construct():
    # Generate all forbidden triples in [1, 180]
    triples = set()
    N = 180
    # x, y must be coprime
    # a = kxy, b = kx(x+y), c = ky(x+y)
    # Max c is 180.
    # k * y * (x+y) <= 180
    # k >= 1, y >= 1, x >= 1
    # y(x+y) <= 180/k
    
    for k in range(1, N + 1):
        max_yxy = N // k
        for y in range(1, N + 1):
            if y * (y + 1) > max_yxy:
                break
            # x can be anything such that gcd(x,y)=1 and kxy <= N and kx(x+y) <= N
            # kx(x+y) <= N => x(x+y) <= N/k
            # x^2 + xy <= N/k
            # x is roughly sqrt(N/k)
            max_x_sq = N // k
            for x in range(1, N + 1):
                if x * (x + y) > max_yxy:
                    break
                if __import__('math').gcd(x, y) != 1:
                    continue
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a <= N and b <= N and c <= N:
                    # Check distinct
                    if len({a, b, c}) == 3:
                        triples.add(frozenset({a, b, c}))
                        
    # Initial set: [46, 180]
    S = set(range(46, 181))
    
    # Pre-compute conflicts for faster lookup?
    # Just iterate through triples and check if all in S
    
    # Local Search / Greedy Removal
    # To avoid infinite loops or slow convergence, we track removed elements
    
    removed = set()
    
    # We need a way to quickly find if a triple is fully in S
    # And to count degrees
    
    # Let's do a few passes of:
    # 1. Find a conflict
    # 2. Remove the "least useful" element
    
    # To speed up, we can map each number to the triples it belongs to
    num_to_triples = {i: [] for i in range(1, 181)}
    for t in triples:
        for v in t:
            num_to_triples[v].append(t)
            
    # Iterative removal
    # We'll use a simple heuristic: remove the element with the fewest conflicts in S
    # "Conflicts in S" = number of triples containing v where the other 2 elements are in S
    
    changed = True
    while changed:
        changed = False
        # Find any triple fully in S
        conflict = None
        for t in triples:
            if t.issubset(S):
                conflict = t
                break
                
        if conflict:
            # Choose which element to remove
            # Score: number of other elements in S that are in a triple with this element?
            # Simpler: number of triples containing this element that are fully in S?
            # If we remove v, we break all triples containing v.
            # We want to break as many as possible?
            # Actually, we want to remove the one that is "most redundant".
            # Let's remove the one with the highest degree in the conflict subgraph?
            # Or the one with the lowest?
            # Usually, removing the "leaf" (element in fewest conflicts) is safer for preserving size?
            # No, if an element is in 100 conflicts, removing it fixes 100.
            # If an element is in 1 conflict, removing it fixes 1.
            # So we should remove the one with the HIGHEST number of conflicts?
            
            counts = {}
            for v in conflict:
                count = 0
                for t2 in num_to_triples[v]:
                    if t2.issubset(S):
                        count += 1
                counts[v] = count
                
            # Remove the one with the max count
            v_to_remove = max(counts, key=counts.get)
            S.remove(v_to_remove)
            removed.add(v_to_remove)
            changed = True
            
    # Now S is valid.
    # Try to add back removed elements
    # Sort removed by some heuristic? Maybe just iterate.
    # If adding an element creates no conflict, add it.
    
    # To be efficient, we can check conflicts involving the new element.
    for v in list(removed):
        # Check if v can be added
        # v is in S? No, it was removed.
        # Check all triples containing v
        ok = True
        for t in num_to_triples[v]:
            # If the other two are in S, then adding v creates a conflict
            others = t - {v}
            if others.issubset(S):
                ok = False
                break
        if ok:
            S.add(v)
            removed.remove(v)
            
    return sorted(list(S))
