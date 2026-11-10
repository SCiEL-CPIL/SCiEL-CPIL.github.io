import time

def construct():
    start_time = time.time()
    limit = 180
    deadline = start_time + 80  # safety margin

    # Precompute all forbidden triples (a, b, c) with distinct elements in [1,180]
    # where 1/a = 1/b + 1/c.
    # For each pair (a, b) with a < b, c = a*b/(b-a). Check if c is integer and in range and distinct.
    # Store for each element x, the set of (y, z) pairs such that x is part of a forbidden triple.
    # Actually, for local search we need: given a set S, check if adding x creates a triple.
    # A triple (a,b,c) is forbidden if all three are in S.
    # For adding x: check all pairs (y,z) in S such that (x,y,z) forms a triple.
    # Precompute for each x, list of (y,z) pairs where {x,y,z} is a forbidden triple.
    
    # Build a map: for each x in [1,180], list of (y,z) pairs (y,z both in [1,180], y!=z, y!=x, z!=x)
    # such that 1/x = 1/y + 1/z (i.e., x,y,z form a reciprocal Schur triple)
    # To check if x conflicts with S: for each (y,z) in triples[x], if y in S and z in S, conflict.
    
    triples = [[] for _ in range(limit + 1)]  # triples[x] = list of (y,z)
    
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= limit and c != a and c != b:
                    # triple (a, b, c): 1/a = 1/b + 1/c
                    # a is the smallest, b and c are larger
                    # Store for each element the pairs of other elements
                    triples[a].append((b, c))
                    triples[b].append((a, c))
                    triples[c].append((a, b))
    
    def has_conflict(x, S_set):
        """Check if adding x to S_set creates a forbidden triple."""
        for (y, z) in triples[x]:
            if y in S_set and z in S_set:
                return True
        return False
    
    # Greedy construction from largest to smallest
    S = []
    S_set = set()
    
    for a in range(limit, 0, -1):
        if time.time() > deadline:
            break
        if not has_conflict(a, S_set):
            S.append(a)
            S_set.add(a)
    
    # Local search: try to add elements not in S
    # Try each element not in S, see if it can be added
    best_S = S[:]
    best_S_set = S_set.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, limit + 1):
            if time.time() > deadline:
                break
            if x in best_S_set:
                continue
            if not has_conflict(x, best_S_set):
                best_S.append(x)
                best_S_set.add(x)
                improved = True
    
    # Local search: try swapping - remove one element, add another
    # For each element in S, try removing it and then adding a different element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best_S):
            if time.time() > deadline:
                break
            # Try removing x
            test_set = best_S_set - {x}
            # Try adding each element not in test_set
            for y in range(1, limit + 1):
                if time.time() > deadline:
                    break
                if y in test_set:
                    continue
                if not has_conflict(y, test_set):
                    # Check if this swap improves: we removed x and added y
                    # Only useful if y > x (we want larger numbers generally)
                    # Actually, cardinality stays same, so this is neutral
                    # But it might enable future additions
                    # Let's only do swaps that allow adding more
                    new_set = test_set | {y}
                    # Check if we can now add more elements
                    extra = 0
                    for z in range(1, limit + 1):
                        if z in new_set:
                            continue
                        if not has_conflict(z, new_set):
                            extra += 1
                    if extra > 0:
                        # Do the swap
                        best_S.remove(x)
                        best_S_set.remove(x)
                        best_S.append(y)
                        best_S_set.add(y)
                        improved = True
                        break
    
    # Final greedy pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, limit + 1):
            if time.time() > deadline:
                break
            if x in best_S_set:
                continue
            if not has_conflict(x, best_S_set):
                best_S.append(x)
                best_S_set.add(x)
                improved = True
    
    # Ensure nonempty
    if not best_S:
        return [limit]
    
    return best_S
