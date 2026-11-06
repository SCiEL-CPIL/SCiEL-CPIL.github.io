import random

def construct():
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> bc = a(b+c)
    # a = bc / (b+c)
    # Iterate b, c
    forbidden = set()
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b:
                    forbidden.add((a, b, c))
    
    # For fast lookup, we can store conflicts in a structure.
    # For each x, store a list of (u, v) such that (x, u, v) is a forbidden triple (with x being the smallest)
    # Or simply, for each x, store a set of pairs (u,v) that are "forbidden together with x".
    # Actually, we just need to check if adding x creates ANY triple.
    # A triple is formed if x is the smallest (a), middle (b), or largest (c).
    
    # Let's precompute for each x, the set of "forbidden pairs" in the other two positions.
    # If x is a: pairs (b, c)
    # If x is b: pairs (a, c)
    # If x is c: pairs (a, b)
    
    conflicts = [set() for _ in range(N + 1)]
    for (a, b, c) in forbidden:
        # x=a conflicts with (b,c)
        conflicts[a].add((b, c))
        # x=b conflicts with (a,c)
        conflicts[b].add((a, c))
        # x=c conflicts with (a,b)
        conflicts[c].add((a, b))

    def can_add(x, S):
        # Check if adding x to S creates a forbidden triple
        # We need to check if there exist u, v in S such that (x, u, v) is a forbidden triple (in any order)
        # Our conflicts[x] contains pairs (u,v) such that {x,u,v} is forbidden.
        # So we just check if any pair in conflicts[x] is a subset of S.
        for (u, v) in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    best_set = set()
    
    # Run greedy multiple times with different random orders
    # Use a fixed seed for reproducibility? The task says "Maximize cardinality", usually deterministic is preferred unless randomness helps explore.
    # Given the small N, we can try many permutations.
    
    for _ in range(100): # 100 restarts
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        S = set()
        for x in elements:
            if can_add(x, S):
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = S
    
    # Try to improve best_set by adding any remaining elements
    # This is a simple "greedy fill" on the result
    for x in range(1, N + 1):
        if x not in best_set:
            if can_add(x, best_set):
                best_set.add(x)

    # One more pass to see if we can add more
    for x in range(1, N + 1):
        if x not in best_set:
            if can_add(x, best_set):
                best_set.add(x)

    return list(best_set)
