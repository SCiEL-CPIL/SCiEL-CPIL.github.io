import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin
    
    N = 180
    elements = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # 1/a = 1/b + 1/c  =>  c = ab / (b - a)
    # We need distinct a, b, c.
    # Since 1/a > 1/b, a < b. Similarly a < c.
    # So a is the smallest element in the triple.
    triples = []
    element_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Ensure uniqueness and sorted order for storage
                    t = tuple(sorted((a, b, c)))
                    # Check if we already have this triple
                    # Using a set of tuples for fast lookup
                    pass
                    
    # Better precomputation to avoid duplicates easily
    triple_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = tuple(sorted((a, b, c)))
                    triple_set.add(t)
                    
    triples = list(triple_set)
    
    # Build index: element -> list of triple indices containing it
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for x in t:
            elem_triples[x].append(idx)
            
    def can_add(S, x):
        """Check if x can be added to S without forming a forbidden triple."""
        if x in S:
            return False
        # Check all triples containing x
        for idx in elem_triples[x]:
            t = triples[idx]
            # If the other two elements are in S, we cannot add x
            if all(y in S for y in t if y != x):
                return False
        return True

    def local_search(S):
        """Try to add any valid element to S until no more can be added."""
        S = set(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try adding elements not in S
            # Random order to break symmetry? Or sorted?
            # Sorted might be faster if we check from 1 to N
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        changed = True
        return S

    def initial_greedy():
        """Start with all elements, remove those causing conflicts."""
        S = set(range(1, N + 1))
        # Count conflicts for each element
        conflict_count = {i: 0 for i in range(1, N + 1)}
        
        # Identify active triples
        active_triples = set()
        for idx, t in enumerate(triples):
            if all(x in S for x in t):
                active_triples.add(idx)
                for x in t:
                    conflict_count[x] += 1
                    
        while active_triples and time.time() < deadline:
            # Find element with max conflict
            max_conf = -1
            target = -1
            for x in range(1, N + 1):
                if x in S and conflict_count[x] > max_conf:
                    max_conf = conflict_count[x]
                    target = x
                    
            if target == -1:
                break
                
            S.remove(target)
            
            # Update conflict counts and active triples
            # Remove all triples containing target from active set
            # And decrement counts for other elements in those triples
            new_active = set()
            new_conf = {i: 0 for i in range(1, N + 1)}
            for idx in active_triples:
                t = triples[idx]
                if target in t:
                    # This triple is broken
                    continue
                else:
                    new_active.add(idx)
                    for x in t:
                        new_conf[x] += 1
            active_triples = new_active
            conflict_count = new_conf
            
        return S

    best_set = set()
    best_size = 0
    
    # 1. Initial Greedy
    S0 = initial_greedy()
    S0 = local_search(S0)
    if len(S0) > best_size:
        best_set = S0
        best_size = len(S0)
        
    # 2. Iterated Local Search (ILS)
    random.seed(42)
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        # Perturb best_set
        # Strategy: Remove k random elements from best_set
        k = random.randint(3, 10)
        S_perturbed = set(best_set)
        if len(S_perturbed) <= k:
            S_perturbed = set()
        else:
            to_remove = random.sample(list(S_perturbed), k)
            for x in to_remove:
                S_perturbed.remove(x)
                
        # Local search on perturbed set
        S_new = local_search(S_perturbed)
        
        if len(S_new) > best_size:
            best_set = S_new
            best_size = len(S_new)
            
    return list(best_set)
