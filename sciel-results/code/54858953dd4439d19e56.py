import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    limit = 180
    
    # Precompute forbidden pairs: for each a in [1, limit], find all (b, c) with b > a, c > a, b != c
    # such that 1/a = 1/b + 1/c
    # c = a*b / (b - a), must be integer, c > a, c != b, c <= limit
    # We store for each a the set of b values that would create a conflict if both b and c are in S
    
    # Actually, let's think differently. For a given set S, we need to check if there exist
    # distinct a, b, c in S with 1/a = 1/b + 1/c.
    # WLOG assume a < b < c (since 1/a = 1/b + 1/c implies a < b and a < c).
    # Then for each pair (b, c) with b < c, we can compute a = bc/(b+c) and check if a is in S and a < b.
    # Alternatively, for each pair (a, b) with a < b, compute c = ab/(b-a) and check if c is in S and c > a and c != b.
    
    # Let's precompute: for each a in [1, limit], and each b in [a+1, limit], compute c = ab/(b-a).
    # If c is integer, c > a, c != b, c <= limit, then (a, b, c) is a forbidden triple.
    
    # Strategy: Greedy from largest to smallest, then local search to improve.
    
    # First, let's do a greedy construction
    S = set()
    S_list = []
    
    for a in range(limit, 0, -1):
        conflict = False
        # Check all b in S where b > a
        for b in S:
            if b <= a:
                continue
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c == a or c == b:
                    continue
                if c in S:
                    conflict = True
                    break
        
        if not conflict:
            S.add(a)
            S_list.append(a)
        
        if time.time() > deadline:
            break
    
    # Now do local search: try to swap out an element for a better one
    # Or try to add elements that were skipped
    
    # Let's try a different approach: use the precomputed forbidden pairs for O(1) lookup
    # Precompute for each a, the set of b values such that (a, b, c) is a valid triple with c in [1, limit]
    
    # forbidden[a] = set of b values where if a and b are both in S, then c = ab/(b-a) must NOT be in S
    # Actually, let's precompute triples: for each a < b < c, check if 1/a = 1/b + 1/c
    
    # Better: precompute for each pair (a, b) with a < b, whether c = ab/(b-a) is a valid integer in (a, limit]
    # and c != b. If so, then (a, b, c) is a forbidden triple.
    
    # Let's use a different greedy strategy: start from the largest numbers and work down,
    # but use precomputed conflict checks.
    
    # Precompute: for each a in [1, limit], for each b in [a+1, limit], compute c.
    # Store conflicts[a] as a dict: b -> c (if c is valid)
    
    conflicts = {}
    for a in range(1, limit + 1):
        conflicts[a] = {}
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c != b and c <= limit:
                    conflicts[a][b] = c
    
    # Now greedy: add elements from largest to smallest
    S = set()
    
    for a in range(limit, 0, -1):
        can_add = True
        # Check if adding a creates any conflict
        # For each b in S where b > a: check if conflicts[a][b] is in S
        # For each b in S where b < a: check if conflicts[b][a] is in S
        for b in S:
            if b > a:
                if b in conflicts[a] and conflicts[a][b] in S:
                    can_add = False
                    break
            elif b < a:
                if a in conflicts[b] and conflicts[b][a] in S:
                    can_add = False
                    break
        
        if can_add:
            S.add(a)
        
        if time.time() > deadline:
            break
    
    # Local search: try to improve by swapping elements
    # Try removing one element and adding two others, or just adding one
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S)
        
        # Try adding any element not in S
        for a in range(1, limit + 1):
            if a in S:
                continue
            can_add = True
            for b in S:
                if b > a:
                    if b in conflicts[a] and conflicts[a][b] in S:
                        can_add = False
                        break
                elif b < a:
                    if a in conflicts[b] and conflicts[b][a] in S:
                        can_add = False
                        break
            
            if can_add:
                S.add(a)
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved and time.time() < deadline - 10:
            for remove_val in list(S):
                S.remove(remove_val)
                for a in range(1, limit + 1):
                    if a in S:
                        continue
                    can_add = True
                    for b in S:
                        if b > a:
                            if b in conflicts[a] and conflicts[a][b] in S:
                                can_add = False
                                break
                        elif b < a:
                            if a in conflicts[b] and conflicts[b][a] in S:
                                can_add = False
                                break
                    
                    if can_add:
                        S.add(a)
                        improved = True
                        break
                
                if not improved:
                    S.add(remove_val)
                
                if improved:
                    break
        
        if time.time() > deadline:
            break
    
    return list(S)
