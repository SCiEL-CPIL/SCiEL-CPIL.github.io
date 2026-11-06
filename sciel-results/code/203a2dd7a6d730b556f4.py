def is_maximal(sidon_set, N):
    sums = set()
    for a in sidon_set:
        for b in sidon_set:
            if a <= b:
                sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in sidon_set:
            continue
        # Check if adding x breaks Sidon property
        # If it breaks, x is blocked.
        # If it DOES NOT break, then the set is NOT maximal (because we could add x).
        
        # Check if A U {x} is Sidon
        # New sums: 2*x and x+a for a in A
        # Check if 2*x in sums
        if 2 * x in sums:
            continue # Blocked
        # Check if x+a in sums for any a in A
        blocked = False
        for a in sidon_set:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False # x could be added, so not maximal
            
    return True
