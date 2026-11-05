def has_4ap(selected):
    """Check if selected contains a nonconstant 4-AP."""
    s = set(selected)
    # Optimization: sort to help with pruning? No, set lookup is O(1).
    for a in s:
        # d can be at most (512 - a) / 3
        max_d = (512 - a) // 3
        for d in range(1, max_d + 1):
            if a + d in s and a + 2*d in s and a + 3*d in s:
                return True
    return False

def construct():
    n = 512
    # Step 1: Generate the ternary set (digits 0, 1 in base 3)
    # This set is 3-AP-free, hence 4-AP-free.
    ternary_set = set()
    for i in range(1, n + 1):
        # Check if i has only 0s and 1s in base 3
        val = i
        valid = True
        while val > 0:
            if val % 3 == 2:
                valid = False
                break
            val //= 3
        if valid:
            ternary_set.add(i)
            
    selected = ternary_set
    
    # Step 2: Greedy extension
    # Try to add remaining numbers in ascending order (deterministic)
    # Ascending order is a different heuristic than random shuffle.
    candidates = [x for x in range(1, n + 1) if x not in selected]
    
    for x in candidates:
        # Check if adding x creates a 4-AP
        conflict = False
        # x is the 4th term: x-3d, x-2d, x-d in selected
        for d in range(1, (x - 1) // 3 + 1):
            if (x - 3*d) in selected and (x - 2*d) in selected and (x - d) in selected:
                conflict = True
                break
        if conflict:
            continue
            
        # x is the 3rd term: x-2d, x-d, x+d in selected
        for d in range(1, min(x - 1, n - x) // 2 + 1): # x-2d >= 1 => d <= (x-1)/2, x+d <= n => d <= n-x
             if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                conflict = True
                break
        if conflict:
            continue

        # x is the 2nd term: x-d, x+d, x+2d in selected
        for d in range(1, min(x - 1, (n - x) // 2) + 1): # x-d >= 1, x+2d <= n
            if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                conflict = True
                break
        if conflict:
            continue

        # x is the 1st term: x+d, x+2d, x+3d in selected
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                conflict = True
                break
        if conflict:
            continue

        selected.add(x)

    return sorted(selected)
