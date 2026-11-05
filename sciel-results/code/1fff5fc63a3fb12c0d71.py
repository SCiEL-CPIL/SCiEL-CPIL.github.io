def construct():
    import time
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    # Precompute conflicts: for each number, which other numbers form a 3-AP with it
    # A 3-AP is (a, b, c) with a + c = 2b, a < b < c
    # For a given x, it conflicts with y if they can be in the same 3-AP
    conflicts = [set() for _ in range(n + 1)]
    
    # Generate all 3-APs and mark conflicts
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c <= n and c > b:
                # a, b, c form a 3-AP
                conflicts[a].add(b)
                conflicts[a].add(c)
                conflicts[b].add(a)
                conflicts[b].add(c)
                conflicts[c].add(a)
                conflicts[c].add(b)
    
    def check_no_ap3(selected_set):
        """Check if selected_set has no 3-term AP"""
        for a in selected_set:
            for b in selected_set:
                if a < b:
                    c = 2 * b - a
                    if c in selected_set:
                        return False
        return True
    
    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a 3-AP"""
        # x can be first, middle, or last term
        # If x is last: need a, b in selected with a < b < x and 2b - a = x
        # If x is middle: need a, c in selected with a < x < c and a + c = 2x
        # If x is first: need b, c in selected with x < b < c and 2b - x = c -> c = 2b - x
        for b in selected_set:
            # x is last: a = 2b - x, need a in selected, a < b
            a = 2 * b - x
            if a in selected_set and a < b:
                return False
            # x is middle: c = 2x - b, need c in selected, c > x
            c = 2 * x - b
            if c in selected_set and c > x:
                return False
            # x is first: c = 2b - x, need c in selected, c > b
            c = 2 * b - x
            if c in selected_set and c > b:
                return False
        return True
    
    # Greedy construction: add numbers in increasing order
    selected = set()
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.add(x)
    
    current_score = sum(1.0 / x for x in selected)
    
    # Local search phase 1: try to add any missing number
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if x in selected:
                continue
            if can_add(x, selected):
                selected.add(x)
                current_score += 1.0 / x
                improved = True
    
    # Local search phase 2: try swap operations
    # Remove a number, try to add one or more smaller numbers
    for x in range(n, 0, -1):
        if time.time() > deadline - 5:
            break
        if x not in selected:
            continue
        
        # Try removing x and adding smaller numbers
        selected.remove(x)
        old_score = current_score
        current_score -= 1.0 / x
        
        best_addition = None
        best_new_score = current_score
        
        for y in range(1, x):
            if y in selected:
                continue
            if can_add(y, selected):
                test_score = current_score + 1.0 / y
                if test_score > best_new_score:
                    best_new_score = test_score
                    best_addition = y
        
        if best_addition is not None and best_new_score > old_score:
            selected.add(best_addition)
            current_score = best_new_score
        else:
            # Revert: put x back
            selected.add(x)
            current_score = old_score
    
    # Final verification and return
    result = list(selected)
    return result
