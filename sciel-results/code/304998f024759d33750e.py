def construct():
    TARGET = 200
    # Greedy selection: start from the largest elements.
    # We maintain the set of achievable subset sums.
    # Initially, only 0 is achievable.
    achievable_sums = {0}
    selected = []
    
    # Iterate from largest to smallest to maximize cardinality 
    # while keeping elements "large" enough to avoid small sum combinations.
    for i in range(100, 0, -1):
        # Check if adding i creates a subset summing to TARGET.
        # A new sum is formed if (sum + i) == TARGET for some existing sum.
        # Or equivalently, if (TARGET - i) is in achievable_sums.
        if (TARGET - i) in achievable_sums:
            continue
        
        # If valid, add i to selected and update achievable sums.
        # New sums are old_sums + i.
        # We only care about sums up to TARGET.
        new_sums = set()
        for s in achievable_sums:
            new_s = s + i
            if new_s <= TARGET:
                new_sums.add(new_s)
        
        # If any new sum is exactly TARGET, we shouldn't have added it, 
        # but the check above prevents that.
        # Update achievable_sums with new_sums.
        achievable_sums.update(new_sums)
        selected.append(i)
        
    return selected
