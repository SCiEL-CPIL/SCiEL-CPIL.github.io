    # ... inside construct ...
    
    # Init used_diffs
    used_diffs = set()
    for c in current_sol:
        used_diffs.update(c[1])

    # ... inside SA loop ...
        # Check conflict
        # Temporarily remove old_diffs
        # We don't want to modify used_diffs if we reject.
        # So we check: new_diffs.isdisjoint(used_diffs - old_diffs)
        # This creates a new set.
        # Alternative:
        # Count overlaps
        # overlap = len(new_diffs & used_diffs)
        # We need overlap to be 0? No, we need new_diffs to be disjoint from (used_diffs \ old_diffs).
        # This means any diff in new_diffs that is in used_diffs MUST be in old_diffs.
        # i.e. (new_diffs & used_diffs) must be a subset of old_diffs.
        # Or simply: (new_diffs & (used_diffs - old_diffs)) is empty.
        
        # Let's just do the set difference, it's small.
        available_diffs = used_diffs.difference(old_diffs)
        if new_diffs.isdisjoint(available_diffs):
             # Accept
             # Update used_diffs
             used_diffs.difference_update(old_diffs)
             used_diffs.update(new_diffs)
             # ...
