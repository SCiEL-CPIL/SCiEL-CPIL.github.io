non_protected = [u for u in cache if u not in protected]
if not non_protected:
    return  # can't evict, infeasible; assume doesn't happen
