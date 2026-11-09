import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Generate seed
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    
    selected_set = set(selected)
    excluded = set(range(1, 101)) - selected_set
    
    # 2. Local Search: Add back excluded elements if safe
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort excluded by value (maybe larger values are safer? or smaller?)
        # Let's try adding in descending order
        for e in sorted(excluded, reverse=True):
            if time.time() > deadline:
                break
            # Check if adding e creates sum 200
            # This is possible if (200 - e) is in reachable
            if (200 - e) in reachable:
                continue
            
            # Safe to add
            selected_set.add(e)
            excluded.remove(e)
            
            # Update reachable sums
            new_sums = {s + e for s in reachable if s + e <= 200}
            if 200 in new_sums:
                # This should not happen based on check, but safe to verify
                # If it does, revert
                selected_set.remove(e)
                excluded.add(e)
                continue
            
            reachable |= new_sums
            improved = True
    
    # 3. Try simple swaps to potentially enable more additions?
    # Since we want max cardinality, swaps don't increase size directly.
    # But they might change `reachable` to allow adding more.
    # This is complex. Let's stick to additions first.
    
    # 4. If time remains, try K-strategy for comparison?
    # Probably not needed if seed+add is strong.
    
    return list(selected_set)
