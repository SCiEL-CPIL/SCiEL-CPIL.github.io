    available = [x for x in range(max_val + 1) if x not in fixed]
    # Remove current marks from available
    for m in current_marks:
        if m in available: available.remove(m) # O(N)
