    candidates = None
    for w in U:
        if candidates is None:
            candidates = set(reverse_neighbors[w])
        else:
            candidates &= set(reverse_neighbors[w])
        if not candidates:
            break
    if candidates:
        # filter out existing code
        valid = [c for c in candidates if c not in code_set]
        if valid:
            # Perform move
    