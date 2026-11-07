    for mask in range(1 << n):
        if time.time() > deadline:
            break
        if popcounts[mask] < min_verts:
            continue
        
        dc = _distinct_degrees(mask)
        if dc > best_degrees:
            best_degrees = dc
            best_mask = mask
