for d in range(1, n):
    for lst in indices[d]:
        curr = 0
        min_val = 0
        max_val = 0
        for idx in lst:
            curr += signs[idx]
            if curr < min_val: min_val = curr
            elif curr > max_val: max_val = curr
        diff = max_val - min_val
        if diff > max_abs: max_abs = diff
