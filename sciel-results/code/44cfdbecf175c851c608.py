if S_val < best_S:
    # Double check disjointness
    is_valid = True
    ...
    if dist < r1 + r2 - 1e-6:
        is_valid = False
    ...
    if is_valid:
        best_S = S_val
        best_params = final_x
