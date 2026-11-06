    # In the recursive call section:
    must_keep_left = i_mid | (outputs & set(v_left))
    must_keep_right = outputs & set(v_right)
    
    # Note: We pass must_keep_right to the right part.
    # But wait, if an output is in i_mid, it is in v_left.
    # If an output is in v_right, it is in must_keep_right.
    # What if an output is in NEITHER? Impossible, it must be in vertices.
    
    ops_left = recursive_pebble(v_left, external_deps, budget, must_keep_left, parents_map, deadline)
    ops_right = recursive_pebble(v_right, i_mid, budget, must_keep_right, parents_map, deadline)
    