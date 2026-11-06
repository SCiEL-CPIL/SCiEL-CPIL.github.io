    current_set = set(current_A)
    # ...
    if new_val in current_set:
        continue
    current_set.remove(old_val)
    current_set.add(new_val)
    # ...
    if revert:
        current_set.remove(new_val)
        current_set.add(old_val)
