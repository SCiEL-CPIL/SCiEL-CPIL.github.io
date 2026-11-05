def update_delta(old_val, new_val):
    if old_val > 1 and new_val > 1: return 0
    if old_val > 1 and new_val <= 1: return -1
    if old_val <= 1 and new_val > 1: return 1
    return 0
