temp_mask = current_mask & ~(1 << remove_idx)
# ...
new_mask = temp_mask
for idx in candidates:
    if not (temp_mask >> idx) & 1:
        if can_add(idx, temp_mask):
            new_mask |= (1 << idx)
