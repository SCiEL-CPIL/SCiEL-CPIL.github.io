    for i in range(n):
        if (masks[i] & used_mask) == 0:
            best_set.add(candidates[i])
            best_sum += vals[i]
            used_mask |= masks[i]
