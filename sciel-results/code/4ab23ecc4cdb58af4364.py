    target_front = fronts[best_end]
    prev_idx = -1
    for k in range(len(target_front[1])):
        if target_front[1][k] == best_len:
            prev_idx = target_front[2][k]
            break
