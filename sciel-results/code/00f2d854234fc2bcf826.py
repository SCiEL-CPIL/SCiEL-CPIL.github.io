def _local_search(S, deadline, max_iters=50):
    best = S
    best_size = best.bit_count()
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # add one
        for i in range(24):
            if time.time() > deadline:
                break
            if not (best >> i) & 1:
                trial = best | (1 << i)
                if _is_valid(trial):
                    best = trial
                    best_size += 1
                    improved = True
                    break
        if improved:
            continue
        # swap one for one
        for i in range(24):
            if time.time() > deadline:
                break
            if (best >> i) & 1:
                base = best & ~(1 << i)
                for j in range(24):
                    if time.time() > deadline:
                        break
                    if not (base >> j) & 1:
                        trial = base | (1 << j)
                        if _is_valid(trial):
                            best = trial
                            # best_size stays same
                            improved = True
                            break
                if improved:
                    break
        if improved:
            continue
        # remove one, add two
        for i in range(24):
            if time.time() > deadline:
                break
            if (best >> i) & 1:
                base = best & ~(1 << i)
                for j in range(24):
                    if time.time() > deadline:
                        break
                    if not (base >> j) & 1:
                        for k in range(j + 1, 24):
                            if time.time() > deadline:
                                break
                            if not (base >> k) & 1:
                                trial = base | (1 << j) | (1 << k)
                                if trial.bit_count() > best_size and _is_valid(trial):
                                    best = trial
                                    best_size = trial.bit_count()
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    return best
