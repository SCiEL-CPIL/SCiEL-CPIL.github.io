<<<<<<< SEARCH
def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    # SA schedule
    T0 = 12.0
    T_end = 0.05
    # We'll run until time_budget expires, with T decaying geometrically
    # Estimate steps per second roughly; we'll just loop and check time
    T = T0
    decay = 0.9995  # per-step decay factor (tuned for a few thousand steps)
    min_T = T_end

    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        # Choose a move type
        r = rng.random()
        if r < 0.45:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is None:
                T = max(min_T, T * decay)
                continue
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(new_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = new_row
                cur_val += delta
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
        elif r < 0.70:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result, delta = _sa_move_row_swap(cur, cur_used, ri, rj, rng, k)
                    if result is not None:
                        if delta <= 0 or rng.random() < math.exp(-delta / T):
                            cur[ri], cur[rj] = result
                            cur_val += delta
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
        elif r < 0.85:
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            # Kick: random restart from best with a small perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            # Perturb a few positions
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else 10**6
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_sidon(cand_row):
                        new_diffs = _row_diffs_set(cand_row)
                        old_diffs = _row_diffs_set(row)
                        other_diffs = cur_used - old_diffs
                        if not (new_diffs & other_diffs):
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)

        T = max(min_T, T * decay)
        # Periodic re-anchoring if we've drifted too far
        if steps % 2000 == 0 and cur_val > best_val + 15:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val

    return best, best_val
=======
def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    # SA schedule
    T0 = 12.0
    T_end = 0.05
    T = T0
    decay = 0.9995
    min_T = T_end

    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        # Choose a move type
        r = rng.random()
        if r < 0.45:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k, cur_val)
            if new_row is None:
                T = max(min_T, T * decay)
                continue
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(new_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = new_row
                cur_val += delta
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
        elif r < 0.70:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result, delta = _sa_move_row_swap(cur, cur_used, ri, rj, rng, k)
                    if result is not None:
                        if delta <= 0 or rng.random() < math.exp(-delta / T):
                            cur[ri], cur[rj] = result
                            cur_val += delta
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
        elif r < 0.85:
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k, cur_val)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            # Kick: random restart from best with a small perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            # Perturb a few positions
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else cur_val + 20
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_sidon(cand_row):
                        new_diffs = _row_diffs_set(cand_row)
                        old_diffs = _row_diffs_set(row)
                        other_diffs = cur_used - old_diffs
                        if not (new_diffs & other_diffs):
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)

        T = max(min_T, T * decay)
        # Periodic re-anchoring if we've drifted too far
        if steps % 2000 == 0 and cur_val > best_val + 15:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val

    return best, best_val
>>>>>>> REPLACE

