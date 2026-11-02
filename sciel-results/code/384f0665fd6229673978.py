<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Phase 1: diverse initializations including CSP-based
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(6):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    
    # Try CSP-based initialization
    if time.time() < deadline - 10:
        csp_rows, _ = _ac3_csp_init(n_rows, k, deadline, rng=random.Random(42))
        if csp_rows is not None and _valid(csp_rows, n_rows, k):
            inits.append(csp_rows)
    
    # Pick the best few as starting points for SA
    inits.sort(key=_max_entry)
    top_inits = inits[:4]

    # Phase 2: run multiple SA trajectories with time budget split
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 4
    budget_per_sa = max(5.0, remaining / n_sa - 1.0)

    overall_best = None
    overall_best_val = float('inf')
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 2:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Phase 3: CSP-based repair pass on the best solution
    if overall_best is not None and time.time() < deadline - 5:
        rng_repair = random.Random(777)
        for attempt in range(3):
            if time.time() >= deadline - 3:
                break
            # Try to improve by repairing each row with CSP
            improved = False
            for ri in range(n_rows):
                if time.time() >= deadline - 3:
                    break
                new_row = _csp_repair_row(ri, overall_best, k, deadline, rng_repair)
                if new_row is not None:
                    old_diffs = _row_diffs_set(overall_best[ri])
                    new_diffs = _row_diffs_set(new_row)
                    other_diffs = set()
                    for i, r in enumerate(overall_best):
                        if i != ri:
                            other_diffs |= _row_diffs_set(r)
                    if not (new_diffs & other_diffs):
                        old_max = _max_entry(overall_best)
                        overall_best[ri] = new_row
                        new_max = _max_entry(overall_best)
                        if new_max < old_max:
                            overall_best_val = new_max
                            improved = True
            if not improved:
                break

    # Phase 4: final polish with a short SA from the overall best
    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)
    elif time.time() < deadline - 5:
        b, bv = _sa_run(n_rows, k, overall_best, max(3.0, deadline - time.time() - 2), seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
=======
def _polish_pass(rows, n_rows, k, deadline, seed=12345):
    """High-precision polish: iteratively try to reduce each row's max by 1."""
    rng = random.Random(seed)
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    
    # Try to reduce the global max by targeting rows that achieve it
    max_rows = [i for i in range(n_rows) if cur[i][k] == cur_val]
    
    for _ in range(50):  # Max 50 attempts
        if time.time() > deadline - 2:
            break
        if not max_rows:
            break
        ri = rng.choice(max_rows)
        old_max = cur[ri][k]
        # Try to rebuild this row with max <= old_max - 1
        other_used = set()
        for i, r in enumerate(cur):
            if i != ri:
                other_used |= _row_diffs_set(r)
        
        target = old_max - 1
        row, _ = _greedy_row_with_update(set(other_used), k, max_val=target, rng=rng)
        if row is not None:
            new_diffs = _row_diffs_set(row)
            if not (new_diffs & other_used):
                cur[ri] = row
                new_val = _max_entry(cur)
                if new_val < cur_val:
                    cur_val = new_val
                    max_rows = [i for i in range(n_rows) if cur[i][k] == cur_val]
                else:
                    max_rows = [i for i in range(n_rows) if cur[i][k] == cur_val]
            else:
                # Try with slightly higher target
                target += 1
                row, _ = _greedy_row_with_update(set(other_used), k, max_val=target, rng=rng)
                if row is not None:
                    new_diffs = _row_diffs_set(row)
                    if not (new_diffs & other_used):
                        cur[ri] = row
                        cur_val = _max_entry(cur)
                        max_rows = [i for i in range(n_rows) if cur[i][k] == cur_val]
    
    return cur, cur_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    # Phase 1: diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(8):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    
    # Try CSP-based initialization
    if time.time() < deadline - 15:
        csp_rows, _ = _ac3_csp_init(n_rows, k, deadline, rng=random.Random(42))
        if csp_rows is not None and _valid(csp_rows, n_rows, k):
            inits.append(csp_rows)
    
    # Pick the best few as starting points for SA
    inits.sort(key=_max_entry)
    top_inits = inits[:5]

    # Phase 2: run multiple SA trajectories with time budget split
    remaining = deadline - time.time()
    n_sa = 5
    budget_per_sa = max(8.0, remaining / n_sa - 2.0)

    overall_best = None
    overall_best_val = float('inf')
    sa_results = []
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 10:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        sa_results.append((bv, b))
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Phase 3: CSP-based repair pass on the best solution
    if overall_best is not None and time.time() < deadline - 15:
        rng_repair = random.Random(777)
        for attempt in range(5):
            if time.time() >= deadline - 10:
                break
            improved = False
            for ri in range(n_rows):
                if time.time() >= deadline - 10:
                    break
                new_row = _csp_repair_row(ri, overall_best, k, deadline, rng_repair)
                if new_row is not None:
                    old_diffs = _row_diffs_set(overall_best[ri])
                    new_diffs = _row_diffs_set(new_row)
                    other_diffs = set()
                    for i, r in enumerate(overall_best):
                        if i != ri:
                            other_diffs |= _row_diffs_set(r)
                    if not (new_diffs & other_diffs):
                        old_max = _max_entry(overall_best)
                        overall_best[ri] = new_row
                        new_max = _max_entry(overall_best)
                        if new_max < old_max:
                            overall_best_val = new_max
                            improved = True
            if not improved:
                break

    # Phase 4: High-precision polish pass
    if overall_best is not None and time.time() < deadline - 5:
        polished, polished_val = _polish_pass(overall_best, n_rows, k, deadline, seed=54321)
        if polished_val < overall_best_val:
            overall_best_val = polished_val
            overall_best = polished

    # Phase 5: final polish SA from the overall best
    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)
    elif time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 5:
            b, bv = _sa_run(n_rows, k, overall_best, remaining_time, seed=999)
            if bv < overall_best_val:
                overall_best_val = bv
                overall_best = b
            # Another polish after final SA
            if time.time() < deadline - 2:
                polished, polished_val = _polish_pass(overall_best, n_rows, k, deadline, seed=11111)
                if polished_val < overall_best_val:
                    overall_best_val = polished_val
                    overall_best = polished

    # Final validation
    if overall_best is None or not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
>>>>>>> REPLACE

