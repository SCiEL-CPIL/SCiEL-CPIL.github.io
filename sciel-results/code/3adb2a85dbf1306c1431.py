<<<<<<< SEARCH
def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 118  # safety margin before the 150 s kill

    # ---- Phase 1: fast greedy fallback (guarantees a valid solution) ----
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = rows
    best_scope = _total_scope(rows)

    # ---- Phase 2: difference-assignment CSP with iterative scope deepening ----
    # Try to find a solution with a smaller scope by assigning every difference
    # in [1, M] to a row. Start near the greedy scope and shrink M.
    greedy_scope = best_scope
    # The counting bound lower bound is roughly total_diffs / n_rows.
    lower = (n_rows * (k * (k + 1) // 2)) // n_rows + 1

    # We attempt a handful of M values descending from greedy_scope-1 down to
    # a floor, each with a bounded time budget. The CSP is expensive, so we
    # only try a few.
    M_budget = max(0, int((deadline - start_time) * 0.45))
    if M_budget > 20:
        per_try = max(6, M_budget // 8)
        tried = 0
        for M in range(min(greedy_scope - 1, 130), lower, -1):
            if tried >= 8:
                break
            if time.time() > deadline - 40:
                break
            t0 = time.time()
            csp_rows, _ = _csp_assign(n_rows, k, M, rng, t0 + per_try)
            tried += 1
            if csp_rows is not None and _valid(csp_rows, k):
                sc = _total_scope(csp_rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = csp_rows

    # ---- Phase 3: local search / polish to shrink the best solution ----
    if time.time() < deadline - 20:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 20)
        if _valid(best_rows, k):
            best_rows = _polish_pass(best_rows, k, deadline - 8)

    # Final safety: ensure validity
    if not _valid(best_rows, k):
        # fall back to the greedy solution
        rows2 = _greedy_construct(n_rows, k)
        if rows2 is not None and _valid(rows2, k):
            return rows2

    return best_rows
=======
def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    Uses multi-restart randomized greedy + simulated annealing.
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 118  # safety margin before the 150 s kill

    # ---- Phase 1: fast greedy fallback (guarantees a valid solution) ----
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = rows
    best_scope = _total_scope(rows)

    # ---- Phase 2: Multi-restart randomized greedy construction ----
    # Try several random restarts to find better initial solutions
    num_restarts = 15
    per_restart_budget = (deadline - start_time - 30) / num_restarts  # leave 30s for SA

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        t0 = time.time()
        restart_deadline = min(deadline - 30, t0 + per_restart_budget)

        # Randomized greedy: at each step, pick from top few candidates
        used = set()
        cand_rows = []
        valid = True
        for _ in range(n_rows):
            row = [0]
            new_diffs = set()
            x = 0
            found = False
            while len(row) < k + 1:
                x += 1
                if x > best_scope + 20:
                    valid = False
                    break
                new = {x - a for a in row}
                if not (new & used) and not (new & new_diffs):
                    # Accept with some probability, or always if few options
                    if len(row) < 2 or rng.random() < 0.6:
                        row.append(x)
                        new_diffs |= new
                        found = True
                        break
            if not valid or not found:
                valid = False
                break
            cand_rows.append(row)
            used |= new_diffs

        if valid and len(cand_rows) == n_rows:
            sc = _total_scope(cand_rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = cand_rows

    # ---- Phase 3: Simulated annealing + local search refinement ----
    if time.time() < deadline - 10:
        sa_deadline = deadline - 10
        best_rows = _simulated_annealing(best_rows, k, rng, sa_deadline)
        if _valid(best_rows, k):
            sc = _total_scope(best_rows)
            if sc < best_scope:
                best_scope = sc

        # Final local search polish
        if time.time() < deadline - 5:
            best_rows = _local_search_scope(best_rows, k, rng, deadline - 5)
            if _valid(best_rows, k):
                sc = _total_scope(best_rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = best_rows

    # Final safety: ensure validity
    if not _valid(best_rows, k):
        rows2 = _greedy_construct(n_rows, k)
        if rows2 is not None and _valid(rows2, k):
            return rows2

    return best_rows
>>>>>>> REPLACE

