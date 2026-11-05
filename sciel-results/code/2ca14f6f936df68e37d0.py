# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(pt, order):
    """Compute makespan for a given job order. pt: (jobs, machines)."""
    m = pt.shape[1]
    n = len(order)
    # completion times per machine, accumulated
    # C[j] for the running job on machine j
    # We process job by job; for each job, completion on machine 0 = prev job's machine0 + p
    # For machine k>0: completion = max(prev job's completion on machine k, current job's completion on machine k-1) + p
    # Track last completion per machine
    last = np.zeros(m, dtype=np.float64)
    makespan = 0.0
    for j in order:
        row = pt[j]
        prev_c = last[0]
        c = prev_c + row[0]
        last[0] = c
        for k in range(1, m):
            c = max(last[k], c) + row[k]
            last[k] = c
        makespan = c
    return makespan


def _neighborhood_moves(order, n, max_block=3):
    """Yield (new_order_tuple, move_key) for block insertions.
    A move: take a block of length L (1..max_block) starting at i, insert before position j
    (j != i and j != i+L to avoid no-ops)."""
    for i in range(n):
        for L in range(1, max_block + 1):
            if i + L > n:
                break
            block = order[i:i+L]
            rest = order[:i] + order[i+L:]
            # insert block before each position in rest (positions 0..len(rest)-1)
            # also after last (position len(rest))
            rlen = len(rest)
            for j in range(rlen + 1):
                # skip no-op: inserting at i (j == i) or at i (j == i, since block removed shifts)
                # no-op when j == i or j == i (block was at i). Also j == i+L equivalent? 
                # When we remove block at i, positions i..i+L-1 become the following elements.
                # Inserting back at j==i reproduces original. j==i+L? after removal, index i+L-1 in rest
                # is the element that was at i+L; inserting before it (j = i+L-1) puts block right after
                # where it was -> also no-op? Let's just test via key equality to be safe, but to save time
                # skip j == i and j == i+L-1 (the two boundaries).
                if j == i or (L < n and j == i + L - 1):
                    continue
                new_order = rest[:j] + block + rest[j:]
                # key for tabu: (i, L, j)
                yield new_order, (i, L, j)


def _tabu_search(pt, n, m, time_limit, rng):
    # initial order via NEH
    totals = pt.sum(axis=1)
    # NEH: insert jobs in order of decreasing total
    order = []
    for j in np.argsort(-totals):
        j = int(j)
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(order) + 1):
            cand = order[:pos] + [j] + order[pos:]
            ms = _makespan(pt, cand)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [j] + order[best_pos]

    best_order = list(order)
    best_ms = _makespan(pt, best_order)
    cur_order = list(order)
    cur_ms = best_ms

    # tabu list of move keys with expiry (iteration)
    tabu = {}
    tabu_tenure = 7
    no_improve = 0
    max_no_improve = 40

    deadline = time.time() + time_limit
    it = 0
    while time.time() < deadline:
        # generate moves
        best_move_ms = np.inf
        best_move_order = None
        best_move_key = None
        # iterate neighborhood; if too many, sample
        moves = list(_neighborhood_moves(cur_order, n, max_block=3))
        if not moves:
            break
        # randomly sample if large
        if len(moves) > 400:
            idx = rng.choice(len(moves), size=400, replace=False)
            moves = [moves[i] for i in idx]
        for new_order, key in moves:
            tabu_until = tabu.get(key, -1)
            if it < tabu_until:
                continue  # tabu
            ms = _makespan(pt, new_order)
            # aspiration: allow if it improves best
            if ms < best_move_ms:
                best_move_ms = ms
                best_move_order = new_order
                best_move_key = key
        if best_move_order is None:
            break
        # apply best move
        cur_order = best_move_order
        cur_ms = best_move_ms
        if cur_ms < best_ms:
            best_ms = cur_ms
            best_order = list(cur_order)
            no_improve = 0
        else:
            no_improve += 1
        # add to tabu
        tabu[best_move_key] = it + tabu_tenure
        # clean tabu
        if len(tabu) > 2000:
            for k in list(tabu.keys()):
                if tabu[k] <= it:
                    del tabu[k]
        it += 1
        if no_improve > max_no_improve:
            # perturb: random restart of a few positions
            for _ in range(3):
                a = rng.randint(n)
                b = rng.randint(n)
                cur_order[a], cur_order[b] = cur_order[b], cur_order[a]
            cur_ms = _makespan(pt, cur_order)
            if cur_ms < best_ms:
                best_ms = cur_ms
                best_order = list(cur_order)
            no_improve = 0
    return best_order


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    start = time.time()
    time_limit = 130.0  # safety margin under 150s
    if n <= 1:
        return list(range(n))
    rng = np.random.default_rng(12345)
    best = _tabu_search(pt, n, m, time_limit, rng)
    return list(best)
# EVOLVE-BLOCK-END
