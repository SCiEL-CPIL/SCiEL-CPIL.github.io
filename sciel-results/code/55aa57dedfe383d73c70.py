"""Flow-shop makespan minimization via NEH + block-adjacent-insertion Tabu Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(order, P):
    """Compute makespan for a given job ordering. Vectorised cumulative sums."""
    # P: (n, m); order: (n,) indices
    seq = P[order]                      # (n, m)
    # completion C[i, j] = C[i, j-1] + p[i, j]; C[i, 0] = C[i-1, 0] + p[i, 0]
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
    n, m = seq.shape
    C = np.empty((n, m), dtype=np.float64)
    # first column
    C[0, 0] = seq[0, 0]
    if n > 1:
        C[1:, 0] = np.cumsum(seq[1:, 0]) + C[0, 0]
    # remaining columns: C[i,j] = max(C[i-1,j], C[i,j-1]) + p[i,j]
    if m > 1:
        for j in range(1, m):
            prev_col = C[:-1, j-1] if n > 1 else np.array([])
            # C[i,j] = max(C[i-1,j], C[i,j-1]) + p[i,j]
            # C[i-1,j] for i in 1..n-1 is C[0..n-2, j]
            if n > 1:
                up = C[0:n-1, j]      # C[i-1, j] for i=1..n-1
                left = C[1:n, j-1]    # C[i, j-1] for i=1..n-1
                C[1:n, j] = np.maximum(up, left) + seq[1:n, j]
            C[0, j] = C[0, j-1] + seq[0, j]
    return C[-1, -1]


def _neh_order(P):
    """NEH heuristic: insert jobs in decreasing total time at best position."""
    n, m = P.shape
    totals = P.sum(axis=1)
    order = np.argsort(-totals)
    seq = [int(order[0])]
    for idx in order[1:]:
        best_cost = np.inf
        best_pos = len(seq)
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [int(idx)] + seq[pos:]
            c = _makespan(np.array(cand), P)
            if c < best_cost:
                best_cost = c
                best_pos = pos
        seq = seq[:best_pos] + [int(idx)] + seq[best_pos:]
    return seq


def _tabu_search(P, init, deadline):
    n, m = P.shape
    best_order = list(init)
    best_cost = _makespan(np.array(best_order), P)
    cur_order = list(init)
    cur_cost = best_cost

    # Tabu list: store (job, from_pos, to_pos) with a tabu tenure; we use a
    # simpler scheme: record the move (block_start, block_len, from, to) as a key
    # and forbid its inverse for `tenure` iterations.
    tabu = {}
    it = 0
    no_improve = 0
    start = time.time()

    while time.time() < deadline:
        # Generate neighborhood: move a block of size 1..3 from pos a to pos b
        n_moves = 0
        best_move_cost = np.inf
        best_move = None
        best_move_order = None

        # Precompute nothing heavy; iterate over blocks.
        for blen in (1, 2, 3):
            for a in range(n - blen + 1):
                block = cur_order[a:a+blen]
                rest = cur_order[:a] + cur_order[a+blen:]
                for b in range(len(rest) + 1):
                    # b is insertion position in rest
                    # Skip trivial (reinsert at same place)
                    # After removing block at a, the original position maps:
                    # if b > a, inserting at b in rest corresponds to original a+blen
                    # if b <= a, corresponds to original b
                    new_order = rest[:b] + block + rest[b:]
                    if new_order == cur_order:
                        continue
                    # tabu check: inverse move key
                    # inverse: block now at b, going back to a
                    inv_key = (b, blen, a)
                    if inv_key in tabu and tabu[inv_key] > it:
                        continue
                    c = _makespan(np.array(new_order), P)
                    n_moves += 1
                    if c < best_move_cost:
                        best_move_cost = c
                        best_move = (a, blen, b)
                        best_move_order = new_order

        if best_move is None:
            break

        a, blen, b = best_move
        # Aspiration: accept if better than best known
        if best_move_cost < cur_cost or best_move_cost < best_cost:
            cur_order = best_move_order
            cur_cost = best_move_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = list(cur_order)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # no improving non-tabu move; accept best to escape (or break)
            # For diversification, accept the best move even if worse, occasionally
            if no_improve > 30 and (no_improve % 5 == 0):
                cur_order = best_move_order
                cur_cost = best_move_cost
            else:
                # accept best move anyway to keep moving (steepest descent with
                # tabu to avoid cycling); if it's worse, it's a diversification step
                cur_order = best_move_order
                cur_cost = best_move_cost
                no_improve += 1

        # record tabu: inverse move forbidden for tenure
        tenure = max(1, int(n * 0.5))
        inv_key = (b, blen, a)
        tabu[inv_key] = it + tenure
        # prune old tabu entries
        if len(tabu) > 500:
            for k in list(tabu.keys()):
                if tabu[k] <= it:
                    del tabu[k]
        it += 1

    return best_order, best_cost


def order_jobs(processing_times):
    P = np.asarray(processing_times, dtype=np.float64)
    n, m = P.shape

    deadline = time.time() + 110.0  # safety margin under 120s

    # NEH initialization
    try:
        init = _neh_order(P)
    except Exception:
        init = list(range(n))

    best_order, best_cost = _tabu_search(P, init, deadline)

    # Final high-precision polish: a couple of steepest single-job insertion passes
    if time.time() < deadline - 5.0:
        for _ in range(3):
            improved = False
            cur = list(best_order)
            c = best_cost
            for a in range(n):
                job = cur[a]
                rest = cur[:a] + cur[a+1:]
                for b in range(len(rest) + 1):
                    cand = rest[:b] + [job] + rest[b:]
                    if cand == cur:
                        continue
                    nc = _makespan(np.array(cand), P)
                    if nc < c:
                        cur = cand
                        c = nc
                        improved = True
            if c < best_cost:
                best_order = cur
                best_cost = c
            if not improved:
                break

    return list(best_order)
# EVOLVE-BLOCK-END
