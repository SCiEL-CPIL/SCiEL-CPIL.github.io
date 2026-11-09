"""
Flow-shop permutation construction using NEH, tabu search with aspiration,
and simulated annealing.  Uses a fast makespan evaluator and a priority-list
dispatching view for diversification.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

# ---------------------------------------------------------------------------
# Fast makespan
# ---------------------------------------------------------------------------
def _makespan(order, P):
    """Return the makespan of the given job order on processing matrix P."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    M = P.shape[1]
    if M == 0:
        return 0.0

    # Pre-extract rows as plain lists of floats for speed
    # P is a numpy array; P[j] is a view.  Convert the whole matrix once?
    # That would be expensive.  Instead, convert each row on demand.
    # For small M, list-based inner loop is faster than numpy element-wise.
    # We'll use a hybrid: if M <= 20, use pure Python lists.
    
    if M <= 20:
        # Convert P rows to lists of floats once (cache would help but P is fixed)
        # For simplicity, do it inline.
        first = order[0]
        row = P[first]
        curr = [0.0] * M
        curr[0] = row[0]
        for m in range(1, M):
            curr[m] = curr[m - 1] + row[m]
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new = [0.0] * M
            new[0] = curr[0] + row[0]
            for m in range(1, M):
                a = curr[m]
                b = new[m - 1]
                new[m] = (a if a > b else b) + row[m]
            curr = new
        return curr[-1]
    else:
        first = order[0]
        curr = np.cumsum(P[first])
        for idx in range(1, n_jobs):
            j = order[idx]
            prev = curr
            new = np.empty(M)
            new[0] = prev[0] + P[j, 0]
            for m in range(1, M):
                new[m] = max(prev[m], new[m - 1]) + P[j, m]
            curr = new
        return curr[-1]


def _makespan_from_prefix(prefix, P):
    """Makespan of just the prefix (used during NEH insertion)."""
    n = len(prefix)
    if n == 0:
        return 0.0
    M = P.shape[1]
    if M == 0:
        return 0.0
    first = prefix[0]
    row = P[first]
    curr = [0.0] * M
    curr[0] = row[0]
    for m in range(1, M):
        curr[m] = curr[m - 1] + row[m]
    for idx in range(1, n):
        j = prefix[idx]
        row = P[j]
        new = [0.0] * M
        new[0] = curr[0] + row[0]
        for m in range(1, M):
            a = curr[m]
            b = new[m - 1]
            new[m] = (a if a > b else b) + row[m]
        curr = new
    return curr[-1]


# ---------------------------------------------------------------------------
# NEH heuristic with randomised tie-breaking
# ---------------------------------------------------------------------------
def _neh(P, mode='total', rng=None):
    """
    NEH heuristic.
    mode: 'total', 'first_last', 'random', 'random_weighted'
    """
    n, M = P.shape
    if n == 0:
        return []
    if n == 1:
        return [0]

    if rng is None:
        rng = random.Random()

    if mode == 'total':
        keys = P.sum(axis=1).astype(float)
    elif mode == 'first_last':
        keys = (P[:, 0] + P[:, -1]).astype(float)
    elif mode == 'random_weighted':
        # Random priority-list: each job gets a random weight; sort descending
        keys = rng.random_sample(n)
    else:
        keys = P.sum(axis=1).astype(float)

    # Add tiny noise to break ties and encourage diversity
    noise = rng.random_sample(n) * 1e-9
    keys = keys + noise
    order = np.argsort(-keys).tolist()

    seq = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
    return seq


# ---------------------------------------------------------------------------
# Tabu search with aspiration criterion
# ---------------------------------------------------------------------------
def _tabu_search(order, P, tabu_len=7, max_iter=200000, deadline=None):
    """
    Tabu search with move = relocate one job to a different position.
    Aspiration: accept a tabu move if it improves the best known solution.
    """
    n = len(order)
    if n < 2:
        return order

    cur = list(order)
    cur_ms = _makespan(cur, P)
    best = list(cur)
    best_ms = cur_ms

    # Tabu memory: for each job, remember until which iteration it is tabu
    # to be moved FROM its current position.  Simpler: store (job, from_pos, to_pos)
    # We use a dictionary: key = (job, pos), value = iteration at which it becomes non-tabu.
    tabu_until = {}
    best_iter = 0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        best_move_ms = float('inf')
        best_move = None  # (job, frm, to)

        # Try moving each job to each other position
        for i in range(n):
            job = cur[i]
            for j in range(n):
                if i == j:
                    continue
                # Check tabu: is (job, i) tabu?
                key = (job, i)
                if key in tabu_until and it < tabu_until[key]:
                    # Tabu move; only allow if it beats best (aspiration)
                    # We'll evaluate it but mark as tabu
                    pass
                # Build new sequence
                new_seq = cur[:i] + cur[i + 1:]
                # Insert at position j in new_seq
                # If j > i, the position shifts by -1
                insert_pos = j
                new_seq = new_seq[:insert_pos] + [job] + new_seq[insert_pos:]
                ms = _makespan(new_seq, P)
                
                is_tabu = (key in tabu_until and it < tabu_until[key])
                aspiration = ms < best_ms
                
                if ms < best_move_ms or (is_tabu and aspiration and ms < best_move_ms):
                    # Actually, we want the best non-tabu move, or best tabu move if aspiration
                    # Let's track best_non_tabu and best_tabu_asp separately
                    pass

        # Redo: track best non-tabu and best tabu-aspiration
        best_nt_ms = float('inf')
        best_nt = None
        best_ta_ms = float('inf')
        best_ta = None

        for i in range(n):
            job = cur[i]
            key = (job, i)
            is_tabu = (key in tabu_until and it < tabu_until[key])
            for j in range(n):
                if i == j:
                    continue
                new_seq = cur[:i] + cur[i + 1:]
                insert_pos = j
                new_seq = new_seq[:insert_pos] + [job] + new_seq[insert_pos:]
                ms = _makespan(new_seq, P)
                if not is_tabu:
                    if ms < best_nt_ms:
                        best_nt_ms = ms
                        best_nt = (i, j, job, ms)
                else:
                    if ms < best_ms and ms < best_ta_ms:
                        best_ta_ms = ms
                        best_ta = (i, j, job, ms)

        # Choose move
        chosen = None
        chosen_ms = None
        if best_nt is not None:
            chosen = best_nt
            chosen_ms = best_nt_ms
        elif best_ta is not None:
            chosen = best_ta
            chosen_ms = best_ta_ms
        else:
            break  # No moves

        i, j, job, ms = chosen
        # Apply move
        new_seq = cur[:i] + cur[i + 1:]
        insert_pos = j
        new_seq = new_seq[:insert_pos] + [job] + new_seq[insert_pos:]
        
        # Set tabu for the moved job from its old position
        tabu_until[(job, i)] = it + tabu_len
        # Also tabu the reverse move?  Optional.
        
        cur = new_seq
        cur_ms = ms
        if cur_ms < best_ms:
            best = list(cur)
            best_ms = cur_ms
            best_iter = it

        # Diversify if stuck
        if it - best_iter > 300:
            # Perturbation: random swap of two jobs
            a = rng.random() * n
            b = rng.random() * n
            a = int(a)
            b = int(b)
            if a != b:
                cur[a], cur[b] = cur[b], cur[a]
                cur_ms = _makespan(cur, P)
            best_iter = it

    return best


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _simulated_annealing(order, P, T0=1.0, cooling=0.9995, max_iter=300000,
                         deadline=None, rng=None):
    """
    Simulated annealing with relocate moves.
    """
    n = len(order)
    if n < 2:
        return order

    if rng is None:
        rng = random.Random()

    cur = list(order)
    cur_ms = _makespan(cur, P)
    best = list(cur)
    best_ms = cur_ms

    T = T0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if T < 1e-8:
            break

        # Random relocate move
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        job = cur[i]
        new_seq = cur[:i] + cur[i + 1:]
        new_seq = new_seq[:j] + [job] + new_seq[j:]
        new_ms = _makespan(new_seq, P)
        delta = new_ms - cur_ms
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur = new_seq
            cur_ms = new_ms
            if cur_ms < best_ms:
                best = list(cur)
                best_ms = cur_ms
        T *= cooling

    return best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Returns a list of job indices (permutation of 0..n-1).
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
    n = P.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]
    M = P.shape[1]

    start = time.time()
    deadline = start + 140.0  # safety margin before 150 s kill

    rng = random.Random(42)

    # ---- Phase 1: diverse NEH constructions ----
    candidates = []
    modes = ['total', 'first_last', 'random_weighted', 'random_weighted', 'random_weighted']
    for mode in modes:
        if time.time() > deadline - 30:
            break
        seq = _neh(P, mode=mode, rng=rng)
        candidates.append(seq)

    # Pick best initial
    best_seq = candidates[0]
    best_ms = _makespan(best_seq, P)
    for s in candidates[1:]:
        ms = _makespan(s, P)
        if ms < best_ms:
            best_ms = ms
            best_seq = s

    # ---- Phase 2: Tabu search ----
    if time.time() < deadline - 10:
        ts_deadline = deadline - 3
        best_seq = _tabu_search(best_seq, P, tabu_len=7, max_iter=200000,
                                deadline=ts_deadline)
        best_ms = _makespan(best_seq, P)

    # ---- Phase 3: Simulated annealing polish ----
    if time.time() < deadline - 5:
        sa_deadline = deadline - 1
        best_seq = _simulated_annealing(best_seq, P, T0=1.0, cooling=0.9998,
                                        max_iter=200000, deadline=sa_deadline,
                                        rng=rng)
        best_ms = _makespan(best_seq, P)

    # ---- Phase 4: quick local improvement (first-improvement relocate) ----
    if time.time() < deadline - 1:
        cur = list(best_seq)
        cur_ms = best_ms
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            n_cur = len(cur)
            for i in range(n_cur):
                if time.time() > deadline - 0.5:
                    break
                job = cur[i]
                for j in range(n_cur):
                    if i == j:
                        continue
                    new_seq = cur[:i] + cur[i + 1:]
                    new_seq = new_seq[:j] + [job] + new_seq[j:]
                    ms = _makespan(new_seq, P)
                    if ms < cur_ms:
                        cur = new_seq
                        cur_ms = ms
                        improved = True
                        break
                if improved:
                    break

    return list(best_seq)

# EVOLVE-BLOCK-END
