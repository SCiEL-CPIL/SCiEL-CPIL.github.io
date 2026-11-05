# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_vec(pt, perm):
    """Vectorized makespan (NumPy). O(n*m)."""
    n, m = pt.shape
    if n == 0:
        return 0.0
    P = pt[perm]  # (n, m)
    if m == 1:
        return float(P.sum())
    C = np.empty((n, m), dtype=np.float64)
    C[0, :] = np.cumsum(P[0, :])
    C[1:, 0] = C[:-1, 0] + P[1:, 0]
    for k in range(1, m):
        C[1:, k] = np.maximum(C[1:, k - 1], C[:-1, k]) + P[1:, k]
    return float(C[-1, -1])


def _insertion_cost_matrix(pt):
    """cost[i,j] = approximate makespan delta of inserting job j right after
    job i (i.e. ... i, j ...). Precomputed once, O(n^2*m).
    Sentinel rows/cols for the ends: index n means 'before first', index n+1
    means 'after last'. We store a (n+2) x (n+2) matrix.
    """
    n, m = pt.shape
    # completion profile of a prefix: for a sequence, the vector of completion
    # times on each machine. The marginal cost of appending job j after a
    # sequence with machine-completion vector C is sum over machines of
    # max(C[k], C[k-1]) ... which is not separable. Instead we approximate
    # using the per-machine 'free' capacity. For a good-enough TSP weight we
    # use: cost[i,j] = sum_k max(0, pt[j,k] - slack) style is complex.
    #
    # Practical approximation that is cheap and effective: the incremental
    # makespan of job j depends on how long the shop is busy before it.
    # We approximate the 'arrival' of job j after i by the completion of i on
    # machine 0 and the completion of i on machine m-1 (the two extremes),
    # and the 'departure' contribution of j.
    #
    # Simpler, robust weight: sum of processing of j plus a coupling term
    # between i and j on the bottleneck machines.
    #
    # We use: w(i,j) = sum_k pt[j,k]  +  beta * sum_k max(pt[i,k], pt[j,k])
    # This rewards pairing jobs that do not pile up on the same machine.
    S = pt.sum(axis=1)  # (n,)
    beta = 0.5
    # (n+2) x (n+2)
    cost = np.zeros((n + 2, n + 2), dtype=np.float64)
    # normal pairs
    for i in range(n):
        row = S + beta * np.maximum(pt[i, :], pt[:, :]).sum(axis=1)
        cost[i, :n] = row
    # start (n) -> j : just the job's own load (no predecessor)
    cost[n, :n] = S
    # i -> end (n+1): 0
    cost[:n, n + 1] = 0.0
    # end/end
    cost[n, n + 1] = 0.0
    cost[n + 1, n + 1] = 0.0
    # j -> start is invalid; keep large
    cost[:n, n] = 1e12
    cost[n + 1, :n] = 1e12
    cost[n + 1, n] = 1e12
    return cost


def _greedy_insertion_start(pt, cost, order_seed, rng):
    """Build a permutation by inserting jobs one by one at min-cost position.
    order_seed: array of job indices in the order to insert.
    """
    n = pt.shape[0]
    seq = []
    for j in order_seed:
        # evaluate insertion positions 0..len(seq)
        best_pos = 0
        best_c = 1e18
        L = len(seq)
        for pos in range(L + 1):
            if pos == 0:
                c = cost[n, j]
            elif pos == L:
                c = cost[seq[-1], n + 1]
            else:
                c = cost[seq[pos - 1], j]
            if c < best_c:
                best_c = c
                best_pos = pos
        seq.insert(best_pos, j)
    return np.array(seq, dtype=np.int64)


def _ox_crossover(a, b, rng):
    """Order crossover. a,b are job-index arrays of length n."""
    n = len(a)
    if n < 3:
        return a.copy()
    i = int(rng.integers(0, n - 1))
    j = int(rng.integers(0, n - 1))
    if i > j:
        i, j = j, i
    if j - i < 1:
        i = int(rng.integers(0, n - 1))
        j = min(i + 1, n - 1)
    child = np.full(n, -1, dtype=np.int64)
    child[i:j + 1] = a[i:j + 1]
    # fill with b's order, skipping those already placed
    placed = set(child[i:j + 1].tolist())
    pos = 0
    for x in b:
        if x not in placed:
            # find next empty slot
            while child[pos] != -1:
                pos += 1
            child[pos] = x
            pos += 1
    return child


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
    if m == 1:
        return list(range(n))

    deadline = time.time() + 145.0
    safety = 140.0
    rng = np.random.default_rng(20240607)
    mk = _makespan_vec

    # Precompute insertion cost matrix (the TSP-decomposition weights).
    cost = _insertion_cost_matrix(pt)

    # ---- Constructive phase: diverse starts ----
    neh = np.argsort(-pt.sum(axis=1))
    starts = [neh.copy()]
    # a few randomized insertion starts
    for _ in range(4):
        seed = rng.permutation(n)
        starts.append(_greedy_insertion_start(pt, cost, seed, rng))

    best_perm = None
    best_ms = 1e18
    for s in starts:
        ms = mk(pt, s)
        if ms < best_ms:
            best_ms = ms
            best_perm = s.copy()

    # ---- Local search: swap moves guided by cost matrix ----
    def local_search(perm, budget=2.0):
        nonlocal best_ms, best_perm
        cur = perm.copy()
        cur_ms = mk(pt, cur)
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < budget and time.time() < safety:
            improved = False
            for i in range(n - 1):
                if time.time() > safety:
                    break
                j = i + 1
                # cheap cost-based improvement test
                c_old = cost[cur[i - 1] if i > 0 else n, cur[i]] + cost[cur[i], cur[j]]
                c_new = cost[cur[i - 1] if i > 0 else n, cur[j]] + cost[cur[j], cur[i]]
                if c_new >= c_old - 1e-9:
                    continue
                cur[i], cur[j] = cur[j], cur[i]
                ms = mk(pt, cur)
                if ms < cur_ms - 1e-12:
                    cur_ms = ms
                    improved = True
                    if ms < best_ms:
                        best_ms = ms
                        best_perm = cur.copy()
                else:
                    cur[i], cur[j] = cur[j], cur[i]
        return cur, cur_ms

    # Run a light local search on the best start to anchor.
    if time.time() < safety:
        _, _ = local_search(best_perm, budget=1.5)

    # ---- GA phase: OX crossover + mutation + tournament ----
    pop_size = max(8, min(40, int(4000 / max(n, 1)) + 8))
    population = [best_perm.copy()]
    # seed population with a few more starts
    for _ in range(pop_size - 1):
        seed = rng.permutation(n)
        p = _greedy_insertion_start(pt, cost, seed, rng)
        population.append(p)
    scores = [mk(pt, p) for p in population]

    def tournament(pop, sc, k=3):
        idx = rng.integers(0, len(pop), size=k)
        best_i = idx[int(np.argmin(sc[i] for i in idx))]
        return pop[best_i].copy()

    gen = 0
    while time.time() < safety - 1.0:
        gen += 1
        new_pop = [best_perm.copy()]
        while len(new_pop) < pop_size and time.time() < safety - 1.0:
            pa = tournament(population, scores)
            pb = tournament(population, scores)
            child = _ox_crossover(pa, pb, rng)
            # mutation: 1-2 swap moves
            for _ in range(2):
                i = int(rng.integers(0, n - 1))
                j = int(rng.integers(0, n - 1))
                if i > j:
                    i, j = j, i
                if j - i < 1:
                    j = min(i + 1, n - 1)
                child[i], child[j] = child[j], child[i]
            # light local polish
            child, cms = local_search(child, budget=0.4)
            new_pop.append(child)
            scores_tmp = cms
        # evaluate new pop
        new_scores = [mk(pt, p) for p in new_pop]
        # keep best individuals
        combined = list(zip(new_pop, new_scores)) + list(zip(population, scores))
        combined.sort(key=lambda x: x[1])
        population = [c[0].copy() for c in combined[:pop_size]]
        scores = [c[1] for c in combined[:pop_size]]
        if scores[0] < best_ms:
            best_ms = scores[0]
            best_perm = population[0].copy()

    # Final polish
    if time.time() < safety:
        _, _ = local_search(best_perm, budget=1.0)

    return list(best_perm)
# EVOLVE-BLOCK-END
