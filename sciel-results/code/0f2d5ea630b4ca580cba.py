"""Flow-shop permutation via multi-start heuristics + Variable Neighborhood Descent (VND)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _row_next(prev, p, m):
    """Completion vector of current job given previous job's completion vector."""
    out = np.empty(m)
    out[0] = prev[0] + p[0]
    for k in range(1, m):
        a = prev[k]
        b = out[k - 1]
        out[k] = (a if a > b else b) + p[k]
    return out


def _makespan(pt_t, perm):
    """pt_t: (n_machines, n_jobs) transposed. perm: sequence of job indices."""
    m = pt_t.shape[0]
    n = len(perm)
    if n == 0:
        return 0.0
    prev = np.zeros(m)
    for j in range(n):
        p = pt_t[:, perm[j]]
        prev = _row_next(prev, p, m)
    return float(prev[-1])


def _neh_initial(pt):
    """NEH heuristic: order by total processing desc, insert each at best position."""
    n_jobs, n_machines = pt.shape
    pt_t = pt.T
    order = list(np.argsort(-pt.sum(axis=1)))
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt_t, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    return list(np.argsort(-pt.sum(axis=1)))


def _spt_initial(pt):
    return list(np.argsort(pt.sum(axis=1)))


def _bottleneck_initial(pt):
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _insertion_ls(pt_t, perm, deadline, max_rounds=20000):
    """Insertion neighborhood local search (first-improvement per pass)."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(pt_t, perm)
    cur = list(perm)
    cur_ms = _makespan(pt_t, cur)
    best = list(cur)
    best_ms = cur_ms
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = cur[i]
            rest = cur[:i] + cur[i + 1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                test = rest[:j] + [job] + rest[j:]
                ms = _makespan(pt_t, test)
                if ms < cur_ms - 1e-9:
                    cur = test
                    cur_ms = ms
                    improved = True
                    if cur_ms < best_ms - 1e-9:
                        best_ms = cur_ms
                        best = list(cur)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _reversal_ls(pt_t, perm, deadline, max_rounds=20000):
    """2-opt segment reversal neighborhood local search."""
    n = len(perm)
    if n <= 2:
        return perm, _makespan(pt_t, perm)
    cur = list(perm)
    cur_ms = _makespan(pt_t, cur)
    best = list(cur)
    best_ms = cur_ms
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                test = cur[:i] + cur[i:j + 1][::-1] + cur[j + 1:]
                ms = _makespan(pt_t, test)
                if ms < cur_ms - 1e-9:
                    cur = test
                    cur_ms = ms
                    improved = True
                    if cur_ms < best_ms - 1e-9:
                        best_ms = cur_ms
                        best = list(cur)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _block_ls(pt_t, perm, deadline, max_rounds=20000):
    """Block (2-block move) neighborhood local search."""
    n = len(perm)
    if n <= 3:
        return perm, _makespan(pt_t, perm)
    cur = list(perm)
    cur_ms = _makespan(pt_t, cur)
    best = list(cur)
    best_ms = cur_ms
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 2):
            if time.time() > deadline:
                break
            for j in range(i + 1, n - 1):
                block = cur[i:j + 1]
                rest = cur[:i] + cur[j + 1:]
                for k in range(len(rest) + 1):
                    if k == i or k == i + 1:
                        continue
                    test = rest[:k] + block + rest[k:]
                    ms = _makespan(pt_t, test)
                    if ms < cur_ms - 1e-9:
                        cur = test
                        cur_ms = ms
                        improved = True
                        if cur_ms < best_ms - 1e-9:
                            best_ms = cur_ms
                            best = list(cur)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _vnd(pt_t, perm, deadline):
    """Variable Neighborhood Descent: cycle insertion -> reversal -> block,
    expand radius until no neighborhood yields an improvement."""
    cur = list(perm)
    cur_ms = _makespan(pt_t, cur)
    best = list(cur)
    best_ms = cur_ms
    neighborhoods = [_insertion_ls, _reversal_ls, _block_ls]
    while time.time() < deadline:
        improved_any = False
        for fn in neighborhoods:
            if time.time() > deadline:
                break
            new_perm, new_ms = fn(pt_t, cur, deadline)
            if new_ms < cur_ms - 1e-9:
                cur = new_perm
                cur_ms = new_ms
                improved_any = True
                if cur_ms < best_ms - 1e-9:
                    best_ms = cur_ms
                    best = list(cur)
        if not improved_any:
            break
    return best, best_ms


def _simulated_annealing(pt_t, perm, deadline, max_iter=40000, seed=0):
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan(pt_t, perm)
    cur = list(perm)
    cur_ms = _makespan(pt_t, cur)
    best = list(cur)
    best_ms = cur_ms
    T = max(cur_ms * 0.05, 1.0)
    T_min = max(cur_ms * 0.0005, 0.01)
    cooling = 0.995
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = cur[i]
        rest = cur[:i] + cur[i + 1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan(pt_t, test)
        delta = new_ms - cur_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            cur = test
            cur_ms = new_ms
            if cur_ms < best_ms:
                best_ms = cur_ms
                best = list(cur)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_t = pt.T  # (n_machines, n_jobs)

    # Diverse constructive starts
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    for s in range(4):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, keep best
    best_ms = float('inf')
    best_perm = candidates[0]
    for cand in candidates:
        ms = _makespan(pt_t, cand)
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # VND on the best candidate
    perm, ms = _vnd(pt_t, best_perm, deadline - 20.0)
    if ms < best_ms:
        best_ms = ms
        best_perm = perm

    # VND on other promising candidates while time remains
    for cand in candidates[1:]:
        if time.time() > deadline - 15.0:
            break
        ms0 = _makespan(pt_t, cand)
        if ms0 > best_ms * 1.15:
            continue
        perm, ms = _vnd(pt_t, cand, deadline - 15.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # SA polish to escape local optima
    if time.time() < deadline - 5.0:
        perm, ms = _simulated_annealing(pt_t, best_perm, deadline - 5.0, max_iter=40000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final insertion polish
    if time.time() < deadline - 2.0:
        perm, ms = _insertion_ls(pt_t, best_perm, deadline - 1.0)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
