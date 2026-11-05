"""Flow-shop permutation: NEH-style seeds + time-budgeted 2-opt / Or-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0


def _makespan(pt, order):
    """Makespan for a job permutation (flow-shop, no idle, machine in order)."""
    if order.size == 0:
        return 0.0
    return float(np.cumsum(pt[order]).max())


def _order_scores(pt, m):
    """Candidate ordering scores (larger = placed earlier)."""
    row = pt.sum(axis=1)
    col = pt.sum(axis=0)
    w = np.linspace(1.0, 0.0, m)
    palmer = -row + 2.0 * (pt @ w)          # Palmer Gantt (minimise)
    return {
        "neh": -row,
        "neh_col": -(row + col),
        "palmer": palmer,
        "palmer2": palmer - col,
    }


def _build_order(pt, scores, rng):
    """Greedy insertion: repeatedly insert the highest remaining score job at best pos."""
    n = pt.shape[0]
    order = np.empty(n, dtype=np.int64)
    remaining = list(range(n))
    # Precompute per-job prefix completion vectors to speed insertion evaluation.
    # We evaluate incrementally with a running completion array.
    comp = np.zeros(pt.shape[1])
    for k in range(n):
        # pick the remaining job with max score
        rem_arr = np.asarray(remaining)
        best_idx = int(rem_arr[int(np.argmax(scores[rem_arr]))])
        remaining.remove(best_idx)
        p = pt[best_idx]
        # find insertion position that minimises new makespan
        best_pos = 0
        best_mk = np.inf
        # completion after inserting before position pos:
        # new_comp[j] = comp[j] + sum_{i<pos} p_i + p[j] + sum_{i>=pos} p_i ... 
        # Efficient: prefix sums of p
        csum = np.cumsum(p)
        total = csum[-1]
        # candidate positions 0..k (k jobs already placed, insert at 0..k)
        # completion vector if inserted at pos:
        #   for machine j: comp[j] + total - (csum[pos-1] if pos>0 else 0) + (csum[pos-1] if pos>0 else 0)
        # Actually: machines before pos finish their jobs then this job's p[j];
        # machines >= pos: this job's p[j] then remaining jobs.
        # Simpler exact: build new completion for each candidate pos in O(m) using
        # prefix/suffix of p.
        # new_comp[j] = comp[j] + p[j] + (sum of p over jobs placed before pos that run
        #   on machine j before this job) + (sum over jobs placed after pos that run
        #   on machine j after this job). Since placed jobs keep relative order and all
        #   run in job order, machines process jobs sequentially: completion on machine j
        #   = sum over jobs in order of p[j]. Inserting at pos adds p[j] at that rank.
        # So new_comp[j] = comp[j] + p[j].  (makespan contribution independent of pos!)
        # BUT makespan = max_j new_comp[j] = max_j (comp[j] + p[j]) same for all pos.
        # That means makespan is identical regardless of insertion position -> we must
        # instead evaluate using the full completion recurrence, which depends on pos.
        # Recompute properly: completion[j] = max over jobs of (completion[j-1] of that job
        #   + p) ... we need per-job completion, not per-machine sum. Use standard DP.
        # For a small n this is fine; keep O(n*m) per insertion worst case.
        # We'll just recompute makespan for each candidate pos using DP on the sequence.
        # Sequence so far (list) + insert best_idx at pos.
        seq = list(order[:k])
        for pos in range(k + 1):
            cand = seq[:pos] + [best_idx] + seq[pos:]
            # DP makespan
            mk = 0.0
            c = 0.0
            for jj in cand:
                c += pt[jj, 0]
                mk = max(mk, c)
                # remaining machines
                cc = c
                for mm in range(1, pt.shape[1]):
                    cc = max(cc, 0) + pt[jj, mm]
                    if cc > mk:
                        mk = cc
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        # place
        seq = seq[:best_pos] + [best_idx] + seq[best_pos:]
        order[k] = best_idx
        # update running completion per machine (sum-based, valid since no idle)
        comp += pt[best_idx]
    return order


def _two_opt(pt, order, rng, deadline):
    """2-opt (reverse segment) local search; returns improved order."""
    n = order.size
    m = pt.shape[1]
    cur = _makespan(pt, order)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # randomised pair order for diversity
        pairs = np.random.permutation(n * (n - 1) // 2)
        for t in pairs:
            if time.time() >= deadline:
                break
            i = int(t // (n - 1))
            j = int(t % (n - 1))
            if j >= i:
                j += 1
            if j >= n or j <= i:
                continue
            new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
            mk = _makespan(pt, new_order)
            if mk < cur - 1e-12:
                order = new_order
                cur = mk
                improved = True
    return order, cur


def _or_opt(pt, order, rng, deadline):
    """Or-opt: relocate blocks of size 1..3 to a new position."""
    n = order.size
    cur = _makespan(pt, order)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(2000):
            if time.time() >= deadline:
                break
            block_len = int(rng.integers(1, 4))
            if block_len > n:
                block_len = n
            i = int(rng.integers(0, n - block_len + 1))
            block = order[i:i + block_len]
            rest = np.concatenate([order[:i], order[i + block_len:]])
            j = int(rng.integers(0, rest.size + 1))
            if j > i:
                j -= block_len
            new_order = np.concatenate([rest[:j], block, rest[j:]])
            mk = _makespan(pt, new_order)
            if mk < cur - 1e-12:
                order = new_order
                cur = mk
                improved = True
    return order, cur


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=float)
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
    start = time.time()
    deadline = start + (_DEADLINE - _MARGIN)
    rng = np.random.default_rng(12345)

    scores_map = _order_scores(pt, m)
    # build a handful of diverse seeds
    seeds = []
    for name in scores_map:
        try:
            o = _build_order(pt, scores_map[name], rng)
            seeds.append(o)
        except Exception:
            pass
    # a few randomised NEH seeds
    for _ in range(3):
        pert = scores_map["neh"] + rng.normal(0.0, 0.5, n)
        try:
            seeds.append(_build_order(pt, pert, rng))
        except Exception:
            pass

    best_order = None
    best_mk = np.inf
    for o in seeds:
        if time.time() >= deadline:
            break
        o2, mk = _two_opt(pt, o.copy(), rng, deadline)
        o3, mk3 = _or_opt(pt, o2, rng, deadline)
        if mk3 < best_mk:
            best_mk = mk3
            best_order = o3

    if best_order is None:
        best_order = np.argsort(-pt.sum(axis=1))
    return list(np.asarray(best_order, dtype=int))
# EVOLVE-BLOCK-END
