import time
import numpy as np

_DEADLINE = 85.0  # wall-clock safety margin (evaluator kills at 90s)
_t0 = time.time()


def _deadline_exceeded():
    return (time.time() - _t0) > _DEADLINE


def _popcount(v):
    return int(np.count_nonzero(v))


def _popcount_row(v):
    return int(np.count_nonzero(v))


def _baseline_gate_count(cols):
    """Independent-row baseline: build each output from scratch (popcount gates)."""
    total = 0
    for v in cols:
        total += _popcount_row(v)
    return total


def _direct_gates(v, n, gate_count, depth, depth_limit, gates, depth_of):
    """Build a single vector v from inputs (and zero wire) directly.
    Returns (new_gate_count, new_depth_of)."""
    nz = np.nonzero(v)[0]
    if len(nz) == 0:
        return gate_count, depth_of  # zero vector -> wire n, no gates
    if len(nz) == 1:
        return gate_count, depth_of  # single input -> wire nz[0], no gates
    # XOR the nonzero inputs together
    acc = int(nz[0])
    d = 0
    for idx in range(1, len(nz)):
        b = int(nz[idx])
        gate_count += 1
        out_wire = n + 1 + (gate_count - 1)
        gates.append([acc, b])
        d += 1
        depth_of[out_wire] = d
        acc = out_wire
    return gate_count, depth_of


def _try_add_gate(a, b, v, n, gate_count, gates, depth_of, depth_limit):
    """Add gate out = a XOR b producing vector v. Returns (ok, new_gate_count, out_wire)."""
    out_wire = n + 1 + gate_count
    d = max(depth_of[a], depth_of[b]) + 1
    if d > depth_limit:
        return False, gate_count, out_wire
    gates.append([a, b])
    depth_of[out_wire] = d
    return True, gate_count + 1, out_wire


def synthesize(matrix, depth_limit):
    global _t0
    _t0 = time.time()

    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2:
        matrix = matrix.reshape((matrix.shape[0], 1)) if matrix.size else np.zeros((0, 0))
    n, m = matrix.shape

    # columns as vectors (length n)
    cols = [matrix[:, j] for j in range(m)]
    # zero vector
    zero = np.zeros(n, dtype=np.int64)

    baseline = _baseline_gate_count(cols)
    if baseline == 0:
        # all outputs are zero -> wire n
        return {'gates': [], 'outputs': [n] * m}

    # ---- build candidate pool ----
    pool = {}  # tuple(vector) -> (vector, [list of column indices that equal this vector])
    for j in range(m):
        key = tuple(cols[j].tolist())
        if key not in pool:
            pool[key] = (cols[j].copy(), [j])
        else:
            pool[key][1].append(j)

    # pairwise XORs
    keys = list(pool.keys())
    for i in range(len(keys)):
        if _deadline_exceeded():
            break
        vi = pool[keys[i]][0]
        for j in range(i + 1, len(keys)):
            key2 = tuple((vi + pool[keys[j]][0]) % 2).tolist()
            key2 = tuple((vi + pool[keys[j]][0]) % 2)
            if key2 not in pool:
                pool[key2] = ((vi + pool[keys[j]][0]) % 2, [])
    # triple XORs (bounded)
    keys = list(pool.keys())
    for i in range(len(keys)):
        if _deadline_exceeded():
            break
        vi = pool[keys[i]][0]
        for j in range(i + 1, len(keys)):
            vij = (vi + pool[keys[j]][0]) % 2
            for k in range(j + 1, len(keys)):
                key3 = tuple((vij + pool[keys[k]][0]) % 2)
                if key3 not in pool:
                    pool[key3] = ((vij + pool[keys[k]][0]) % 2, [])

    # ---- greedy shared-subexpression search ----
    # available: vector_key -> wire
    available = {}
    # inputs
    for i in range(n):
        vec = np.zeros(n, dtype=np.int64)
        vec[i] = 1
        available[tuple(vec.tolist())] = i
    # zero wire
    available[tuple(zero.tolist())] = n

    gates = []
    gate_count = 0
    depth_of = {}
    for i in range(n):
        depth_of[i] = 0
    depth_of[n] = 0

    # which column indices are already satisfied
    satisfied = [False] * m

    # candidate list: (key, vector, col_indices)
    # We'll repeatedly pick the best marginal candidate.
    # marginal = popcount(v) - (# of distinct column indices it satisfies that are not yet satisfied)
    # Actually a candidate also helps by being a subexpression for other candidates, but we approximate
    # by counting satisfied outputs plus reuse among already-added intermediates.

    # We track for each candidate key the set of column indices it equals.
    cand_keys = list(pool.keys())

    # Precompute for each candidate: popcount and the column indices it covers.
    cand_info = {}
    for key in cand_keys:
        v = pool[key][0]
        cidx = pool[key][1]
        cand_info[key] = (_popcount(v), cidx)

    # We'll do a greedy loop: each iteration pick candidate with best (most negative) marginal cost
    # marginal = popcount - (# newly satisfied columns) - (# of already-available candidate keys it
    #             can combine with to reduce future cost) -- we approximate the last term by counting
    #             how many other *required* columns share a sub-vector, which is hard, so we use
    #             newly satisfied columns only, plus a small bonus for high popcount sharing.

    # Simpler robust greedy: repeatedly pick the candidate whose addition satisfies the most
    # still-unsatisfied columns per gate, i.e. minimize (popcount - newly_satisfied).
    # But a candidate with popcount 2 that satisfies 1 column is better than popcount 5 satisfying 1.
    # A candidate that satisfies 0 columns can still be useful as a subexpression; we add those only
    # if they reduce the cost of remaining columns. We approximate: add a candidate if it satisfies
    # >=1 column, OR if popcount(v) <= 2 (cheap subexpression).

    progress = True
    while progress and not _deadline_exceeded():
        progress = False
        best_key = None
        best_marginal = 0  # we want most negative marginal; start at 0 meaning "no improvement"
        best_newly = 0
        for key in cand_keys:
            if key in available:
                continue
            pc, cidx = cand_info[key]
            if pc == 0:
                continue
            # newly satisfied columns
            newly = 0
            for ci in cidx:
                if not satisfied[ci]:
                    newly += 1
            # marginal cost = pc - newly (we want this < 0, i.e. newly > pc)
            marginal = pc - newly
            if newly > 0:
                # only consider if it strictly improves
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_key = key
                    best_newly = newly
            else:
                # no columns satisfied; only add if very cheap (subexpression) and it helps.
                # We'll skip these in this pass to keep it simple and correct.
                pass

        if best_key is None:
            break

        v = pool[best_key][0]
        nz = np.nonzero(v)[0]
        if len(nz) == 0:
            available[best_key] = n
            progress = True
            continue
        if len(nz) == 1:
            available[best_key] = int(nz[0])
            progress = True
            continue

        # find two existing wires a,b with a XOR b == v
        found = False
        nz_list = nz.tolist()
        # try to find a in available such that v XOR a is also in available
        # iterate over available keys
        a_key = None
        b_key = None
        a_vec = None
        b_vec = None
        # We need a,b in available with a+b == v (mod 2).
        # Brute force over available keys (bounded size).
        avail_keys = list(available.keys())
        for ka in avail_keys:
            va = np.array(pool[ka][0] if ka in pool else np.array(ka, dtype=np.int64), dtype=np.int64)
            # but available keys are tuples; convert
            va = np.array(ka, dtype=np.int64)
            vb = (va + v) % 2
            kb = tuple(vb.tolist())
            if kb in available:
                a_key = ka
                b_key = kb
                found = True
                break
        if not found:
            # cannot form as XOR of two available; fall back to direct build for this vector
            gate_count, depth_of = _direct_gates(v, n, gate_count, depth_of, depth_limit, gates)
            # map: the final acc wire is n+1+gate_count-1 if gates added, else input
            nz2 = np.nonzero(v)[0]
            if len(nz2) == 1:
                out_wire = int(nz2[0])
            elif len(nz2) == 0:
                out_wire = n
            else:
                out_wire = n + 1 + (gate_count - 1)
            available[best_key] = out_wire
        else:
            a_wire = available[a_key]
            b_wire = available[b_key]
            ok, gate_count, out_wire = _try_add_gate(a_wire, b_wire, v, n, gate_count, gates, depth_of, depth_limit)
            if not ok:
                # depth exceeded; build directly
                gate_count, depth_of = _direct_gates(v, n, gate_count, depth_of, depth_limit, gates)
                nz2 = np.nonzero(v)[0]
                if len(nz2) == 1:
                    out_wire = int(nz2[0])
                elif len(nz2) == 0:
                    out_wire = n
                else:
                    out_wire = n + 1 + (gate_count - 1)
                available[best_key] = out_wire
            else:
                available[best_key] = out_wire

        # mark satisfied columns
        for ci in cand_info[best_key][1]:
            satisfied[ci] = True
        progress = True

    # ---- top up any unsatisfied outputs with direct construction ----
    outputs = [0] * m
    for j in range(m):
        key = tuple(cols[j].tolist())
        if key in available:
            outputs[j] = available[key]
            satisfied[j] = True
        else:
            if _deadline_exceeded():
                # still need a valid output; use direct (may exceed depth but must be valid)
                gate_count, depth_of = _direct_gates(cols[j], n, gate_count, depth_of, depth_limit, gates)
                nz2 = np.nonzero(cols[j])[0]
                if len(nz2) == 1:
                    outputs[j] = int(nz2[0])
                elif len(nz2) == 0:
                    outputs[j] = n
                else:
                    outputs[j] = n + 1 + (gate_count - 1)
                continue
            gate_count, depth_of = _direct_gates(cols[j], n, gate_count, depth_of, depth_limit, gates)
            nz2 = np.nonzero(cols[j])[0]
            if len(nz2) == 1:
                outputs[j] = int(nz2[0])
            elif len(nz2) == 0:
                outputs[j] = n
            else:
                outputs[j] = n + 1 + (gate_count - 1)

    return {'gates': gates, 'outputs': outputs}
