"""Exact XOR-circuit synthesis: GF(2) iterative CSE + depth-16 feasibility + deadline guard."""
import time
import numpy as np


def _row_to_vec(row):
    v = 0
    for j, b in enumerate(row):
        if b:
            v |= (1 << j)
    return v


def _decompose(vec):
    """Split bit-vector into two non-empty parts a,b with a^b=vec."""
    low = vec & 1
    if low == 0:
        low = 1 << (vec.bit_length() - 1)
    return low, vec ^ low


def _build_circuit(extracted, cur_vecs, n):
    """extracted: operand vectors in CSE creation order (each becomes a gate).
    cur_vecs: final per-output vectors. Returns (gates, outputs) with gates=[w1,w2]."""
    gates = []
    wire_of = {0: n}
    next_w = n + 1

    def get_wire(x):
        nonlocal next_w
        if x in wire_of:
            return wire_of[x]
        if x == 0:
            return n
        if x & (x - 1) == 0:
            j = x.bit_length() - 1
            if j < n:
                wire_of[x] = j
                return j
        aa, bb = _decompose(x)
        wa = get_wire(aa)
        wb = get_wire(bb)
        w = next_w
        gates.append([wa, wb])
        wire_of[x] = w
        next_w += 1
        return w

    for op in extracted:
        a, b = _decompose(op)
        wa = get_wire(a)
        wb = get_wire(b)
        w = next_w
        gates.append([wa, wb])
        wire_of[op] = w
        next_w += 1
    outputs = [get_wire(v) for v in cur_vecs]
    return gates, outputs


def _depth_feasible(gates, n, depth_limit):
    if not gates:
        return True
    depth = {}
    for k in range(len(gates)):
        w1, w2 = gates[k]
        d1 = 0 if w1 <= n else depth.get(w1 - (n + 1), -1) + 1
        d2 = 0 if w2 <= n else depth.get(w2 - (n + 1), -1) + 1
        d = max(d1, d2) + 1
        depth[k] = d
        if d > depth_limit:
            return False
    return True


def _cse_extract(vecs, n, deadline, iters=600):
    """Returns (extracted, cur_vecs) or None on deadline."""
    cur = list(vecs)
    extracted = []
    used = set()
    for _ in range(iters):
        if time.time() >= deadline:
            return None
        counts = {}
        for v in cur:
            if v == 0:
                continue
            sub = v
            while sub:
                if sub not in used:
                    counts[sub] = counts.get(sub, 0) + 1
                sub &= (sub - 1)
        best = None
        best_c = 1
        for c, cnt in counts.items():
            if cnt > best_c:
                best_c = cnt
                best = c
        if best is None:
            break
        new_cur = []
        for v in cur:
            new_cur.append(v ^ best if (v & best) else v)
        cur = new_cur
        extracted.append(best)
        used.add(best)
    return extracted, cur


def synthesize(matrix, depth_limit):
    t0 = time.time()
    deadline = t0 + 84.0
    M = np.asarray(matrix)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    m, n = M.shape
    vecs = [_row_to_vec(M[i]) for i in range(m)]

    # baseline trivial circuit
    bg, bo = _build_circuit([], vecs, n)
    best = (len(bg), bg, bo) if _depth_feasible(bg, n, depth_limit) else None

    if best is None:
        # even trivial infeasible (shouldn't happen for depth>=1); return minimal
        return {'gates': [], 'outputs': [n] * m}

    if time.time() < deadline:
        res = _cse_extract(vecs, n, deadline)
        if res is not None:
            extracted, cur = res
            cg, co = _build_circuit(extracted, cur, n)
            if _depth_feasible(cg, n, depth_limit) and len(cg) < best[0]:
                best = (len(cg), cg, co)

    return {'gates': best[1], 'outputs': best[2]}
