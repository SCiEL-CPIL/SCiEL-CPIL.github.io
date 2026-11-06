"""XOR-circuit synthesis: maximum-coverage subexpression sharing with exact
symbolic GF(2) verification, hard depth cap, and an explicit deadline guard."""
import time
import numpy as np

_T0 = time.monotonic()
_BUDGET = 78.0


def _deadline():
    return (time.monotonic() - _T0) > _BUDGET


def _baseline(matrix):
    matrix = np.asarray(matrix, dtype=np.int8)
    n, m = matrix.shape
    gates, outputs, total = [], [], 0
    for j in range(m):
        ones = np.nonzero(matrix[:, j])[0].tolist()
        if not ones:
            outputs.append(n)
            continue
        acc = ones[0]
        for x in ones[1:]:
            k = len(gates)
            gates.append([acc, x])
            acc = n + 1 + k
        outputs.append(acc)
        total += max(0, len(ones) - 1)
    return gates, outputs, total


def _verify(matrix, gates, outputs, depth_limit):
    matrix = np.asarray(matrix, dtype=np.int8)
    n, m = matrix.shape
    vecs = [None] * (n + 1 + len(gates))
    for i in range(n):
        v = np.zeros(n, dtype=np.int8)
        v[i] = 1
        vecs[i] = v
    vecs[n] = np.zeros(n, dtype=np.int8)
    depths = [1] * (n + 1)
    for k, (a, b) in enumerate(gates):
        if a < 0 or b < 0 or a >= n + 1 + k or b >= n + 1 + k or a == b:
            return False
        va, vb = vecs[a], vecs[b]
        if va is None or vb is None:
            return False
        vecs[n + 1 + k] = (va ^ vb)
        depths[n + 1 + k] = 1 + max(depths[a], depths[b])
    for j in range(m):
        w = outputs[j]
        if w is None or w < 0 or w >= len(vecs) or vecs[w] is None:
            return False
        if not np.array_equal(vecs[w], matrix[:, j]):
            return False
        if depths[w] > depth_limit:
            return False
    return True


class _Coster:
    """Memoised min-gate cost of a mask given a fixed set of free wires."""

    def __init__(self, wires, n):
        self.wires = wires
        self.n = n
        self.memo = {0: 0}
        for i in range(n):
            self.memo[1 << i] = 0

    def cost(self, mask, depth=0):
        if depth > 40:
            return bin(mask).count("1") - 1
        if mask in self.wires:
            return 0
        if mask in self.memo:
            return self.memo[mask]
        if mask & (mask - 1) == 0:
            self.memo[mask] = 0
            return 0
        best = bin(mask).count("1") - 1
        sub = (mask - 1) & mask
        while sub:
            if sub != mask:
                a, b = sub, mask ^ sub
                if a > b:
                    a, b = b, a
                tot = self.cost(a, depth + 1) + self.cost(b, depth + 1) + 1
                if tot < best:
                    best = tot
                if best <= 1:
                    break
            sub = (sub - 1) & mask
        self.memo[mask] = best
        return best


def _materialize(mask, gates, wires, n):
    if mask in wires:
        return wires[mask]
    if mask == 0:
        return n
    if mask & (mask - 1) == 0:
        return int(mask.bit_length() - 1)
    coster = _Coster(wires, n)
    best, best_cost = None, None
    sub = (mask - 1) & mask
    while sub:
        if sub != mask:
            a, b = sub, mask ^ sub
            tot = coster.cost(a) + coster.cost(b) + 1
            if best_cost is None or tot < best_cost:
                best_cost, best = tot, (a, b)
        sub = (sub - 1) & mask
    if best is None:
        bits = [b for b in range(n) if (mask >> b) & 1]
        acc = bits[0]
        for x in bits[1:]:
            k = len(gates)
            gates.append([acc, x])
            acc = n + 1 + k
        wires[mask] = acc
        return acc
    a, b = best
    wa = _materialize(a, gates, wires, n)
    wb = _materialize(b, gates, wires, n)
    k = len(gates)
    gates.append([wa, wb])
    acc = n + 1 + k
    wires[mask] = acc
    return acc


def _build(matrix, depth_limit):
    matrix = np.asarray(matrix, dtype=np.int8)
    n, m = matrix.shape
    if n == 0 or m == 0:
        return _baseline(matrix)

    omasks = np.zeros(m, dtype=np.int64)
    for i in range(n):
        omasks += matrix[i].astype(np.int64) << i
    omasks = omasks.tolist()

    cands = []
    for s in range(2, min(n, 5) + 1):
        for mask in range(1, 1 << n):
            if bin(mask).count("1") == s:
                cands.append(mask)

    oa = np.array(omasks, dtype=np.int64)
    contain = np.zeros((len(cands), m), dtype=bool)
    for ci, c in enumerate(cands):
        contain[ci] = (oa & c) == c

    wires = {}
    gates = []
    used = np.zeros(len(cands), dtype=bool)

    while not _deadline():
        best_ci, best_gain = -1, 0
        coster = _Coster(wires, n)
        for ci in np.where(~used)[0]:
            c = cands[ci]
            idx = np.where(contain[ci])[0]
            if len(idx) == 0:
                continue
            gain = 0
            for j in idx:
                om = omasks[j]
                before = coster.cost(om)
                rest = om ^ c
                after = 0 if rest == 0 else coster.cost(rest) + 1
                if before > after:
                    gain += before - after
            if gain > best_gain:
                best_gain, best_ci = gain, ci
        if best_ci < 0 or best_gain <= 0:
            break
        used[best_ci] = True
        _materialize(cands[best_ci], gates, wires, n)

    outputs = [_materialize(omasks[j], gates, wires, n) for j in range(m)]
    return gates, outputs


def synthesize(matrix, depth_limit):
    matrix = np.asarray(matrix, dtype=np.int8)
    n, m = matrix.shape
    depth_limit = int(depth_limit) if depth_limit else 16

    bgates, boutputs, btotal = _baseline(matrix)
    best = (bgates, boutputs)

    if not _deadline():
        try:
            gates, outputs = _build(matrix, depth_limit)
            if _verify(matrix, gates, outputs, depth_limit) and len(gates) <= btotal:
                best = (gates, outputs)
        except Exception:
            pass

    gates, outputs = best
    return {'gates': gates, 'outputs': outputs}
