"""XOR-circuit synthesis: greedy CSE with shared intermediates + depth guard."""
import time

_DEADLINE = time.time() + 75.0  # safety margin under 90s wall-clock


def _independent(matrix, depth_limit):
    """Independent per-row synthesis (the baseline)."""
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    gates = []
    outputs = []
    for r in range(m):
        ones = [c for c in range(n) if matrix[r][c]]
        acc = None
        for c in ones:
            if acc is None:
                acc = c
            else:
                acc = n + 1 + len(gates)
                gates.append([acc, c])
        outputs.append(acc if acc is not None else n)
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    t0 = time.time()
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    if n == 0:
        return {'gates': [], 'outputs': []}
    if depth_limit < 1:
        return _independent(matrix, depth_limit)

    base = _independent(matrix, depth_limit)

    # Each row as a set of wires
    row_sets = [frozenset(c for c in range(n) if matrix[r][c]) for r in range(m)]
    row_depth = [len(s) - 1 for s in row_sets]

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    used = set()
    gates = []
    next_wire = n + 1

    def benefit(i, j):
        b = 0
        for s in row_sets:
            if i in s and j in s:
                b += 1
        return b

    while pairs:
        if time.time() > _DEADLINE:
            break
        best = None
        best_b = 1  # need benefit >= 2 to save a gate
        for (i, j) in pairs:
            if (i, j) in used:
                continue
            b = benefit(i, j)
            if b < 2:
                continue
            ok = True
            for r in range(m):
                s = row_sets[r]
                if i in s and j in s:
                    if row_depth[r] + 1 > depth_limit:
                        ok = False
                        break
            if not ok:
                continue
            if b > best_b or (b == best_b and best is None):
                best_b = b
                best = (i, j)
        if best is None:
            break
        i, j = best
        used.add((i, j))
        w = next_wire
        next_wire += 1
        gates.append([i, j])  # produces wire w = i XOR j

        for r in range(m):
            s = row_sets[r]
            if i in s and j in s:
                row_sets[r] = frozenset((s - {i, j}) | {w})
                row_depth[r] += 1

    outputs = []
    for s in row_sets:
        lst = sorted(s)
        if not lst:
            outputs.append(n)
        elif len(lst) == 1:
            outputs.append(lst[0])
        else:
            acc = lst[0]
            for c in lst[1:]:
                w = next_wire
                next_wire += 1
                gates.append([acc, c])
                acc = w
            outputs.append(acc)

    # Verify depth
    depth_of = [0] * (n + 1)
    for k in range(len(gates)):
        a, b = gates[k]
        depth_of.append(max(depth_of[a], depth_of[b]) + 1)
    if len(gates) and max(depth_of) > depth_limit:
        return base

    if len(gates) >= len(base['gates']):
        return base

    return {'gates': gates, 'outputs': outputs}
