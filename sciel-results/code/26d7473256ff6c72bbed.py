import time

def synthesize(matrix, depth_limit):
    start = time.time()
    deadline = start + 75.0

    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])
    zero_wire = n

    rows = []
    for r in range(m):
        mask = 0
        for c in range(n):
            if matrix[r][c]:
                mask |= (1 << c)
        rows.append(mask)

    gates = []
    next_wire = [n + 1]

    def add_gate(a, b):
        out = next_wire[0]
        next_wire[0] += 1
        gates.append([a, b])
        return out

    # Build basis via elimination
    basis = []  # (pivot_bit, mask, wire)
    for mask in rows:
        cur = mask
        for (pb, bm, w) in basis:
            if (cur >> pb) & 1:
                cur ^= bm
        if cur != 0:
            pb = cur.bit_length() - 1
            bits = [i for i in range(n) if (cur >> i) & 1]
            if len(bits) == 1:
                w = bits[0]
            else:
                w = bits[0]
                for i in bits[1:]:
                    w = add_gate(w, i)
            basis.append((pb, cur, w))

    outputs = []
    for mask in rows:
        cur = mask
        combo = []
        for idx, (pb, bm, w) in enumerate(basis):
            if (cur >> pb) & 1:
                cur ^= bm
                combo.append(idx)
        if not combo:
            outputs.append(zero_wire)
        elif len(combo) == 1:
            outputs.append(basis[combo[0]][2])
        else:
            w = basis[combo[0]][2]
            for idx in combo[1:]:
                w = add_gate(w, basis[idx][2])
            outputs.append(w)

    return {'gates': gates, 'outputs': outputs}
