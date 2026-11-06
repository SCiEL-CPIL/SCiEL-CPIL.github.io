"""Rewritten program: benefit-driven XOR-circuit synthesis with subexpression sharing."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    rows = [list(r) for r in matrix]
    m = len(rows)
    n = len(rows[0]) if m else 0
    if n == 0:
        return {'gates': [], 'outputs': []}

    targets = []
    for i in range(m):
        v = 0
        row = rows[i]
        for j in range(n):
            if row[j]:
                v |= (1 << j)
        targets.append(v)

    available = {}
    depth = {}
    for i in range(n):
        available[1 << i] = i
        depth[i] = 0
    available[0] = n
    depth[n] = 0

    gates = []
    next_wire = n + 1

    def new_gate(a, b):
        nonlocal next_wire
        w = next_wire
        next_wire += 1
        gates.append([a, b])
        depth[w] = max(depth[a], depth[b]) + 1
        return w

    remaining = set(targets)
    remaining.discard(0)

    MAX_PAIR_LIST = 250
    MAX_ITERS = 10000
    it = 0
    while remaining and it < MAX_ITERS:
        it += 1
        avail_list = sorted(available.keys(), key=lambda x: (x.bit_count(), x))
        if len(avail_list) > MAX_PAIR_LIST:
            avail_list = avail_list[:MAX_PAIR_LIST]
        cand_pairs = {}
        L = len(avail_list)
        for i in range(L):
            a = avail_list[i]
            da = depth[available[a]]
            for j in range(i + 1, L):
                b = avail_list[j]
                z = a ^ b
                if z in available:
                    continue
                if max(da, depth[available[b]]) + 1 > 16:
                    continue
                if z not in cand_pairs:
                    cand_pairs[z] = (a, b)
        best_z = None
        best_benefit = 0
        for z, (a, b) in cand_pairs.items():
            benefit = 0
            for t in remaining:
                before = t.bit_count() - 1
                after = (t ^ z).bit_count()
                if before > after:
                    benefit += before - after
            if benefit > best_benefit:
                best_benefit = benefit
                best_z = z
        if best_z is None:
            t = min(remaining, key=lambda x: (x.bit_count(), x))
            bits = [1 << j for j in range(n) if (t >> j) & 1]
            w = n if not bits else bits[0]
            for bit in bits[1:]:
                w = new_gate(w, bit)
            available[t] = w
            remaining.discard(t)
        else:
            a, b = cand_pairs[best_z]
            w = new_gate(available[a], available[b])
            available[best_z] = w
            remaining.discard(best_z)

    outputs = [available[t] for t in targets]
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
