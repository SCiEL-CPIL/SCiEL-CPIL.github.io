import time
import random


def synthesize(matrix, depth_limit):
    start = time.time()
    DEADLINE = start + 50.0  # safety margin under 60s/90s wall-clock

    M = [list(row) for row in matrix]
    m = len(M)
    n = len(M[0]) if m else 0
    if m == 0:
        return {'gates': [], 'outputs': []}

    # Row bitmasks over the n input wires.
    rows = []
    for r in M:
        v = 0
        for j in range(n):
            if r[j]:
                v |= (1 << j)
        rows.append(v)

    required = set(v for v in rows if v != 0)
    base_vals = [1 << j for j in range(n)] + [0]  # wires 0..n-1 inputs, n = zero

    def simulate(gates):
        vals = base_vals[:]
        for (a, b) in gates:
            vals.append(vals[a] ^ vals[b])
        return vals

    def verify(gates, outputs):
        vals = simulate(gates)
        if len(outputs) != len(rows):
            return False
        for o, v in zip(outputs, rows):
            if o < 0 or o >= len(vals) or vals[o] != v:
                return False
        return True

    def independent():
        """Guaranteed-correct fallback: compute each row directly (sharing equal rows)."""
        gates = []
        idmap = {wid: wid for wid in range(n + 1)}
        next_id = n + 1
        row2id = {}
        outputs = []
        for v in rows:
            if v == 0:
                outputs.append(n)
                continue
            if v in row2id:
                outputs.append(row2id[v])
                continue
            bits = [j for j in range(n) if (v >> j) & 1]
            if len(bits) == 1:
                wid = bits[0]
            else:
                acc = bits[0]
                for j in bits[1:]:
                    gates.append((idmap[acc], idmap[j]))
                    idmap[next_id] = next_id
                    acc = next_id
                    next_id += 1
                wid = acc
            row2id[v] = wid
            outputs.append(wid)
        return gates, outputs

    def prune(gates):
        """Drop gates whose outputs are not (transitively) used by any required row; renumber."""
        vals = simulate(gates)
        val2id = {}
        for idx, val in enumerate(vals):
            if val in required and val not in val2id:
                val2id[val] = idx
        need = [False] * len(vals)
        for v in required:
            if v in val2id:
                need[val2id[v]] = True
        gneed = [False] * len(gates)
        for _ in range(len(gates) + 1):
            progress = False
            for g in range(len(gates) - 1, -1, -1):
                out = n + 1 + g
                if need[out] and not gneed[g]:
                    gneed[g] = True
                    a, b = gates[g]
                    if not need[a]:
                        need[a] = True
                        progress = True
                    if not need[b]:
                        need[b] = True
                        progress = True
            if not progress:
                break
        new_gates = []
        idmap = {wid: wid for wid in range(n + 1)}
        next_id = n + 1
        for g in range(len(gates)):
            if gneed[g]:
                a, b = gates[g]
                new_gates.append((idmap[a], idmap[b]))
                idmap[n + 1 + g] = next_id
                next_id += 1
        vals2 = simulate(new_gates)
        val2id2 = {}
        for idx, val in enumerate(vals2):
            if val in required and val not in val2id2:
                val2id2[val] = idx
        outputs = []
        for v in rows:
            if v == 0:
                outputs.append(n)
            else:
                outputs.append(val2id2.get(v, -1))
        return new_gates, outputs

    def build(key, seed):
        rng = random.Random(seed)
        avail = base_vals[:]
        avail_set = set(avail)
        gates = []
        remaining = set(v for v in required if v not in avail_set)
        max_steps = (1 << n) + 10
        step = 0
        while remaining and step < max_steps:
            best = None
            best_key = None
            L = len(avail)
            for i in range(L):
                a = avail[i]
                for j in range(i + 1, L):
                    b = avail[j]
                    v = a ^ b
                    if v in avail_set:
                        continue
                    covers = 1 if v in required else 0
                    helps = 0
                    for r in remaining:
                        if (r ^ v) in avail_set:
                            helps += 1
                    k = key(covers, helps, v, rng)
                    if best_key is None or k > best_key:
                        best_key = k
                        best = (i, j, v)
                if time.time() > DEADLINE:
                    break
            if best is None:
                break
            i, j, v = best
            avail.append(v)
            avail_set.add(v)
            gates.append((i, j))
            if v in required:
                remaining.discard(v)
            step += 1
            if time.time() > DEADLINE:
                break
        if remaining:
            return None
        val2id = {}
        for idx, val in enumerate(avail):
            if val not in val2id:
                val2id[val] = idx
        outputs = []
        for v in rows:
            if v == 0:
                outputs.append(n)
            else:
                outputs.append(val2id.get(v, -1))
        if any(o < 0 for o in outputs):
            return None
        return gates, outputs

    def keyA(c, h, v, rng):
        return (c, h)
    def keyB(c, h, v, rng):
        return (c, h, -v.bit_count())
    def keyC(c, h, v, rng):
        return (h, c)
    def keyD(c, h, v, rng):
        return (c * 4 + h,)
    def keyE(c, h, v, rng):
        return (c, h, rng.random())
    def keyF(c, h, v, rng):
        return (c * 2 + h, -v.bit_count())

    variants = [keyA, keyB, keyC, keyD, keyE, keyF]
    best = None
    try:
        for vi, key in enumerate(variants):
            if time.time() > DEADLINE:
                break
            res = build(key, seed=vi)
            if res is None:
                continue
            gates, outputs = res
            pg, po = prune(gates)
            if verify(pg, po):
                if best is None or len(pg) < best[0]:
                    best = (len(pg), pg, po)
    except Exception:
        best = None

    if best is None:
        pg, po = independent()
        return {'gates': pg, 'outputs': po}
    return {'gates': best[1], 'outputs': best[2]}
