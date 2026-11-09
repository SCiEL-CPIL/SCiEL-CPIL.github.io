"""XOR-circuit synthesis via minimum-weight perfect matching + CSE + local search."""
import time
import numpy as np

# EVOLVE-BLOCK-START

_TIME_LIMIT = 70.0  # safety margin under 90s wall-clock
_DEADLINE = None


def _now():
    return time.time()


def _remaining():
    global _DEADLINE
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - _now())


def _bit_vector_to_int(vec):
    """Convert a 1-D numpy bool/int array (row of matrix) to an int bit-vector."""
    n = vec.shape[0]
    x = 0
    # pack bits; use int for arbitrary width
    for i, v in enumerate(vec):
        if v:
            x |= (1 << i)
    return x


def _int_to_bit_vector(x, n):
    return np.array([(x >> i) & 1 for i in range(n)], dtype=np.int64)


def _popcount(x):
    return bin(x).count("1")


def _hamming(a, b):
    return _popcount(a ^ b)


def _verify(matrix, gates, outputs, n):
    """Exact GF(2) verification. matrix: (m, n) target. gates: [[wire, wire], ...].
    outputs: list of wire indices. Returns True if outputs reproduce matrix rows."""
    m = matrix.shape[0]
    # wire value: int bit-vector over n input bits
    wire_val = [0] * (n + 1 + len(gates))
    # inputs 0..n-1 are unit vectors
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0  # constant zero
    for k, (a, b) in enumerate(gates):
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
    # outputs must equal the m target rows
    for r in range(m):
        target = _bit_vector_to_int(matrix[r])
        if outputs[r] >= len(wire_val):
            return False
        if wire_val[outputs[r]] != target:
            return False
    return True


def _depth_ok(gates, outputs, n):
    """Compute depth (longest path) of the DAG; inputs depth 0, gate depth = max+1."""
    m = len(gates)
    depth = [0] * (n + 1 + m)
    # inputs and zero are depth 0
    for k in range(m):
        a, b = gates[k]
        depth[n + 1 + k] = max(depth[a], depth[b]) + 1
    maxd = 0
    for o in outputs:
        if o < len(depth):
            maxd = max(maxd, depth[o])
    return maxd


def _cse(gates, n):
    """Common subexpression elimination: replace duplicate XORs, remap wires.
    Returns (new_gates, remap) where remap[old_wire]=new_wire."""
    # map (a,b) sorted -> gate index producing it
    seen = {}
    remap = list(range(n + 1 + len(gates)))  # identity start
    new_gates = []
    for k, (a, b) in enumerate(gates):
        wa, wb = remap[a], remap[b]
        if wa == wb:
            remap[n + 1 + k] = n  # zero
            continue
        key = (min(wa, wb), max(wa, wb))
        if key in seen:
            remap[n + 1 + k] = n + 1 + seen[key]
            continue
        new_gates.append([wa, wb])
        remap[n + 1 + k] = n + 1 + len(new_gates) - 1
        seen[key] = len(new_gates) - 1
    # compact: build final wire list, remap again to dense
    used = set()
    for g in new_gates:
        used.add(g[0]); used.add(g[1])
    # outputs handled by caller; we return gates + a wire->dense map for inputs/zero
    return new_gates, remap


def _baseline_gate_count(matrix, n):
    """Independent-row baseline: each output row built independently by XOR of its set bits."""
    total = 0
    for r in range(matrix.shape[0]):
        total += _popcount(_bit_vector_to_int(matrix[r])) - 1
    return max(total, 0)


def _greedy_construct(matrix, n, deadline_fn):
    """Greedy: repeatedly create the new wire that best reduces total remaining distance."""
    m = matrix.shape[0]
    targets = [_bit_vector_to_int(matrix[r]) for r in range(m)]
    # available wires: inputs + zero
    wires = [1 << i for i in range(n)] + [0]
    wire_names = list(range(n)) + [n]
    gates = []

    def cost_of_new(x):
        # total hamming distance from each target to nearest available wire after adding x
        c = 0
        allw = wires + [x]
        for t in targets:
            best = min(_hamming(t, w) for w in allw)
            c += best
        return c

    # baseline cost without new wire
    base_cost = sum(min(_hamming(t, w) for w in wires) for t in targets)

    while _remaining() > 2.0:
        best_gain = 0
        best_pair = None
        cur_wires = wires
        # try XORs of pairs of distinct available wires
        L = len(cur_wires)
        for i in range(L):
            for j in range(i + 1, L):
                x = cur_wires[i] ^ cur_wires[j]
                if x == 0:
                    continue
                # gain = base_cost - cost after adding x (approx using current wires)
                c = 0
                for t in targets:
                    d = min(_hamming(t, w) for w in cur_wires)
                    d2 = min(d, _hamming(t, x))
                    c += d2
                gain = base_cost - c
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j, x)
        if best_pair is None or best_gain <= 0:
            break
        i, j, x = best_pair
        new_idx = n + 1 + len(gates)
        gates.append([wire_names[i], wire_names[j]])
        wires.append(x)
        wire_names.append(new_idx)
        base_cost = sum(min(_hamming(t, w) for w in wires) for t in targets)

    # now build outputs from available wires (greedy set-cover-ish: pick wire closest to each target,
    # but we must represent target exactly -> use inputs directly as fallback)
    return gates, wire_names


def _construct_by_matching(matrix, n, max_iters, deadline_fn):
    """Iterative min-weight perfect matching to choose which pairs to XOR.
    Left nodes: remaining target masks. Right nodes: available wire masks.
    Edge weight = cost of XORing a target with a wire (hamming) minus benefit;
    we instead pick pairs of wires whose XOR is most useful. We use matching over
    (target, wire) to decide which target to satisfy next by XORing a wire with a
    new intermediate. Simplified: repeatedly match targets to wires minimizing
    hamming, then create the XOR that closes the gap."""
    m = matrix.shape[0]
    targets = [_bit_vector_to_int(matrix[r]) for r in range(m)]
    wires = [1 << i for i in range(n)] + [0]
    wire_names = list(range(n)) + [n]
    gates = []
    remaining = list(range(m))

    for it in range(max_iters):
        if _remaining() < 2.0:
            break
        if not remaining:
            break
        # build cost matrix |remaining| x |wires|
        R = len(remaining)
        W = len(wires)
        if R == 0 or W == 0:
            break
        C = np.zeros((R, W), dtype=np.float64)
        for ri, ti in enumerate(remaining):
            t = targets[ti]
            for wi, w in enumerate(wires):
                C[ri, wi] = _hamming(t, w)
        # min weight matching (LAPJV). Pad to square if needed.
        try:
            from scipy.sparse.csgraph import minimum_weight_full_bipartite_matching
            # minimum_weight_full_bipartite_matching expects square? It handles rectangular
            # returning dict {row: col}
            match = minimum_weight_full_bipartite_matching(C)
        except Exception:
            break
        # For each matched target, if best wire not exact, create XOR of that wire with
        # the closest existing wire to the residual.
        new_wires = []
        new_names = []
        progressed = False
        for ri, ti in enumerate(remaining):
            t = targets[ti]
            if ri in match:
                wi = match[ri]
                if wi < W and _hamming(t, wires[wi]) == 0:
                    # exact: this target can be output as wire wi
                    continue
            # pick a wire to XOR with to reduce distance
            best_wi = 0
            best_d = 10**9
            for wi, w in enumerate(wires):
                d = _hamming(t, w)
                if d < best_d:
                    best_d = d
                    best_wi = wi
            if best_d == 0:
                continue
            # residual = t ^ wires[best_wi]; try to make residual by XORing two existing wires
            residual = t ^ wires[best_wi]
            found = False
            for a in range(len(wires)):
                for b in range(a + 1, len(wires)):
                    if wires[a] ^ wires[b] == residual:
                        new_idx = n + 1 + len(gates)
                        gates.append([wire_names[a], wire_names[b]])
                        wires.append(residual)
                        wire_names.append(new_idx)
                        found = True
                        progressed = True
                        break
                if found:
                    break
            if not found:
                # create residual by XOR of best_wi with a wire approximating residual
                # fallback: XOR best_wi with the input that has most overlap with residual
                # pick wire w2 minimizing hamming(residual, w2)
                b2 = 0
                bd2 = 10**9
                for wi2, w2 in enumerate(wires):
                    d = _hamming(residual, w2)
                    if d < bd2:
                        bd2 = d
                        b2 = wi2
                if b2 != best_wi:
                    new_idx = n + 1 + len(gates)
                    gates.append([wire_names[best_wi], wire_names[b2]])
                    wires.append(wires[best_wi] ^ wires[b2])
                    wire_names.append(new_idx)
                    progressed = True

        if not progressed:
            break
        # recompute remaining (targets not yet exactly representable)
        remaining = []
        for ti in range(m):
            t = targets[ti]
            ok = False
            for w in wires:
                if w == t:
                    ok = True
                    break
            if not ok:
                remaining.append(ti)

    return gates, wire_names


def _build_outputs(matrix, gates, wire_names, n):
    """Given available wires (inputs+zero+gates), assign each target row to a wire that
    exactly equals it; if none, build it from inputs (guaranteed). Returns outputs list."""
    m = matrix.shape[0]
    wire_set = {}
    for idx, name in enumerate(wire_names):
        pass
    # recompute wire values
    val = [0] * (n + 1 + len(gates))
    for i in range(n):
        val[i] = 1 << i
    val[n] = 0
    for k, (a, b) in enumerate(gates):
        val[n + 1 + k] = val[a] ^ val[b]
    val_by_mask = {}
    for idx, v in enumerate(val):
        if v not in val_by_mask:
            val_by_mask[v] = idx
    outputs = []
    for r in range(m):
        t = _bit_vector_to_int(matrix[r])
        if t in val_by_mask:
            outputs.append(val_by_mask[t])
        else:
            # build from inputs: XOR of set bits
            # represent as a chain; but we can just use inputs if single bit
            bits = [i for i in range(n) if (t >> i) & 1]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                # we must create it: add gates on the fly
                cur = bits[0]
                for b in bits[1:]:
                    new_idx = n + 1 + len(gates)
                    gates.append([cur, b])
                    cur = new_idx
                outputs.append(cur)
    return outputs


def _local_search(gates, outputs, matrix, n, iters, deadline_fn):
    """Try to reduce gate count by dropping redundant gates and swapping.
    Returns (gates, outputs) with same function and lower/equal gate count."""
    m = matrix.shape[0]

    def eval_count(g):
        return len(g)

    def try_drop(g, outs):
        # try removing each gate; if still valid, keep removed
        for idx in range(len(g) - 1, -1, -1):
            if _remaining() < 1.0:
                break
            cand = g[:idx] + g[idx + 1:]
            # remap outputs: any output referring to wire n+1+idx or beyond shifts
            new_outs = []
            ok = True
            for o in outs:
                if o == n + 1 + idx:
                    ok = False
                    break
                if o > n + 1 + idx:
                    new_outs.append(o - 1)
                else:
                    new_outs.append(o)
            if not ok:
                continue
            # also gates referencing removed wire must be remapped
            new_g = []
            ok2 = True
            for (a, b) in cand:
                na = a - 1 if a > n + 1 + idx else a
                nb = b - 1 if b > n + 1 + idx else b
                new_g.append([na, nb])
            if _verify(matrix, new_g, new_outs, n):
                g = new_g
                outs = new_outs
        return g, outs

    best_g, best_o = gates, outputs
    for _ in range(iters):
        if _remaining() < 1.0:
            break
        g2, o2 = try_drop(best_g, best_o)
        if len(g2) < len(best_g):
            best_g, best_o = g2, o2
        else:
            break
    return best_g, best_o


def synthesize(matrix, depth_limit):
    global _DEADLINE
    _DEADLINE = _now() + _TIME_LIMIT
    matrix = np.asarray(matrix, dtype=np.int64)
    m, n = matrix.shape

    best = None  # (gate_count, gates, outputs)

    def consider(gates, outputs):
        nonlocal best
        if not _verify(matrix, gates, outputs, n):
            return
        if _depth_ok(gates, outputs, n) > depth_limit:
            return
        gc = len(gates)
        if best is None or gc < best[0]:
            best = (gc, gates, outputs)

    # Heuristic 1: greedy
    try:
        g1, names1 = _greedy_construct(matrix, n, _remaining)
        o1 = _build_outputs(matrix, g1, names1, n)
        g1, o1 = _local_search(g1, o1, matrix, n, 3, _remaining)
        consider(g1, o1)
    except Exception:
        pass

    # Heuristic 2: matching-based
    try:
        g2, names2 = _construct_by_matching(matrix, n, max_iters=40, deadline_fn=_remaining)
        o2 = _build_outputs(matrix, g2, names2, n)
        g2, o2 = _local_search(g2, o2, matrix, n, 3, _remaining)
        consider(g2, o2)
    except Exception:
        pass

    # Fallback: pure input construction (always valid)
    if best is None or _remaining() < 1.0:
        gates = []
        outputs = []
        for r in range(m):
            t = _bit_vector_to_int(matrix[r])
            bits = [i for i in range(n) if (t >> i) & 1]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                cur = bits[0]
                for b in bits[1:]:
                    new_idx = n + 1 + len(gates)
                    gates.append([cur, b])
                    cur = new_idx
                outputs.append(cur)
        if _verify(matrix, gates, outputs, n) and _depth_ok(gates, outputs, n) <= depth_limit:
            consider(gates, outputs)

    if best is None:
        # last resort: return empty (should not happen)
        return {'gates': [], 'outputs': [n] * m}

    return {'gates': best[1], 'outputs': best[2]}

# EVOLVE-BLOCK-END
