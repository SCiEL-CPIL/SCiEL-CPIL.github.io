"""XOR-circuit synthesis: min-weight matching + multi-heuristic + gate-drop local search."""
import time
import heapq
import itertools
import numpy as np

_DEADLINE = 85.0  # safety margin under 90s wall-clock
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _popcount(v):
    return v.bit_count()

def _xor_cost(a, b):
    return _popcount(a ^ b)

def _popcount_np(v):
    return int(np.count_nonzero(v))

def _popcount_np_vec(v):
    return np.count_nonzero(v)


def _build_circuit(wires, gates, n, outputs):
    """wires: list of ints (index i -> value), gates: list of (i,j), outputs: list of wire indices."""
    return {'gates': [[g[0], g[1]] for g in gates], 'outputs': list(outputs)}


def _verify(matrix, n, gates, outputs, depth_limit):
    """Return (valid, max_depth)."""
    m = len(matrix)
    vals = [0] * (n + 1 + len(gates))
    for i in range(n):
        vals[i] = matrix[i]
    for k, (a, b) in enumerate(gates):
        vals[n + 1 + k] = vals[a] ^ vals[b]
    for j, w in enumerate(outputs):
        if vals[w] != matrix[n + j]:
            return False, 0
    # depth
    depth = [0] * (n + 1 + len(gates))
    for k in range(len(gates)):
        a, b = gates[k]
        depth[n + 1 + k] = 1 + max(depth[a], depth[b])
    md = max(depth[w] for w in outputs)
    return (md <= depth_limit), md


def _baseline_gate_count(matrix):
    n = len(matrix)
    m = n + 1  # outputs count
    total = 0
    for j in range(n, m):
        total += _popcount(matrix[j])
    return total


def _heuristic_greedy(matrix, depth_limit, seed):
    """Greedy common-subexpression construction. Returns (gates, outputs)."""
    n = len(matrix)
    m = n + 1
    rng = np.random.default_rng(seed)
    wires = [matrix[i] for i in range(n)]
    wires.append(0)
    gates = []
    targets = [matrix[j] for j in range(n, m)]

    # priority: how many targets share a pair of existing wires
    def pair_score(a, b):
        v = wires[a] ^ wires[b]
        if v == 0:
            return -1
        # count how many targets equal v or contain v
        cnt = 0
        for t in targets:
            if t == v:
                cnt += 3
            elif (t & v) == v and t != 0:
                cnt += 1
        return cnt

    best = None
    best_score = 0
    # build candidate pairs
    n_wires = len(wires)
    for a in range(n_wires):
        for b in range(a + 1, n_wires):
            s = pair_score(a, b)
            if s > best_score:
                best_score = s
                best = (a, b)
    # iterate
    iters = 0
    while best is not None and best_score > 0 and iters < 2000:
        a, b = best
        v = wires[a] ^ wires[b]
        new_idx = len(wires)
        wires.append(v)
        gates.append((a, b))
        iters += 1
        # recompute best (cheap for small n)
        n_wires = len(wires)
        best = None
        best_score = 0
        for a2 in range(n_wires):
            for b2 in range(a2 + 1, n_wires):
                s = pair_score(a2, b2)
                if s > best_score:
                    best_score = s
                    best = (a2, b2)
    # now assign outputs: for each target find a wire equal to it, else build from inputs
    outputs = []
    for j in range(n, m):
        t = matrix[j]
        found = None
        for idx, v in enumerate(wires):
            if v == t:
                found = idx
                break
        if found is None:
            # build t as XOR of input wires (popcount gates)
            # start from 0 wire
            cur = n  # zero wire
            bits = [i for i in range(n) if (t >> i) & 1]
            for i in bits:
                new_idx = len(wires)
                wires.append(wires[cur] ^ wires[i])
                gates.append((cur, i))
                cur = new_idx
            found = cur
        outputs.append(found)
    return gates, outputs


def _heuristic_matching(matrix, depth_limit, seed):
    """Iterative min-weight bipartite matching to pick globally-shared XORs."""
    n = len(matrix)
    m = n + 1
    targets = [matrix[j] for j in range(n, m)]
    wires = [matrix[i] for i in range(n)]
    wires.append(0)
    gates = []
    wire_index = {v: i for i, v in enumerate(wires)}

    def try_match():
        # left = targets, right = existing wires (as pairs we can form)
        # We want to pick a new wire v = a^b that is 'useful' (matches many targets)
        # Approximate min-weight matching: score a candidate (a,b) by number of targets it satisfies
        best = None
        best_score = -1
        n_w = len(wires)
        for a in range(n_w):
            for b in range(a + 1, n_w):
                v = wires[a] ^ wires[b]
                if v == 0:
                    continue
                sc = 0
                for t in targets:
                    if t == v:
                        sc += 5
                    elif (t & v) == v:
                        sc += 1
                if sc > best_score:
                    best_score = sc
                    best = (a, b, v)
        return best, best_score

    iters = 0
    while iters < 3000:
        best, sc = try_match()
        if best is None or sc <= 0:
            break
        a, b, v = best
        new_idx = len(wires)
        wires.append(v)
        gates.append((a, b))
        wire_index[v] = new_idx
        iters += 1
    outputs = []
    for j in range(n, m):
        t = matrix[j]
        if t in wire_index:
            outputs.append(wire_index[t])
        else:
            cur = n
            bits = [i for i in range(n) if (t >> i) & 1]
            for i in bits:
                new_idx = len(wires)
                wires.append(wires[cur] ^ wires[i])
                gates.append((cur, i))
                wire_index[wires[new_idx]] = new_idx
                cur = new_idx
            outputs.append(cur)
    return gates, outputs


def _gate_drop_local_search(matrix, depth_limit, gates, outputs):
    """Try to drop redundant gates. Returns (gates, outputs) possibly improved."""
    n = len(matrix)
    m = n + 1
    def eval_count(g):
        return len(g)
    best_gates = gates
    best_outputs = outputs
    improved = True
    while improved:
        improved = False
        for k in range(len(gates)):
            # try removing gate k
            new_gates = gates[:k] + gates[k+1:]
            # recompute which wires map where: gate k produced wire n+1+k
            # After removal, wires after k shift down by 1
            def remap(w):
                if w <= n:
                    return w
                # w is a gate output: original gate index = w - (n+1)
                gidx = w - (n + 1)
                if gidx == k:
                    return None
                if gidx > k:
                    return n + 1 + (gidx - 1)
                return w
            # check all gates and outputs still reference valid wires
            ok = True
            new_outputs = []
            for w in outputs:
                r = remap(w)
                if r is None:
                    ok = False
                    break
                new_outputs.append(r)
            if not ok:
                continue
            new_gates2 = []
            for (a, b) in new_gates:
                ra = remap(a)
                rb = remap(b)
                if ra is None or rb is None:
                    ok = False
                    break
                new_gates2.append((ra, rb))
            if not ok:
                continue
            valid, _ = _verify(matrix, n, new_gates2, new_outputs, depth_limit)
            if valid and len(new_gates2) < len(best_gates):
                best_gates = new_gates2
                best_outputs = new_outputs
                gates = new_gates2
                outputs = new_outputs
                improved = True
                break
    return best_gates, best_outputs


def synthesize(matrix, depth_limit):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    matrix = [int(x) for x in matrix]
    n = len(matrix)
    m = n + 1
    baseline = _baseline_gate_count(matrix)

    candidates = []

    # Heuristic 1: greedy
    if _remaining() > 5:
        g, o = _heuristic_greedy(matrix, depth_limit, seed=1)
        if _verify(matrix, n, g, o, depth_limit)[0]:
            candidates.append((g, o))

    # Heuristic 2: matching
    if _remaining() > 5:
        g, o = _heuristic_matching(matrix, depth_limit, seed=2)
        if _verify(matrix, n, g, o, depth_limit)[0]:
            candidates.append((g, o))

    # Local search on each candidate
    final = None
    for g, o in candidates:
        if _remaining() < 3:
            break
        g2, o2 = _gate_drop_local_search(matrix, depth_limit, g, o)
        if _verify(matrix, n, g2, o2, depth_limit)[0]:
            if final is None or len(g2) < len(final[0]):
                final = (g2, o2)

    if final is None:
        # fallback: trivial direct construction
        gates = []
        wires = matrix + [0]
        outputs = []
        for j in range(n, m):
            t = matrix[j]
            cur = n
            bits = [i for i in range(n) if (t >> i) & 1]
            for i in bits:
                ni = len(wires)
                wires.append(wires[cur] ^ wires[i])
                gates.append((cur, i))
                cur = ni
            outputs.append(cur)
        final = (gates, outputs)

    gates, outputs = final
    return _build_circuit(wires, gates, n, outputs)
