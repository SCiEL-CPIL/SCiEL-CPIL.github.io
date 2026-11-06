import time
import random
from itertools import combinations

def _precompute(n, edges):
    m = len(edges)
    # For each vertex: bitmask of edges it belongs to
    vedges = [0] * n
    for i, e in enumerate(edges):
        for v in e:
            vedges[v] |= (1 << i)
    # For each edge: bitmask of its 5 vertices
    edge_vmasks = [0] * m
    for i, e in enumerate(edges):
        msk = 0
        for v in e:
            msk |= (1 << v)
        edge_vmasks[i] = msk
    return vedges, edge_vmasks

def is_2_colorable(n, edges):
    """Exact 2-colorability check via DPLL with unit propagation.
    Returns True iff there is a 2-coloring where no edge is monochromatic."""
    m = len(edges)
    if m == 0:
        return True
    vedges, edge_vmasks = _precompute(n, edges)

    def check(white, black):
        if white & black:
            return None
        # unit propagation
        queue = []
        for i in range(m):
            em = edge_vmasks[i]
            bw = em & white
            bb = em & black
            if bw & bb:
                return False
            bw_count = bw.bit_count()
            bb_count = bb.bit_count()
            if bw_count == 5 or bb_count == 5:
                return False
            if bw_count == 4:
                u = (em ^ bw).bit_length() - 1
                if (black >> u) & 1:
                    return False
                queue.append((1 << u, 1))  # force black
            elif bb_count == 4:
                u = (em ^ bb).bit_length() - 1
                if (white >> u) & 1:
                    return False
                queue.append((1 << u, 0))  # force white
        # propagate
        for mask, is_black in queue:
            if is_black:
                if (white & mask):
                    return False
                if (black & mask):
                    continue
                black |= mask
            else:
                if (black & mask):
                    return False
                if (white & mask):
                    continue
                white |= mask
        if white & black:
            return False
        # check no monochromatic edge now
        for i in range(m):
            em = edge_vmasks[i]
            if (em & white) == em or (em & black) == em:
                return False
        return (white, black)

    def dpll(white, black):
        state = check(white, black)
        if state is None:
            return False
        if state is False:
            return False
        white, black = state
        # choose uncolored vertex with most constraints
        uncolored = ~white & ~black
        uncolored &= (1 << n) - 1
        if uncolored == 0:
            return True
        # pick vertex with max degree (most edges) among uncolored
        best_v = -1
        best_deg = -1
        bits = uncolored
        while bits:
            low = bits & (-bits)
            v = low.bit_length() - 1
            deg = vedges[v].bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            bits ^= low
        if best_v == -1:
            return True
        bv = 1 << best_v
        if dpll(white | bv, black):
            return True
        if dpll(white, black | bv):
            return True
        return False

    return dpll(0, 0)

def construct():
    start = time.time()
    deadline = start + 115.0

    def time_left():
        return deadline - time.time()

    best = None  # (n, edges_list)

    def try_record(n, edges):
        nonlocal best
        # verify
        if not is_2_colorable(n, edges):
            if best is None or len(edges) < len(best[1]):
                best = (n, list(edges))
                return True
        return False

    # Phase 1: greedy deletion starting from complete K_n^5 for small n
    for n in [9, 10, 11, 12, 13]:
        if time_left() < 10.0:
            break
        all_edges = [tuple(e) for e in combinations(range(n), 5)]
        current = all_edges[:]
        # ensure non-colorable start
        if is_2_colorable(n, current):
            continue
        # greedy deletion passes
        max_passes = 60
        for p in range(max_passes):
            if time_left() < 5.0:
                break
            random.shuffle(current)
            removed_any = False
            for edge in current[:]:
                if time_left() < 5.0:
                    break
                cand = [e for e in current if e != edge]
                if not is_2_colorable(n, cand):
                    current = cand
                    removed_any = True
                    if best is None or len(current) < len(best[1]):
                        best = (n, list(current))
                    if len(current) <= 51:
                        break
            if len(current) <= 51:
                break
            if not removed_any:
                break
    if best is not None:
        n, edges = best
        return {'n': n, 'edges': [list(e) for e in edges]}

    # Fallback: complete K9^5 (guaranteed non-2-colorable)
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
