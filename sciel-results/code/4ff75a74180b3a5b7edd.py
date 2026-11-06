import time
import random
from itertools import combinations

# ---------- fast 2-colourability checker (bit-mask backtracking + unit propagation) ----------
# A 5-edge is monochromatic iff it is all-0 or all-1.
# For each edge, the forbidden assignments are the two monochromatic masks.
# We represent a partial colouring by two bitmasks (pos, neg) of assigned vertices.
# Unit propagation: if all but one vertex of an edge are assigned the same colour,
# the last must be the opposite colour; if it is already assigned the same, contradiction.

def is_non_2colorable(edges, n):
    """Return True if the hypergraph has NO proper 2-colouring (i.e. it is non-2-colourable)."""
    # Precompute, for each edge, the vertex masks useful for propagation.
    # edge_mask: 5-bit vertex mask; we also store the set of vertices.
    em = [int(''.join('1' if v in e else '0' for v in range(n)), 2) for e in edges]
    # For quick per-vertex edge incidence (for propagation we scan edges; keep list of (mask, verts))
    ev = [(m, list(e)) for m, e in zip(em, edges)]

    # Recursive search over unassigned vertices.
    # state: pos_mask, neg_mask
    # We'll use a simple recursion with propagation.
    full = (1 << n) - 1

    # Build for each vertex the list of edge indices containing it (for fast propagation).
    inc = [[] for _ in range(n)]
    for i, (m, vs) in enumerate(ev):
        for v in vs:
            inc[v].append(i)

    def check(pos, neg):
        # pos, neg: bitmasks of assigned vertices (disjoint)
        # propagate
        changed = True
        while changed:
            changed = False
            for i, (m, vs) in enumerate(ev):
                # determine colour counts among assigned vertices in this edge
                posc = 0
                negc = 0
                unassigned = []
                for v in vs:
                    if (pos >> v) & 1:
                        posc += 1
                    elif (neg >> v) & 1:
                        negc += 1
                    else:
                        unassigned.append(v)
                if posc >= 5:
                    return False  # contradiction: all positive
                if negc >= 5:
                    return False
                # unit propagation: 4 of one colour force the 5th opposite
                if posc == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if (neg >> v) & 1:
                        return False
                    pos |= (1 << v)
                    changed = True
                elif negc == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if (pos >> v) & 1:
                        return False
                    neg |= (1 << v)
                    changed = True
        # if all assigned, success (a valid 2-colouring exists)
        if (pos | neg) == full:
            return True
        # pick an unassigned vertex (first)
        un = full & ~(pos | neg)
        v = (un & -un).bit_length() - 1
        # try 0 then 1
        if check(pos | (1 << v), neg):
            return True
        if check(pos, neg | (1 << v)):
            return True
        return False

    return not check(0, 0)


def _complete9():
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}


def _greedy_reduce(edges, n, deadline):
    """Try to remove edges while keeping non-2-colourable. Returns best (fewest) edge list."""
    edges = list(edges)
    best = list(edges)
    rng = random.Random(12345)
    while time.time() < deadline:
        # attempt a removal pass
        cur = list(best)
        improved = True
        order = list(range(len(cur)))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            e = cur[idx]
            cand = cur[:idx] + cur[idx+1:]
            if is_non_2colorable(cand, n):
                cur = cand
                improved = True
        if len(cur) < len(best):
            best = cur
        if not improved:
            # try a few random restarts of ordering
            restarts = 0
            while restarts < 5 and time.time() < deadline:
                cur = list(best)
                order = list(range(len(cur)))
                rng.shuffle(order)
                any_removed = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    e = cur[idx]
                    cand = cur[:idx] + cur[idx+1:]
                    if is_non_2colorable(cand, n):
                        cur = cand
                        any_removed = True
                if len(cur) < len(best):
                    best = cur
                    any_removed = True
                if not any_removed:
                    break
                restarts += 1
            if not any_removed:
                break
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s
    base = _complete9()
    n = base['n']
    edges = base['edges']
    # base is known non-2-colourable; verify quickly (guard)
    if not is_non_2colorable(edges, n):
        return base
    reduced = _greedy_reduce(edges, n, deadline)
    if time.time() > deadline - 5.0:
        return {'n': n, 'edges': base['edges']}
    return {'n': n, 'edges': reduced}
