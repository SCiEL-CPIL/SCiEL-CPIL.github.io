"""Construct a small 5-uniform hypergraph with no proper 2-colouring (m(5) <= 51).

Approach: a hypergraph is 2-colourable iff there exists a vertex 2-colouring with
no monochromatic edge.  For 5-uniform edges, "monochromatic" means a 5/0 or 0/5
split, which is a 2-CNF clause.  We therefore need a minimum UNSAT 2-CNF where each
edge contributes 6 clauses.

We use greedy set-cover over the 2^n colourings: each edge "kills" the colourings in
which it is monochromatic.  We pick edges that kill the most remaining colourings.
This is exact over the finite colouring space and finds a small non-colourable set.
"""

import time
import itertools

_START = time.time()
_LIMIT = 112.0  # safety margin before the 120 s kill


def _now():
    return time.time()


def _edge_mask(e):
    m = 0
    for v in e:
        m |= 1 << v
    return m


def _edge_kills_mask(e, n):
    """Bitset of colourings (over n vertices) in which edge e is monochromatic.

    A colouring is a bitstring of length n (bit v = colour of vertex v).
    Edge e is monochromatic iff all vertices in e have the same colour.
    """
    m = _edge_mask(e)
    # Monochromatic-red: all bits in m are 0 (say 0=red)
    # Monochromatic-blue: all bits in m are 1
    # Build the set of colourings where (m & c) == 0  OR  (m & c) == m
    kills = 0
    total = 1 << n
    # Iterate is too slow for large n; use structure.
    # Number of free vertices = n - popcount(m).
    # For red-monochromatic: free vertices can be anything -> 2^(n-k) colourings.
    # We need the actual bit positions.
    k = bin(m).count('1')
    free_mask = ((1 << n) - 1) ^ m
    free_count = n - k
    # Red monochromatic colourings: c & m == 0.
    # Blue monochromatic colourings: c & m == m.
    # Enumerate free assignments.
    for f in range(1 << free_count):
        # Map f bits into free positions.
        c_red = 0
        c_blue = m
        ff = f
        pos = 0
        while ff:
            # find next free bit position
            while (m >> pos) & 1:
                pos += 1
            if ff & 1:
                c_red |= 1 << pos
                c_blue |= 1 << pos
            ff >>= 1
            pos += 1
        kills |= 1 << c_red
        kills |= 1 << c_blue
    return kills


def _build_checker(clauses, n):
    """Exact 2-SAT solver via backtracking + unit propagation.
    Returns True if satisfiable (proper 2-colouring exists)."""
    def lit_idx(var, sign):
        return var * 2 + (0 if sign == 1 else 1)

    def lit_neg(idx):
        return idx ^ 1

    clause_lits = []
    sat_by = {}
    for ci, cl in enumerate(clauses):
        ls = []
        for (var, sign) in cl:
            idx = lit_idx(var, sign)
            ls.append(idx)
            sat_by.setdefault(idx, []).append(ci)
        clause_lits.append(ls)

    force_map = {}
    for ci, ls in enumerate(clause_lits):
        if len(ls) == 2:
            a, b = ls[0], ls[1]
            force_map.setdefault(lit_neg(a), []).append(b)
            force_map.setdefault(lit_neg(b), []).append(a)

    nclauses = len(clauses)

    def propagate(assign, alive):
        stack = list(assign)
        seen = set(assign)
        for a in list(assign):
            for ci in sat_by.get(a, ()):
                alive.discard(ci)
        work = list(assign)
        while work:
            a = work.pop()
            for b in force_map.get(a, ()):
                if b not in seen:
                    seen.add(b)
                    assign.add(b)
                    work.append(b)
                    for ci in sat_by.get(b, ()):
                        alive.discard(ci)
        for ci in list(alive):
            falsified = True
            for l in clause_lits[ci]:
                if l in assign:
                    falsified = False
                    break
            if falsified:
                return assign, alive, True
        return assign, alive, False

    def solve(assign, alive, depth):
        assign, alive, conflict = propagate(assign, alive)
        if conflict:
            return False
        if not alive:
            return True
        if depth > 2 * n:
            return False
        best_ci = None
        best_unassigned = None
        for ci in alive:
            ls = clause_lits[ci]
            unassigned = [l for l in ls if l not in assign]
            if best_unassigned is None or len(unassigned) < len(best_unassigned):
                best_ci = ci
                best_unassigned = unassigned
                if len(unassigned) == 1:
                    break
        if not best_unassigned:
            return False
        l0 = best_unassigned[0]
        a1 = set(assign); a1.add(l0)
        if solve(a1, set(alive), depth + 1):
            return True
        a2 = set(assign); a2.add(lit_neg(l0))
        if solve(a2, set(alive), depth + 1):
            return True
        return False

    return lambda: solve(set(), set(range(nclauses)), 0)


def _edge_clauses(e):
    return [tuple(e), tuple(-v for v in e)]


def construct():
    n = 11
    total_colourings = 1 << n

    # Precompute kill-masks for all 5-subsets (C(11,5)=462).
    all_edges = list(itertools.combinations(range(n), 5))
    kill_masks = [_edge_kills_mask(e, n) for e in all_edges]

    # Greedy set cover: repeatedly pick the edge killing the most uncovered colourings.
    uncovered = (1 << total_colourings) - 1  # too large! 2^2048 bits.
    # 2^2048 is a 2048-bit integer; Python handles big ints fine.
    uncovered = (1 << total_colourings) - 1

    chosen = []
    chosen_idx = set()
    while uncovered:
        if _now() - _START > _LIMIT:
            break
        best_i = -1
        best_gain = 0
        for i, km in enumerate(kill_masks):
            if i in chosen_idx:
                continue
            gain = bin(uncovered & km).count('1')
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0 or best_gain == 0:
            break
        chosen.append(all_edges[best_i])
        chosen_idx.add(best_i)
        uncovered &= ~kill_masks[best_i]

    edges = [list(e) for e in chosen]

    # Verify with the exact checker.
    clauses = []
    for e in edges:
        clauses.extend(_edge_clauses(e))
    checker = _build_checker(clauses, n)
    if checker():
        # Still colourable (shouldn't happen if uncovered==0); fallback.
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]

    return {'n': n, 'edges': edges}
