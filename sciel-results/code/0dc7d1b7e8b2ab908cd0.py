"""Minimize edges of a non-2-colorable 5-uniform hypergraph."""
import time
import random
from itertools import combinations

DEADLINE = 145.0  # safety margin under 150s wall clock
_start = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start)


def _edge_mask(edge):
    m = 0
    for v in edge:
        m |= (1 << v)
    return m


def _is_2_colorable(n, masks):
    """
    Return True if hypergraph is 2-colorable (exists coloring with no mono edge).
    Uses DPLL-style backtracking with unit propagation on the constraint
    that each edge must have at least one 0 and one 1.
    
    We model: for each edge e, clause (OR of vertices in e) AND (OR of NOT vertices in e).
    This is a SAT problem. We use a simple DPLL with propagation.
    """
    num_edges = len(masks)
    if num_edges == 0:
        return True
    
    # Precompute for each edge: the mask
    # State: assignment as two bitmasks (zeros, ones), or use a single int where
    # bit v = 0 means unassigned, but that's tricky. Use two masks.
    
    # For efficiency, we'll use a recursive DPLL with memoization on (zero_mask, one_mask, edge_idx)
    # But that state space is huge. Instead, use a simpler approach:
    # Try to find a coloring by backtracking on vertices.
    
    # Better: use the fact that we just need to check satisfiability of a 2-CNF-like structure.
    # Each edge gives: (v1 OR v2 OR v3 OR v4 OR v5) AND (NOT v1 OR NOT v2 OR NOT v3 OR NOT v4 OR NOT v5)
    # This is not 2-CNF, it's 5-CNF.
    
    # Use backtracking with unit propagation on the 5-clauses.
    # A clause is "unit" if it has exactly one unassigned literal.
    
    memo = {}
    
    def propagate(zmask, omask, i):
        """
        zmask: vertices assigned 0
        omask: vertices assigned 1
        i: current edge index (we process edges in order)
        Returns (zmask, omask, i, ok)
        """
        changed = True
        while changed:
            changed = False
            for j in range(i, num_edges):
                m = masks[j]
                # Count assignments
                z_in = m & zmask
                o_in = m & omask
                unassigned = m & ~(zmask | omask)
                
                # Check if edge is violated (all same color)
                if (z_in == m) or (o_in == m):
                    return (zmask, omask, i, False)
                
                # If edge already has both colors, it's satisfied regardless of unassigned
                if (z_in != 0) and (o_in != 0):
                    continue
                
                # If all but one assigned, and they're all the same, force the last
                if unassigned and (unassigned & (unassigned - 1)) == 0:
                    vbit = unassigned
                    if z_in == (m & ~vbit):
                        # All other in this edge are 0, so this must be 1
                        omask |= vbit
                        changed = True
                    elif o_in == (m & ~vbit):
                        # All other in this edge are 1, so this must be 0
                        zmask |= vbit
                        changed = True
        return (zmask, omask, i, True)
    
    def rec(zmask, omask, i):
        if i >= num_edges:
            return True
        
        key = (zmask, omask, i)
        if key in memo:
            return memo[key]
        
        zmask, omask, i, ok = propagate(zmask, omask, i)
        if not ok:
            memo[key] = False
            return False
        
        if i >= num_edges:
            memo[key] = True
            return True
        
        m = masks[i]
        unassigned = m & ~(zmask | omask)
        
        # If edge already satisfied (has both colors), move on
        if (m & zmask) and (m & omask):
            memo[key] = rec(zmask, omask, i + 1)
            return memo[key]
        
        # If no unassigned vertices, edge is monochromatic -> fail
        if unassigned == 0:
            memo[key] = False
            return False
        
        # Branch on a vertex in this edge
        vbit = unassigned & (-unassigned)
        
        # Try assigning 0 first, then 1
        if rec(zmask | vbit, omask, i):
            memo[key] = True
            return True
        if rec(zmask, omask | vbit, i):
            memo[key] = True
            return True
        
        memo[key] = False
        return False
    
    return rec(0, 0, 0)


def _core_extract(n, edges):
    """Iteratively remove edges that are not needed for non-2-colorability."""
    if _time_left() < 5.0:
        return None
    masks = [_edge_mask(e) for e in edges]
    present = [True] * len(masks)
    improved = True
    while improved:
        if _time_left() < 5.0:
            break
        improved = False
        for idx in range(len(masks)):
            if not present[idx]:
                continue
            if _time_left() < 5.0:
                break
            present[idx] = False
            sub = [masks[j] for j in range(len(masks)) if present[j]]
            if not _is_2_colorable(n, sub):
                improved = True
            else:
                present[idx] = True
    if _time_left() < 3.0:
        return None
    final = [masks[j] for j in range(len(masks)) if present[j]]
    out = []
    for m in final:
        e = []
        for v in range(n):
            if m & (1 << v):
                e.append(v)
        out.append(e)
    return out


def construct():
    best = None
    best_count = 10 ** 9

    def consider(n, edges):
        nonlocal best, best_count
        if edges is None:
            return
        if _time_left() < 3.0:
            return
        masks = [_edge_mask(e) for e in edges]
        if _is_2_colorable(n, masks):
            return
        seen = set()
        uniq = []
        for e in edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                uniq.append(list(t))
        if len(uniq) < best_count and 5 <= n <= 26 and len(uniq) <= 200:
            best = {'n': n, 'edges': uniq}
            best_count = len(uniq)

    # Try complete 5-graph on 9 vertices (126 edges)
    if _time_left() > 10.0:
        edges9 = [list(e) for e in combinations(range(9), 5)]
        masks9 = [_edge_mask(e) for e in edges9]
        if not _is_2_colorable(9, masks9):
            r = _core_extract(9, edges9)
            consider(9, r)

    # Try random seeds on n=10,11,12
    random.seed(42)
    for n in (10, 11, 12, 13):
        if _time_left() < 15.0:
            break
        for attempt in range(50):
            if _time_left() < 10.0:
                break
            k = random.randint(50, min(150, len(list(combinations(range(n), 5)))))
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            edges = [list(e) for e in all_edges[:k]]
            masks = [_edge_mask(e) for e in edges]
            if not _is_2_colorable(n, masks):
                r = _core_extract(n, edges)
                consider(n, r)
                if best_count <= 55:
                    break

    if best is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    return best
