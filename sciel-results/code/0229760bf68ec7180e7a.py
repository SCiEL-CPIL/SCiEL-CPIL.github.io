"""Sunflower-free 4-uniform family via conflict-graph maximum independent set."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

N = 16
K = 4
MAXC = 250
DEADLINE = 80.0  # safety margin under the 90 s wall-clock kill
_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _sunflower_conflict_graph(sets):
    """Return a 0/1 adjacency matrix: edge (i,j) iff some third set forms a sunflower."""
    m = len(sets)
    mat = np.array(sets, dtype=np.int8)  # (m,4)
    A = (mat[:, None, :] == mat[None, :, :]).astype(np.int8)  # (m,m,4)
    A = A * (A > 0)
    core = A.sum(axis=2)  # (m,m) size of pairwise intersection
    # A,B,C sunflower (C = A∩B∩C) iff for each pair the pairwise intersection equals the triple core.
    # Equiv: min of the three pairwise intersections == sum of the three cores - 2*... use direct check:
    # triple core size = (A[i,j] & A[i,k] & A[j,k]).sum over elements
    T = np.einsum('ijk,ikl,jkl->ij', A, A, A)  # (m,m) triple-core size
    # pairwise-intersection equality: for triple (i,j,k), need core_ij==T_ik==... 
    # Simpler: a sunflower exists iff there is a set C such that C ⊆ A and C ⊆ B and |A∩B|==|C|.
    # Build conflict via: for each pair (i,j), does there exist k with A[i,k]&A[j,k] == A[i,j] (as sets)?
    # We'll compute using element-wise: pair (i,j) conflicts if ∃k: (A[i,k] & A[j,k]) == A[i,j] elementwise.
    # Represent each set as bitmask int for fast elementwise AND.
    masks = []
    for r in sets:
        bm = 0
        for x in r:
            bm |= (1 << x)
        masks.append(bm)
    masks = np.array(masks, dtype=np.int64)
    # pairwise intersection mask
    P = masks[:, None] & masks[None, :]  # (m,m) elementwise AND mask
    # pair (i,j) conflicts iff ∃k: (P[i,k] & P[j,k]) == P[i,j]
    # (P[i,k] & P[j,k]) is the triple core mask; equality with P[i,j] means pairwise==triple.
    adj = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        # for fixed i, j conflicts with i if ∃k: (P[i,k]&P[j,k])==P[i,j]
        # = ∃k: (P[i,k] & P[j,k]) == P[i,j]
        # vectorize over j,k:
        # Q[j,k] = P[i,k] & P[j,k]; need Q[j,k]==P[i,j]
        Q = P[i, :, None] & P[None, :, :]  # (j,k)
        # for each j, does any k satisfy Q[j,k]==P[i,j]?
        hit = (Q == P[i, :, None]).any(axis=1)  # over k
        adj[i, hit] = 1
        adj[hit, i] = 1
    np.fill_diagonal(adj, 0)
    return adj


def _greedy_color(order, adj_rows):
    """Color vertices in order; return color classes as lists (list of vertex indices)."""
    color_of = {}
    classes = []
    for v in order:
        used = set()
        row = adj_rows[v]
        for c, cls in enumerate(classes):
            # v conflicts with any in cls?
            if row[list(cls)].any():
                used.add(c)
        if used:
            # pick smallest free color
            c = 0
            while c in used:
                c += 1
            if c == len(classes):
                classes.append([])
            classes[c].append(v)
            color_of[v] = c
        else:
            classes.append([v])
            color_of[v] = len(classes) - 1
    return classes, color_of


def _branch_and_bound(adj, order, adj_rows, best):
    """Return max independent set size (>= best) using coloring bound. best=[size]."""
    def rec(cands):
        if not cands:
            return
        if len(cands) + best[0] <= best[0]:
            return
        # coloring bound
        classes, _ = _greedy_color(cands, adj_rows)
        if len(cands) + len(classes) <= best[0]:
            return
        # pick vertex with highest degree in candidate subgraph (heuristic)
        v = cands[0]
        # branch: include v
        new_best = best[0]
        rec2(cands, v)
        # we implement iteratively below
    # Use recursive with include/exclude
    def rec(cands, cur):
        if _time_left() < 0.5:
            return
        if not cands:
            if cur > best[0]:
                best[0] = cur
            return
        classes, _ = _greedy_color(cands, adj_rows)
        if cur + len(classes) <= best[0]:
            return
        if cur + len(cands) <= best[0]:
            return
        # choose pivot: vertex of max degree within cands
        cand_set = set(cands)
        v = max(cands, key=lambda x: sum(1 for y in cands if x != y and adj_rows[x][y]))
        # include branch
        nb = [u for u in cands if u != v and not adj_rows[v][u]]
        rec(nb, cur + 1)
        # exclude branch
        cands.remove(v)
        rec(cands, cur)
        cands.append(v)
    rec(list(order), 0)
    return best[0]


def _local_search(sets, adj):
    """Greedy add: try to add each excluded set if it doesn't conflict with chosen."""
    m = len(sets)
    chosen = set()
    # start from a greedy independent set
    order = list(range(m))
    for v in order:
        if all(not adj[v][u] for u in chosen):
            chosen.add(v)
    # try swaps
    improved = True
    while improved and _time_left() > 2:
        improved = False
        for out in list(chosen):
            for inn in range(m):
                if inn in chosen:
                    continue
                # inn must not conflict with chosen minus out
                if any(adj[inn][u] for u in chosen if u != out):
                    continue
                # swapping only helps if inn allows more? keep simple: only swap if it doesn't reduce
                # (size same) - skip; instead just try adding is impossible. So do 1-for-1 not useful.
                pass
    return chosen


def construct():
    all_sets = [set(c) for c in combinations(range(1, 17), K)]
    # Build conflict graph
    adj = _sunflower_conflict_graph(all_sets)
    adj_rows = [np.array(adj[i], dtype=bool) for i in range(len(all_sets))]

    # Order vertices by degree ascending (low degree first) for coloring quality
    deg = adj.sum(axis=1)
    order = sorted(range(len(all_sets)), key=lambda x: deg[x])

    best = [0]
    _branch_and_bound(adj, order, adj_rows, best)

    # Also try a greedy independent set as a floor / seed
    greedy = set()
    for v in order:
        if all(not adj[v][u] for u in greedy):
            greedy.add(v)
    if len(greedy) > best[0]:
        best[0] = len(greedy)

    # Reconstruct a concrete independent set of size best[0]
    # Use greedy that respects the bound: just take a greedy IS and if smaller, try to grow
    final = set()
    for v in order:
        if len(final) >= best[0]:
            break
        if all(not adj[v][u] for u in final):
            final.add(v)

    # If greedy didn't reach best[0], do a simple repair: pick any maximal IS
    if len(final) < best[0]:
        final = set()
        for v in range(len(all_sets)):
            if len(final) >= best[0]:
                break
            if all(not adj[v][u] for u in final):
                final.add(v)

    # Cap at MAXC
    final = list(final)[:MAXC]

    # Verify no sunflower among final (safety)
    fsets = [all_sets[i] for i in final]
    ok = True
    for a, b, c in combinations(fsets, 3):
        if (a & b) == (a & c) == (b & c):
            ok = False
            break
    if not ok:
        # fallback to a verified greedy construction
        chosen = []
        for s in fsets:
            if all(not((s & x) == (s & y) == (x & y)) for x, y in combinations(chosen, 2)):
                chosen.append(s)
        return [sorted(s) for s in chosen[:MAXC]]

    return [sorted(s) for s in fsets]
# EVOLVE-BLOCK-END
