"""Greedy biclique partition: repeatedly extract the largest common-neighbor biclique."""
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _as_int(x):
    if isinstance(x, bool):
        raise ValueError("bool not allowed")
    if isinstance(x, int):
        return x
    raise ValueError("not int")

def _trivial_partition(edges):
    # each edge its own biclique [u],[v]
    return [[[int(u)], [int(v)]] for (u, v) in edges]

def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    n = int(P['n'])
    # adjacency sets for the pinned graph
    adj = [set() for _ in range(n)]
    for (a, b) in P['edges']:
        a = int(a); b = int(b)
        adj[a].add(b)
        adj[b].add(a)

    # remaining edges as frozenset pairs
    remaining = set()
    for (a, b) in P['edges']:
        a = int(a); b = int(b)
        remaining.add(frozenset((a, b)))

    bicliques = []

    while remaining:
        if time.time() > DEADLINE:
            # finish trivially with remaining edges to stay valid
            for e in remaining:
                u, v = tuple(e)
                bicliques.append([[int(u)], [int(v)]])
            break

        # For each remaining edge (u,v), common neighbors = adj[u] ∩ adj[v]
        # that form a biclique [u]+L , v+R where L⊆N(u)\{v}, R⊆N(v)\{u},
        # and every (l,r) pair is an edge. We seek max |L|*|R| (+1 for the seed edge).
        best_edges = None
        best_score = 0
        # iterate over remaining edges as the "anchor" pair (u in L-side, v in R-side)
        for e in remaining:
            u, v = tuple(e)
            # candidate L: neighbors of u (excluding v) that are also adjacent to v
            # candidate R: neighbors of v (excluding u) that are also adjacent to u
            # For a valid biclique [L,R], we need: all L pairwise? no - within-side edges allowed.
            # We need: (a) u-v edge (yes, it's in remaining), (b) for all l in L, l adjacent to all r in R,
            #          (c) u adjacent to all r in R, (d) v adjacent to all l in L.
            # So L ⊆ N(v) (excluding u), R ⊆ N(u) (excluding v), and L×R ⊆ edges.
            # Additionally each l in L must be adjacent to v (so (l,v) is an edge) and each r in R adjacent to u.
            L_cand = [w for w in adj[v] if w != u]   # l must be adjacent to v
            R_cand = [w for w in adj[u] if w != v]   # r must be adjacent to u
            # Now need L×R all edges. Find largest L⊆L_cand, R⊆R_cand with L×R⊆E.
            # Greedy: build R then filter L, or use common-neighbor intersection.
            # For a fixed R, valid L = { l in L_cand : for all r in R, r in adj[l] }
            # We want to maximize |L|+|R| (or |L|*|R|). Try: pick R as a set, compute L.
            # Efficient approach: try each r in R_cand as the "extreme" and intersect.
            # Start R = R_cand, then L = intersection of adj[r] for r in R ∩ L_cand.
            # Then shrink R to those r where r adjacent to all l in L, iterate.
            # Simpler: enumerate a few candidate R sizes.
            # Use iterative refinement:
            R = set(R_cand)
            improved = True
            while improved:
                improved = False
                # L = L_cand ∩ (∩_{r in R} adj[r])
                L = set(L_cand)
                for r in R:
                    L &= adj[r]
                    if not L:
                        break
                if not L:
                    # drop r's that cause empty; remove the r with fewest connections
                    # fallback: try removing each r
                    newR = None
                    for r in list(R):
                        R2 = set(R); R2.discard(r)
                        L2 = set(L_cand)
                        ok = True
                        for r2 in R2:
                            L2 &= adj[r2]
                        if len(L2) > 0:
                            newR = R2
                            break
                    if newR is None:
                        R = set()
                        break
                    R = newR
                    improved = True
                    continue
                # Now shrink R to those r adjacent to all l in L
                newR = set()
                for r in R:
                    ok = True
                    for l in L:
                        if l not in adj[r]:
                            ok = False
                            break
                    if ok:
                        newR.add(r)
                if len(newR) != len(R):
                    R = newR
                    improved = True

            L = list(L) if (R and L) else []
            # recompute L given final R
            if R:
                Lset = set(L_cand)
                for r in R:
                    Lset &= adj[r]
                L = list(Lset)
            else:
                L = []
            size = len(L) + len(R) + 1  # +1 for anchor edge u-v
            if size > best_score:
                best_score = size
                best_edges = (u, v, L, R)

        if best_edges is None or best_score < 2:
            # no biclique larger than single edge; take one edge
            e = next(iter(remaining))
            u, v = tuple(e)
            bicliques.append([[int(u)], [int(v)]])
            remaining.discard(e)
            continue

        u, v, L, R = best_edges
        L = list(L); R = list(R)
        left = [int(u)] + [int(x) for x in L]
        right = [int(v)] + [int(x) for x in R]
        bicliques.append([left, right])
        # remove all cross edges from remaining
        for a in left:
            for b in right:
                remaining.discard(frozenset((a, b)))

    # finalize types
    out = []
    for (L, R) in bicliques:
        out.append([[int(x) for x in L], [int(x) for x in R]])
    return out
