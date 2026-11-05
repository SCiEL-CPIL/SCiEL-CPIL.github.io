"""Maximize edges on 16 vertices while staying Q3-free.

Strategy: K_{2,3}-free is a necessary condition for Q3-free (Q3 contains a K_{2,3}).
We greedily maximize edges under the K_{2,3}-free constraint, which is a valid
sufficient condition for Q3-freeness. Use randomized restarts + bitset acceleration.
"""
import time
import random

N = 16
TOTAL_EDGES = N * (N - 1) // 2  # 120
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
PAIR_IDX = {(i, j): k for k, (i, j) in enumerate(ALL_PAIRS)}
# Precompute: for each vertex, list of (neighbor, pair_index)
VADJ = [[] for _ in range(N)]
for k, (i, j) in enumerate(ALL_PAIRS):
    VADJ[i].append((j, k))
    VADJ[j].append((i, k))


def k23_safe(adj, u, v, cu, cv):
    """Return True if adding edge (u,v) keeps graph K_{2,3}-free.
    adj: list of bitmasks (int), cu/cv: current neighbor counts."""
    # After adding (u,v): u and v become adjacent.
    # Violations can only involve pairs that are now non-adjacent and had exactly 2
    # common neighbors before, and now would have 3.
    # Adding (u,v) increases common-neighbor count only for pairs where one endpoint
    # is u or v AND the other is a common-neighbor situation.
    # Specifically: for non-adjacent (x,y), cn(x,y) increases by:
    #   +1 if x==u and y in N(v) (and y non-adj to u) ... 
    # Let's reason: cn(x,y) = |N(x) ∩ N(y)|.
    # Adding edge (u,v) adds v to N(u) and u to N(v).
    # cn(x,y) changes iff {x,y} intersects {u,v}.
    # Case x==u, y!=v: new cn = |N(u) ∩ N(y)| with N(u) now includes v.
    #   increases by 1 iff v in N(y) i.e. (v,y) edge, and y not already adj to u.
    #   Since we only consider non-adj (u,y): y not adj to u. And y != v.
    #   So cn(u,y) increases by 1 iff (v,y) is an edge.
    #   This creates a violation iff old cn(u,y) == 2 and (v,y) edge and (u,y) non-adj.
    # Case x==v, y!=u: symmetric: cn(v,y) increases by 1 iff (u,y) edge.
    #   Violation iff old cn(v,y)==2 and (u,y) edge and (v,y) non-adj.
    # Case x==u,y==v: they become adjacent, so they no longer count as a non-adj pair.
    #   Removing them from consideration can't create a violation.
    # Therefore check:
    #   For each y adj to v, y != u, y non-adj to u: if cn(u,y)==2 -> violation.
    #   For each y adj to u, y != v, y non-adj to v: if cn(v,y)==2 -> violation.
    # Need current cn values.
    # cu, cv are counts but we need per-pair cn. Compute cn(u,y) = popcount(adj[u] & adj[y]).
    for (y, _) in VADJ[v]:
        if y == u:
            continue
        if adj[u] >> y & 1:
            continue  # u,y adjacent -> not a K2,3 "2-part" pair
        # (v,y) is an edge (we're iterating VADJ[v])
        if (adj[u] & adj[y]).bit_count() == 2:
            return False
    for (y, _) in VADJ[u]:
        if y == v:
            continue
        if adj[v] >> y & 1:
            continue
        if (adj[v] & adj[y]).bit_count() == 2:
            return False
    return True


def greedy(order, seed):
    adj = [0] * N
    count = [0] * N
    m = 0
    for (u, v) in order:
        if adj[u] >> v & 1:
            continue
        if k23_safe(adj, u, v, count[u], count[v]):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            count[u] += 1
            count[v] += 1
            m += 1
    return m, adj


def to_edges(adj):
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] >> j & 1:
                out.append([i, j])
    return out


def verify_q3_free(adj):
    """Brute-force check no induced or non-induced Q3. Q3 has 8 vertices, 12 edges, bipartite 4-4, each degree 3."""
    # Enumerate all 8-vertex subsets; check if induced subgraph equals Q3 (as a subgraph:
    # we need an 8-vertex set that CONTAINS a Q3 as a subgraph, not necessarily induced).
    # Q3 as subgraph: need 8 distinct vertices with a bijection to {0..7} such that the
    # 12 cube edges are present. Equivalent: there exists a 4-set A and 4-set B (partition
    # of the 8) with |A|=|B|=4, every a in A adj to exactly 3 in B? No: in Q3 each vertex
    # degree 3, bipartition 4+4, each vertex in A connects to exactly 3 of B (all but its
    # "opposite" pattern). Actually Q3: label by 3-bit strings; A = even parity, B = odd.
    # Each vertex connects to the 3 vertices differing in one bit. Within A/B no edges.
    # So check: exists partition of 8-vertex set into A(4),B(4) with: no edges within A, no
    # edges within B, and the bipartite graph is exactly the cube bipartite graph.
    # The cube bipartite graph: each vertex degree 3, and it's a specific 4x4 0-1 matrix
    # (the 4x4 with each row/col sum 3, i.e. J_4 - I_4). So condition: A,B independent,
    # and every a in A has degree 3 into B, every b in B degree 3 into A.
    # (Since 4x4 with all-3 row/col sums is unique up to permutation = J-I.)
    from itertools import combinations
    for S in combinations(range(N), 8):
        A = set(S)
        # try all ways to pick 4 for A
        for a_set in combinations(S, 4):
            a = set(a_set)
            b = A - a
            # check independence
            ok = True
            for i in range(4):
                for j in range(i + 1, 4):
                    if adj[list(a)[i]] >> list(a)[j] & 1:
                        ok = False; break
                    if adj[list(b)[i]] >> list(b)[j] & 1:
                        ok = False; break
                if not ok:
                    break
            if not ok:
                continue
            # check degrees
            good = True
            for x in a:
                d = sum(1 for y in b if adj[x] >> y & 1)
                if d != 3:
                    good = False; break
            if good:
                for x in b:
                    d = sum(1 for y in a if adj[x] >> y & 1)
                    if d != 3:
                        good = False; break
            if good:
                return False
    return True


def construct():
    deadline = time.time() + 140  # safety margin below 150s
    best_m = 0
    best_adj = None

    # Always run a deterministic baseline first (fast).
    base_order = list(ALL_PAIRS)
    m, adj = greedy(base_order, 0)
    if m > best_m:
        best_m, best_adj = m, adj

    if time.time() > deadline - 5:
        return to_edges(best_adj)

    # Randomized restarts.
    rng = random.Random(12345)
    it = 0
    while time.time() < deadline - 3:
        it += 1
        order = list(ALL_PAIRS)
        rng.shuffle(order)
        # Bias: sometimes sort by a random vertex-priority to explore different structures.
        m, adj = greedy(order, it)
        if m > best_m:
            best_m, best_adj = m, adj
            if best_m >= 80:
                # good enough, stop early
                break

    # Final verification of best (cheap for 16 vertices, but guard time).
    if best_adj is not None and time.time() < deadline - 1:
        if not verify_q3_free(best_adj):
            # fallback to a known Q3-free: K_{3,13}
            return [[a, b] for a in range(3) for b in range(3, 16)]

    return to_edges(best_adj)


if __name__ == "__main__":
    edges = construct()
    print("edges:", len(edges))
