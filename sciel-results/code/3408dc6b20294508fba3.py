"""Combined program: exact DP triangle-free table + targeted local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 18
FULL = (1 << N) - 1


def _precompute_indep(adj, n=N):
    """indep[mask] = True iff the vertex set mask is an independent set."""
    size = 1 << n
    indep = [False] * size
    indep[0] = True
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        indep[mask] = indep[prev] and (adj[v] & prev) == 0
    return indep


def _precompute_tri_free(adj, n=N):
    """tri_free[mask] = True iff the induced subgraph on mask is triangle-free.

    DP: mask is triangle-free iff prev is triangle-free AND the neighbours of v
    inside prev form an independent set (otherwise v closes a triangle)."""
    size = 1 << n
    indep = _precompute_indep(adj, n)
    tri = [False] * size
    tri[0] = True
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        if not tri[prev]:
            tri[mask] = False
        else:
            tri[mask] = indep[adj[v] & prev]
    return tri


def _max_tf_size(tri, n=N):
    """Largest triangle-free induced subgraph size (exact)."""
    best = 0
    for mask in range(1 << n):
        if tri[mask]:
            c = mask.bit_count()
            if c > best:
                best = c
    return best


def _tf_12_masks(tri, n=N):
    """All 12-vertex masks that are triangle-free."""
    return [mask for mask in range(1 << n)
            if tri[mask] and mask.bit_count() == 12]


def _can_add_edge(adj, u, v):
    """Can (u,v) be added without creating a K4?"""
    common = adj[u] & adj[v]
    cm = common
    while cm:
        lsb = cm & (-cm)
        w = lsb.bit_length() - 1
        cm ^= lsb
        if adj[w] & common:
            return False
    return True


def _edges_to_adj(edges, n=N):
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _is_k4_free_adj(adj, n=N):
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au >> v) & 1:
                continue
            common = au & adj[v]
            cm = common
            while cm:
                lsb = cm & (-cm)
                w = lsb.bit_length() - 1
                cm ^= lsb
                if adj[w] & (common & ~(1 << w)):
                    return False
    return True


def _gen_max_k4_free(rng, n=N):
    """Random maximal K4-free graph (greedy over shuffled edge order)."""
    adj = [0] * n
    order = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(order)
    for (u, v) in order:
        if _can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj


def _local_search(adj, tri, deadline, rng, max_iter=400):
    """Targeted local search: repeatedly break triangle-free 12-subsets by
    adding edges inside them (if K4-free), else remove an edge to enable one."""
    adj = list(adj)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        tf = _tf_12_masks(tri, N)
        if not tf:
            break
        mask = rng.choice(tf)
        # candidate non-edges inside the triangle-free subset
        cands = []
        for u in range(N):
            if not (mask >> u) & 1:
                continue
            for v in range(u + 1, N):
                if (mask >> v) & 1 and not (adj[u] >> v) & 1:
                    cands.append((u, v))
        if not cands:
            break
        rng.shuffle(cands)
        added = False
        for (u, v) in cands:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                tri = _precompute_tri_free(adj, N)
                added = True
                break
        if not added:
            # no addable edge here; try to remove an edge incident to the subset
            # to open up an addable edge next round.
            rem_cands = []
            for u in range(N):
                if not (mask >> u) & 1:
                    continue
                for v in range(u + 1, N):
                    if (adj[u] >> v) & 1:
                        rem_cands.append((u, v))
            if not rem_cands:
                break
            (u, v) = rng.choice(rem_cands)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            tri = _precompute_tri_free(adj, N)
    return adj, tri


def construct():
    start = time.time()
    deadline = start + 145.0
    rng = random.Random(12345)

    # Seed: 3 disjoint K6's -> triangle-free subgraph of size 12
    seed_adj = [0] * N
    for base in (0, 6, 12):
        for i in range(base, base + 6):
            for j in range(i + 1, base + 6):
                seed_adj[i] |= (1 << j)
                seed_adj[j] |= (1 << i)

    seed_tri = _precompute_tri_free(seed_adj, N)
    best_adj = list(seed_adj)
    best_tri = seed_tri
    best_score = _max_tf_size(seed_tri, N)

    def consider(adj):
        nonlocal best_adj, best_tri, best_score
        if not _is_k4_free_adj(adj, N):
            return
        tri = _precompute_tri_free(adj, N)
        sc = _max_tf_size(tri, N)
        if sc < best_score:
            best_score = sc
            best_adj = list(adj)
            best_tri = tri

    # Phase 1: random maximal K4-free graphs, exact-evaluated
    while time.time() < deadline - 20.0:
        adj = _gen_max_k4_free(rng, N)
        consider(adj)
        if best_score <= 11:
            break

    # Phase 2: targeted local search from best graph and from fresh starts
    while time.time() < deadline - 5.0:
        if best_score <= 11:
            break
        if time.time() < deadline - 15.0:
            adj = _gen_max_k4_free(rng, N)
        else:
            adj = list(best_adj)
        adj_ls, tri_ls = _local_search(adj, _precompute_tri_free(adj, N),
                                       deadline - 5.0, rng, max_iter=200)
        consider(adj_ls)
        if best_score <= 11:
            break

    # Final: return best found
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] >> v & 1:
                edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
