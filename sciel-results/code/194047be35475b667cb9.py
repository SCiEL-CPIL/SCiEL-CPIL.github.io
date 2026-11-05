"""C5-free graph without independent 9-set (Ramsey R(C5,K9)).
Maximize n. Base construction: 8 disjoint K4s (n=32, alpha=8, C5-free).
Search family: add C5-safe inter-K4 edges and/or a 33rd vertex, verify exactly."""
import time
import random

# EVOLVE-BLOCK-START

def _build_8k4():
    """8 disjoint K4s -> (n, adj_mask_list, edge_list)."""
    blocks, size = 8, 4
    n = blocks * size
    adj = [0] * n
    edges = []
    for b in range(blocks):
        base = b * size
        for i in range(size):
            for j in range(i + 1, size):
                u, v = base + i, base + j
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append([u, v])
    return n, adj, edges


def _has_c5(adj, n):
    """Exact C5 subgraph test via bitset BFS (paths of length 4 closing back)."""
    for v in range(n):
        nv = adj[v]
        for a in range(n):
            if not (nv >> a) & 1 or a == v:
                continue
            for b in range(n):
                if b == v or b == a:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(n):
                    if c == v or c == a or c == b:
                        continue
                    if not (adj[b] >> c) & 1:
                        continue
                    # need d in N(c) with d in N(v), d distinct from v,a,b,c
                    cand = (adj[c] & adj[v]) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    if cand:
                        return True
    return False


def _max_is(adj, n):
    """Maximum independent set size (clique in complement) via bitset Bron-Kerbosch."""
    comp = [0] * n
    for u in range(n):
        m = 0
        for v in range(n):
            if u != v and not (adj[u] >> v) & 1:
                m |= 1 << v
        comp[u] = m
    best = [0]

    def bronk(rc, P, X):
        if P == 0 and X == 0:
            if rc > best[0]:
                best[0] = rc
            return
        if P == 0:
            return
        union = P | X
        bp, bc = -1, -1
        t = union
        while t:
            lb = t & (-t)
            u = lb.bit_length() - 1
            c = (comp[u] & P).bit_count()
            if c > bc:
                bc, bp = c, u
            t ^= lb
        cand = P & ~(comp[bp])
        t = cand
        while t:
            lb = t & (-t)
            w = lb.bit_length() - 1
            bronk(rc + 1, P & comp[w], X & comp[w])
            P &= ~lb
            X |= lb
            t ^= lb

    bronk(0, (1 << n) - 1, 0)
    return best[0]


def construct():
    deadline = time.time() + 110.0  # safety margin under 150s wall-clock

    n, adj, edges = _build_8k4()
    best = {'n': n, 'edges': edges}

    # Verify base result (should be C5-free, alpha=8).
    if _has_c5(adj, n):
        raise RuntimeError("base graph unexpectedly contains a C5")
    if _max_is(adj, n) >= 9:
        raise RuntimeError("base graph has independent set of size >= 9")

    # ---- Search for n=33: add a 33rd vertex with C5-safe neighbor set ----
    # In 8 disjoint K4s, a new vertex may connect to <=1 vertex per K4
    # (2 in same K4 => 5-cycle). With <=1 per K4, alpha(v)=9 always holds,
    # so a bare 33rd vertex is impossible; but we also add inter-K4 edges
    # (C5-safe) that can only shrink the independent set.
    if time.time() < deadline - 25.0:
        N = n
        adj33 = [0] * (N + 1)
        for u in range(N):
            adj33[u] = adj[u]
        # inter-K4 edges: for each pair of K4s, one cross edge (C5-safe)
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                u, v = 4 * b1, 4 * b2
                adj33[u] |= 1 << v
                adj33[v] |= 1 << u
                edges.append([u, v])
        # try to add vertex 32 with a C5-safe neighbor set
        for trial in range(200):
            if time.time() > deadline - 20.0:
                break
            if trial == 0:
                sel = [4 * b for b in range(8)]
            else:
                sel = [4 * b + random.randrange(4) for b in range(8)]
            a33 = adj33[:]
            for s in sel:
                a33[32] |= 1 << s
                a33[s] |= 1 << 32
            if _has_c5(a33, N + 1):
                continue
            if _max_is(a33, N + 1) < 9:
                e33 = edges[:]
                for s in sel:
                    e33.append([s, 32])
                best = {'n': N + 1, 'edges': e33}
                return best
            # also keep the inter-edge graph as a candidate if it stays valid
        # if no 33rd vertex worked, check whether the inter-edge graph itself
        # is still C5-free and alpha<9 (it is n=32, not an improvement, skip)

    return best

# EVOLVE-BLOCK-END
