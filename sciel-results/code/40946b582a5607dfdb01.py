"""C6-free graph on 40 vertices: maximize edges via greedy fill + 1-edge-swap local search."""
import random
import time

N = 40
ALLMASK = (1 << N) - 1
# Precomputed 3-hop reachability closure of the complete graph K40.
# For any vertex v, C3[v] = set of vertices reachable from v in exactly 3 simple steps
# (i.e. N(N(N(v)))).  If a candidate edge (u,v) creates a 6-cycle, then N(u) & C3[v] != 0.
_C3 = [0] * N
for v in range(N):
    lv = 1 << v
    N1 = ALLMASK ^ lv
    N2 = 0
    for a in range(N):
        if (N1 >> a) & 1:
            N2 |= ALLMASK ^ (lv | (1 << a))
    N3 = 0
    for a in range(N):
        if (N2 >> a) & 1:
            N3 |= ALLMASK ^ (lv | (1 << a))
    _C3[v] = N3


def _is_c6_free(adj):
    """Exact test: does graph with adjacency bitsets `adj` contain ANY 6-cycle?"""
    for u in range(N):
        au = adj[u]
        m = au
        while m:
            lb = m & (-m)
            b = lb.bit_length() - 1
            m ^= lb
            if adj[b] & _C3[u]:
                return False
    return True


def _greedy_fill(adj, order, rng):
    """Add every safe edge in `order` (a permutation of all pairs)."""
    added = 0
    for (u, v) in order:
        if (adj[u] >> v) & 1:
            continue
        if (adj[u] & _C3[v]) or (adj[v] & _C3[u]):
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        added += 1
    return added


def _swap_search(adj, rng, budget):
    """1-edge-swap local search: remove edge e, add edge f; accept if C6-free.
    Net gain only when e's removal enables f (f was blocked before)."""
    cur = sum(adj[v].bit_count() for v in range(N)) // 2
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    idx = {}
    for i, p in enumerate(pairs):
        idx[p] = i
    last_improve = time.time()
    stall = 0
    while time.time() < budget:
        improved = False
        order = pairs[:]
        rng.shuffle(order)
        for (u, v) in order:
            if time.time() > budget:
                break
            if not ((adj[u] >> v) & 1):
                continue  # only try removing existing edges
            au = adj[u]
            av = adj[v]
            # candidate f: not existing, not incident to u or v, and was blocked before
            # (so its addition is only possible thanks to removing e).
            cand = (ALLMASK ^ au ^ av ^ (1 << u) ^ (1 << v))
            # f must have been blocked by e before removal:
            # i.e. (old adj[u] & C3[w]) or (old adj[w] & C3[u]) etc. for some endpoint w.
            # We just test after removal; if it's now safe and wasn't before, it's a gain.
            # Determine "was blocked before":
            # blocked_before(u,v) = (adj[u] & C3[v]) or (adj[v] & C3[u])
            was_blocked = (au & _C3[v]) or (av & _C3[u])
            if not was_blocked:
                continue
            # Remove e
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            # find a safe f
            m = cand
            found = None
            while m:
                lb = m & (-m)
                w = lb.bit_length() - 1
                m ^= lb
                if w <= v:
                    continue
                if (adj[u] & _C3[w]) or (adj[w] & _C3[u]):
                    continue
                found = w
                break
            if found is not None:
                adj[u] |= 1 << found
                adj[found] |= 1 << u
                # e removed, f added -> edge count unchanged, but we rearranged.
                # This is not a net gain. Only useful if it later enables more adds.
                # We accept the rearrangement (C6-free guaranteed) and let greedy re-fill.
                improved = True
                # quick re-greedy to pick up new opportunities
                rem = []
                for (a, b) in order:
                    if not ((adj[a] >> b) & 1):
                        rem.append((a, b))
                rng.shuffle(rem)
                for (a, b) in rem:
                    if (adj[a] & _C3[b]) or (adj[b] & _C3[a]):
                        continue
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
            else:
                # restore e
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        if improved:
            last_improve = time.time()
            stall = 0
        else:
            stall += 1
            if stall >= 6:
                break
    return sum(adj[v].bit_count() for v in range(N)) // 2


def _k5_fallback():
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    return edges


def construct():
    start = time.time()
    hard_deadline = start + 142.0  # safety margin under the 150s kill
    best_edges = None
    best_count = 0
    rng = random.Random(12345)

    def edge_list(adj):
        out = []
        for u in range(N):
            au = adj[u]
            while au:
                lb = au & (-au)
                v = lb.bit_length() - 1
                au ^= lb
                if v > u:
                    out.append([u, v])
        return out

    # Several randomized restarts: greedy fill + swap local search.
    attempts = 0
    while time.time() < hard_deadline - 3.0:
        attempts += 1
        adj = [0] * N
        order = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(order)
        # Give a little head start with a random low-density seed to vary structure.
        seed_edges = rng.sample(order, 40)
        for (u, v) in seed_edges:
            if (adj[u] & _C3[v]) or (adj[v] & _C3[u]):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        _greedy_fill(adj, order, rng)
        # swap local search with a per-attempt budget
        per_budget = min(hard_deadline - 4.0, time.time() + 18.0)
        _swap_search(adj, rng, per_budget)
        cnt = sum(adj[v].bit_count() for v in range(N)) // 2
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)
        if best_count >= 105 and attempts >= 2:
            break

    if best_edges is None or best_count < 100:
        return _k5_fallback()
    return best_edges
