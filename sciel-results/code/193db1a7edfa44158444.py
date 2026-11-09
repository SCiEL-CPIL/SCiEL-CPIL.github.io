"""C6-free graph on 40 vertices: greedy build + local improvement."""
import time
import random

N = 40
DEADLINE = 115.0  # safety margin before 150s kill


def _has_c6(adj, u, v):
    """Check if adding edge (u,v) to graph with adjacency sets `adj` creates a C6.
    A C6 containing (u,v) is a path of length 5 from u to v in the current graph.
    Equivalently, there exist a,b,c,d with u-a-b-c-d-v all distinct (a..d not u,v).
    """
    if u in adj[v]:
        return True
    # Find all length-5 paths u -> a -> b -> c -> d -> v
    # a in N(u), b in N(a)\{u}, c in N(b)\{a}, d in N(c)\{b}, and d in N(v)
    nu = adj[u]
    nv = adj[v]
    for a in nu:
        for b in adj[a]:
            if b == u or b == v:
                continue
            for c in adj[b]:
                if c == a or c == u or c == v:
                    continue
                for d in adj[c]:
                    if d == b or d == u or d == v:
                        continue
                    if d in nv:
                        return True
    return False


def _creates_c6(adj, u, v):
    """Same as _has_c6 but assumes edge (u,v) is NOT yet present."""
    nu = adj[u]
    nv = adj[v]
    for a in nu:
        if a == v:
            continue
        for b in adj[a]:
            if b == u or b == v:
                continue
            for c in adj[b]:
                if c == a or c == u or c == v:
                    continue
                for d in adj[c]:
                    if d == b or d == u or d == v:
                        continue
                    if d in nv:
                        return True
    return False


def _edge_list(adj):
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges


def _try_add(adj, u, v):
    if u == v or v in adj[u]:
        return False
    if _creates_c6(adj, u, v):
        return False
    adj[u].add(v)
    adj[v].add(u)
    return True


def _try_remove(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)


def construct():
    start = time.time()
    rng = random.Random(42)

    best_adj = [set() for _ in range(N)]
    best_count = 0

    # Phase 1: Greedy build in random order, multiple restarts
    for trial in range(50):
        if time.time() - start > DEADLINE - 30:
            break
        adj = [set() for _ in range(N)]
        order = list(range(N * (N - 1) // 2))
        # Map index to pair
        def pair_at(i):
            # unrank combination
            # i = u*(N-1) - u*(u+1)//2 + (v-u-1)  ... simpler: precompute
            return None
        # Precompute all pairs
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(pairs)
        for (u, v) in pairs:
            if time.time() - start > DEADLINE - 30:
                break
            if not _creates_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        cnt = sum(len(s) for s in adj) // 2
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]

    # Phase 2: Local improvement - try to add edges that were rejected,
    # or swap: remove one edge, add two (net +1)
    adj = [set(s) for s in best_adj]
    improved = True
    while improved and time.time() - start < DEADLINE - 5:
        improved = False
        # Try adding any missing edge
        for u in range(N):
            if time.time() - start > DEADLINE - 5:
                break
            for v in range(u + 1, N):
                if v in adj[u]:
                    continue
                if not _creates_c6(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
        # Try swaps: remove edge (a,b), add edges (c,d) and (e,f)
        # This is expensive; do a limited random search
        if time.time() - start < DEADLINE - 5:
            for _ in range(2000):
                if time.time() - start > DEADLINE - 5:
                    break
                # pick a random existing edge to remove
                u = rng.randrange(N)
                if not adj[u]:
                    continue
                v = rng.choice(list(adj[u]))
                if u > v:
                    u, v = v, u
                # temporarily remove
                adj[u].discard(v)
                adj[v].discard(u)
                # try to add two random missing edges
                added = []
                for _ in range(2):
                    c = rng.randrange(N)
                    d = rng.randrange(N)
                    if c == d or d in adj[c]:
                        continue
                    if not _creates_c6(adj, c, d):
                        adj[c].add(d)
                        adj[d].add(c)
                        added.append((c, d))
                if len(added) >= 1:
                    improved = True
                    # keep it
                else:
                    # revert
                    adj[u].add(v)
                    adj[v].add(u)

    cnt = sum(len(s) for s in adj) // 2
    if cnt > best_count:
        best_count = cnt
        best_adj = adj

    edges = _edge_list(best_adj)
    return edges


if __name__ == "__main__":
    e = construct()
    print(len(e))
