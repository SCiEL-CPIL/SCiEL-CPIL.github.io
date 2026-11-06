import time
import random
import math

N = 40
DEADLINE = 110.0  # safety margin before 120s wall-clock kill
_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - _START)


def _has_c6(adj):
    """Return True iff the graph (adjacency as list of int bitmasks) contains a C6."""
    for a in range(N):
        a1 = adj[a]
        if a1 == 0:
            continue
        for b in range(N):
            if not (a1 >> b) & 1 or b <= a:
                continue
            a2 = adj[b]
            if a2 == 0:
                continue
            for c in range(N):
                if not (a2 >> c) & 1 or c <= b:
                    continue
                a3 = adj[c]
                if a3 == 0:
                    continue
                for d in range(N):
                    if not (a3 >> d) & 1 or d <= c:
                        continue
                    a4 = adj[d]
                    if a4 == 0:
                        continue
                    for e in range(N):
                        if not (a4 >> e) & 1 or e <= d:
                            continue
                        a5 = adj[e]
                        if a5 == 0:
                            continue
                        for f in range(N):
                            if not (a5 >> f) & 1 or f <= e:
                                continue
                            # f must be adjacent to a to close the 6-cycle
                            if (a5 >> f) & 1 and (adj[f] >> a) & 1:
                                return True
    return False


def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _count_edges(adj):
    return sum(bin(x).count('1') for x in adj) // 2


def construct():
    random.seed(20240517)

    def _timeup():
        return (time.time() - _START) >= DEADLINE

    # ---- Phase 1: build a dense random regular-ish seed with no C6 ----
    # Use a greedy degree-target construction: aim for degree ~ 10-12.
    def _build_seed(target_deg, seed):
        rng = random.Random(seed)
        adj = [0] * N
        deg = [0] * N
        # Randomize vertex order
        order = list(range(N))
        rng.shuffle(order)
        # Try to connect each vertex to target_deg others, avoiding C6
        for v in order:
            attempts = 0
            while deg[v] < target_deg and attempts < 200:
                attempts += 1
                w = rng.randrange(N)
                if w == v or (adj[v] >> w) & 1:
                    continue
                # tentatively add
                adj[v] |= (1 << w)
                adj[w] |= (1 << v)
                if _has_c6(adj):
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)
                else:
                    deg[v] += 1
                    deg[w] += 1
        return adj

    # Try several seeds and target degrees, keep the densest valid one.
    best_adj = None
    best_m = -1
    for target in (12, 11, 10, 9):
        for seed in range(6):
            if _timeup():
                break
            adj = _build_seed(target, seed)
            m = _count_edges(adj)
            if m > best_m and not _has_c6(adj):
                best_m = m
                best_adj = adj
        if _timeup():
            break

    if best_adj is None:
        # Fallback: 8 disjoint K5 (guaranteed valid, 80 edges)
        best_adj = [0] * N
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    _add_edge(best_adj, 5 * k + a, 5 * k + b)
        best_m = _count_edges(best_adj)

    # ---- Phase 2: local search (edge addition + swap) ----
    adj = list(best_adj)
    cur_m = best_m
    best_adj = list(adj)
    best_m = cur_m

    def _try_add_random(adj, m):
        # Pick a random non-edge; add if no C6.
        nonlocal cur_m, best_adj, best_m
        for _ in range(50):
            u = random.randrange(N)
            v = random.randrange(N)
            if u == v or (adj[u] >> v) & 1:
                continue
            _add_edge(adj, u, v)
            if _has_c6(adj):
                _remove_edge(adj, u, v)
            else:
                m += 1
                if m > best_m:
                    best_m = m
                    best_adj = list(adj)
                return m
        return m

    def _try_swap(adj, m):
        # Remove a random edge, then try to add two new edges (net +1) or
        # add one new edge (net 0) to explore the neighborhood.
        nonlocal cur_m, best_adj, best_m
        # pick a random existing edge
        e_list = []
        for u in range(N):
            x = adj[u]
            while x:
                lb = x & (-x)
                v = lb.bit_length() - 1
                if u < v:
                    e_list.append((u, v))
                x &= x - 1
        if not e_list:
            return m
        u, v = random.choice(e_list)
        _remove_edge(adj, u, v)
        m -= 1
        # try to add up to 2 new edges
        added = 0
        for _ in range(40):
            a = random.randrange(N)
            b = random.randrange(N)
            if a == b or (adj[a] >> b) & 1:
                continue
            _add_edge(adj, a, b)
            if _has_c6(adj):
                _remove_edge(adj, a, b)
            else:
                m += 1
                added += 1
                if added == 2:
                    break
        if m > best_m:
            best_m = m
            best_adj = list(adj)
        # If we couldn't get back to at least original m, keep the improved state
        # anyway (it's still valid). This allows exploration.
        return m

    # Simulated-annealing-ish loop
    t = 0.0
    T = 1.0
    cooling = 0.9995
    step = 0
    while _remaining() > 2.0:
        step += 1
        if step % 3 == 0:
            cur_m = _try_add_random(adj, cur_m)
        else:
            cur_m = _try_swap(adj, cur_m)
        T *= cooling
        if _timeup():
            break

    # Return best valid graph found
    edges = []
    for u in range(N):
        x = best_adj[u]
        while x:
            lb = x & (-x)
            v = lb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            x &= x - 1
    return edges


if __name__ == "__main__":
    e = construct()
    print(len(e))
