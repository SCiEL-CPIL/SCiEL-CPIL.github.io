"""
Combined program: uses the PG(2,5) polarity graph as a C4-free 90-edge core,
then applies fast bitmask-based local search to find the single extra edge
that adds the fewest C4s.
"""
import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    n = 31
    target_edges = 91
    q = 5

    # --- Build PG(2,5) polarity graph (90 edges, C4-free) ---
    def gf_inv(a, mod=5):
        return pow(a, mod - 2, mod)

    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = gf_inv(x, q)
                    nx, ny, nz = 1, (y * inv_x) % q, (z * inv_x) % q
                elif y != 0:
                    inv_y = gf_inv(y, q)
                    nx, ny, nz = 0, 1, (z * inv_y) % q
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 31, f"Expected 31 points, got {len(points)}"

    def adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        val = (x1 * x2 + y1 * y2 - z1 * z2) % q
        return val == 0

    core_edges = []
    for i in range(31):
        for j in range(i + 1, 31):
            if adjacent(i, j):
                core_edges.append((i, j))

    if len(core_edges) != 90:
        # Fallback: greedy construction
        return _greedy_construct(t0, deadline)

    # Build bitmask adjacency
    adj = [0] * n
    for u, v in core_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # --- Fast C4 counter using bitmasks ---
    def count_c4(adj):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj[i] & adj[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    # --- Find best single edge to add ---
    # Adding edge (u,v) creates C4s = number of length-3 paths from u to v
    # = sum over a in N(u) of |N(a) ∩ N(v)| ... wait, that's not quite right.
    # Actually: new C4s from adding (u,v) = number of pairs (a,b) such that
    # a ~ u, b ~ v, and a ~ b. This equals sum over a in N(u) of |N(a) ∩ N(v)|
    # but only counting a != v and b != u (which is automatic since u,v not adjacent yet).
    # Wait, more carefully: a C4 containing new edge (u,v) has form u-a-b-v-u
    # where a ~ u, b ~ v, a ~ b, a != v, b != u, a != b.
    # Since u and v are not adjacent, a != v and b != u are automatic.
    # Also a != b is needed.
    
    best_edge = None
    best_add = float('inf')

    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                continue
            # Count new C4s: pairs (a, b) with a in N(u), b in N(v), a ~ b, a != b
            add = 0
            nu = adj[u]
            nv = adj[v]
            for a in range(n):
                if (nu >> a) & 1:
                    # count b in N(v) such that a ~ b and b != a
                    common = adj[a] & nv
                    # b must be in N(v), adjacent to a, and b != a
                    cnt = common.bit_count()
                    # if a is in N(v), we need to exclude b=a
                    if (nv >> a) & 1:
                        cnt -= 1
                    if cnt > 0:
                        add += cnt
            if add < best_add:
                best_add = add
                best_edge = (u, v)
            if add == 0:
                break
        if best_add == 0:
            break

    if best_edge is None:
        return _greedy_construct(t0, deadline)

    # Add the best edge
    u, v = best_edge
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    final_edges = list(core_edges) + [(u, v)]

    # Verify exactly 91 distinct edges
    edge_set = set()
    for e in final_edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        edge_set.add(key)
    assert len(edge_set) == 91, f"Expected 91 edges, got {len(edge_set)}"

    # --- Local search to see if we can do better ---
    # Try swapping the added edge with other non-edges, or even restructuring
    # But first, let's try a quick local search on the full 91-edge graph.
    
    current_adj = adj[:]
    current_edges = final_edges[:]
    current_c4 = count_c4(current_adj)
    
    best_adj = current_adj[:]
    best_edges = current_edges[:]
    best_c4 = current_c4

    if time.time() < deadline - 10:
        time_left = deadline - time.time() - 10
        best_adj, best_edges, best_c4 = _local_search(
            current_adj, current_edges, current_c4, seed=42, time_limit=time_left
        )

    return [list(e) for e in best_edges]


def _local_search(adj, edges, c4, seed, time_limit):
    """Fast bitmask local search: swap one edge for another."""
    random.seed(seed)
    n = 31
    current_adj = adj[:]
    current_edges = edges[:]
    current_c4 = c4
    edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)

    best_adj = current_adj[:]
    best_edges = current_edges[:]
    best_c4 = current_c4

    t_start = time.time()
    T = 1.0
    cooling_rate = 0.99995

    def count_c4_local(adj_local):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj_local[i] & adj_local[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    while time.time() - t_start < time_limit:
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        key_old = (min(u, v), max(u, v))

        # Pick a random non-edge
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                break

        if (x, y) == key_old:
            continue

        # Remove old edge
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        edge_set.remove(key_old)

        # Add new edge
        current_adj[x] |= (1 << y)
        current_adj[y] |= (1 << x)
        edge_set.add((x, y))

        current_edges[idx] = (x, y)

        new_c4 = count_c4_local(current_adj)
        delta = new_c4 - current_c4

        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0.01:
                if random.random() < math.exp(-delta / T):
                    accept = True

        if accept:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = current_adj[:]
                best_edges = current_edges[:]
            T *= cooling_rate
        else:
            # Revert
            current_adj[x] &= ~(1 << y)
            current_adj[y] &= ~(1 << x)
            edge_set.remove((x, y))

            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            edge_set.add(key_old)

            current_edges[idx] = (u, v)

    return best_adj, best_edges, best_c4


def _greedy_construct(t0, deadline):
    """Fallback greedy construction if polarity graph fails."""
    random.seed(42)
    n = 31
    target_edges = 91

    def count_c4(adj):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj[i] & adj[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    adj = [0] * n
    edges = []
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)

    for u, v in all_edges:
        if len(edges) >= target_edges:
            break
        if (adj[u] >> v) & 1:
            continue
        # Count new C4s
        cost = 0
        for a in range(n):
            if (adj[u] >> a) & 1:
                common = (adj[a] & adj[v]).bit_count()
                cost += common
        if cost <= 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))

    # If we didn't get enough edges, just add more randomly
    if len(edges) < target_edges:
        remaining = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if not ((adj[i] >> j) & 1)]
        random.shuffle(remaining)
        for u, v in remaining:
            if len(edges) >= target_edges:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))

    # Quick local search
    if time.time() < deadline - 10:
        time_left = deadline - time.time() - 10
        c4 = count_c4(adj)
        adj, edges, c4 = _local_search(adj, edges, c4, seed=99, time_limit=time_left)

    return [list(e) for e in edges]
