import time
import random
import numpy as np

N = 40
DEADLINE = 108  # safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u,v) creates a C6.
    C6 exists iff there is a simple path u-a-b-c-d-v of length 5.
    Using bitmasks: reach2[u] = vertices at distance exactly 2 from u,
    reach3[u] = vertices at distance exactly 3 from u.
    A C6 exists iff reach2[u] & reach3[v] is non-empty (with distinctness).
    """
    nu = adj[u]
    nv = adj[v]

    # reach2[u] = union of adj[a] for a in N(u), a != v
    # But we need to be careful: reach2 should be vertices reachable in exactly 2 steps
    # For the C6 check, we need: exists a in N(u), d in N(v), b in N(a)\{u,v,d}, c in N(d)\{u,v,a}
    # such that b-c is an edge and all 6 vertices distinct.
    #
    # Faster: for each a in N(u) (a!=v), for each d in N(v) (d not in {u,a}):
    #   need exists b in N(a) \ {u,v,a,d} and c in N(d) \ {u,v,a,d} with edge b-c.
    #   Equivalent: (N(a) \ {u,v,a,d}) & (N(N(d) \ {u,v,a,d})) != 0
    #   But simpler: for each a, compute candidates_b = N(a) & ~{u,v,a,d}
    #                for each d, candidates_c = N(d) & ~{u,v,a,d}
    #                check if any b in candidates_b has edge to any c in candidates_c.
    #
    # Even faster with bitmasks:
    #   For fixed a: b_mask_base = adj[a] & ~(1<<u) & ~(1<<v) & ~(1<<a)
    #   For each d in N(v) \ {u,a}: 
    #       b_mask = b_mask_base & ~(1<<d)
    #       c_mask = adj[d] & ~(1<<u) & ~(1<<v) & ~(1<<a) & ~(1<<d)
    #       if any b in b_mask is adjacent to any c in c_mask: C6 exists.
    #       Check: OR of adj[b] for b in b_mask, then & c_mask.

    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        b_base = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
        if not b_base:
            continue

        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            b_mask = b_base & ~(1 << d)
            if not b_mask:
                continue
            c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
            if not c_mask:
                continue

            # Check if any vertex in b_mask is adjacent to any vertex in c_mask
            # Union of neighborhoods of all b in b_mask, intersect with c_mask
            bm = b_mask
            union = 0
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                union |= adj[b]
            if union & c_mask:
                return True

    return False


def _count_edges(adj):
    total = 0
    for i in range(N):
        total += adj[i].bit_count()
    return total // 2


def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _greedy_from_order(adj, order):
    count = 0
    for u, v in order:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count


def _local_search(adj, deadline, rng, max_rounds=50):
    """
    Local search: repeatedly try to add any missing edge that keeps C6-free.
    Then try swaps to escape local optima.
    """
    all_edges = _make_all_edges()
    best_count = _count_edges(adj)
    cur = adj[:]

    for rnd in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        # Try all missing edges
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        for u, v in shuffled:
            if time.time() >= deadline:
                break
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            improved = True

        cnt = _count_edges(cur)
        if cnt > best_count:
            best_count = cnt
            adj = cur[:]

        if not improved:
            # Try swaps: remove one edge, add another
            swap_improved = False
            present_edges = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if present_edges:
                rng.shuffle(present_edges)
                for u1, v1 in present_edges[:20]:
                    if time.time() >= deadline:
                        break
                    _remove_edge(cur, u1, v1)
                    added = False
                    for _ in range(15):
                        u2, v2 = rng.choice(all_edges)
                        if (cur[u2] >> v2) & 1:
                            continue
                        if _creates_c6(cur, u2, v2):
                            continue
                        _add_edge(cur, u2, v2)
                        added = True
                        break
                    if added:
                        swap_improved = True
                    else:
                        _add_edge(cur, u1, v1)
                    if swap_improved:
                        break
            if not swap_improved:
                break

    return adj, best_count


def _anneal(adj, deadline, rng):
    """
    Simulated annealing: accept improving moves always, 
    accept neutral with some probability, accept worsening rarely.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()

    start = time.time()
    total_time = max(1.0, deadline - start)

    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total_time)
        temp = max(0.01, 1.0 - frac)

        r = rng.random()
        if r < 0.5:
            # Add random edge
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.8:
            # Swap
            present = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if not present:
                continue
            u1, v1 = rng.choice(present)
            _remove_edge(cur, u1, v1)
            added = False
            for _ in range(10):
                u2, v2 = rng.choice(all_edges)
                if (cur[u2] >> v2) & 1:
                    continue
                if _creates_c6(cur, u2, v2):
                    continue
                _add_edge(cur, u2, v2)
                added = True
                break
            if not added:
                _add_edge(cur, u1, v1)
                continue
            move = ("swap", (u1, v1), (u2, v2))
            delta = 0
        else:
            # Remove random edge (escape)
            present = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if not present:
                continue
            u1, v1 = rng.choice(present)
            _remove_edge(cur, u1, v1)
            delta = -1
            move = ("remove", u1, v1)

        new_count = cur_count + delta
        if new_count > best_count:
            best_count = new_count
            best = cur[:]
        cur_count = new_count

        # Accept/reject
        if delta >= 0:
            continue
        # delta < 0: accept with probability temp
        if rng.random() < temp * 0.3:
            continue
        else:
            # Revert
            if move[0] == "add":
                _remove_edge(cur, move[1], move[2])
            elif move[0] == "swap":
                (u1, v1), (u2, v2) = move[1], move[2]
                _remove_edge(cur, u2, v2)
                _add_edge(cur, u1, v1)
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
            cur_count = _count_edges(cur)

    return best, best_count


def _k5_seed():
    """8 disjoint K5s as starting point."""
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                _add_edge(adj, base + i, base + j)
    return adj


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)
    all_edges = _make_all_edges()

    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed + greedy + local search + anneal
    seed = _k5_seed()
    order = all_edges[:]
    rng.shuffle(order)
    _greedy_from_order(seed, order)
    refresh_best(seed)

    seed, c = _local_search(seed, deadline, rng, max_rounds=30)
    refresh_best(seed)

    seed, c = _anneal(seed, deadline, rng)
    refresh_best(seed)

    # Phase 2: Multiple random restarts
    while time.time() < deadline:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_from_order(adj, order)
        refresh_best(adj)

        adj, c = _local_search(adj, deadline, rng, max_rounds=20)
        refresh_best(adj)

        if time.time() < deadline:
            adj, c = _anneal(adj, deadline, rng)
            refresh_best(adj)

    # Reconstruct edge list
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges


if __name__ == "__main__":
    construct()
