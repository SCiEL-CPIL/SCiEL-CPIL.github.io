import time
import random
import numpy as np

N = 40
DEADLINE = 115  # wall-clock seconds (safety margin under 150s kill)

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    That happens iff there is a simple path of length 5 between u and v:
        u - a - b - c - d - v
    with all six vertices distinct.
    adj: list of int bitmasks, adj[i] has bit j set iff edge (i, j) present.
    """
    nu = adj[u]
    nv = adj[v]
    uv_mask = (1 << u) | (1 << v)

    # Iterate a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # Iterate d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = uv_mask | (1 << a) | (1 << d)

            # b in N(a) \ forbidden, c in N(d) \ forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if adj[b] & c_mask:
                    return True

    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _random_seed(adj):
    """Add random edges greedily to seed."""
    all_edges = _make_all_edges()
    random.shuffle(all_edges)
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _local_search(adj, max_iter, deadline, rng):
    """
    Local search: greedy fill + systematic 2-opt swaps.
    Returns (best_adj, best_count).
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False

        # Pass 1: greedy additions in random order
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True

        # Pass 2: systematic 2-opt swaps
        # For each existing edge, try removing it and see if we can add
        # more edges than we removed (net gain)
        swap_order = all_edges[:]
        rng.shuffle(swap_order)
        for e1 in swap_order:
            if time.time() >= deadline:
                break
            u1, v1 = e1
            if not ((adj[u1] >> v1) & 1):
                continue
            _remove_edge(adj, u1, v1)

            # Try to add as many edges as possible after removal
            added = 0
            add_order = all_edges[:]
            rng.shuffle(add_order)
            for u2, v2 in add_order:
                if time.time() >= deadline:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added += 1
                if added >= 2:
                    break

            if added > 0:
                improved = True
            else:
                # No improvement, restore original edge
                _add_edge(adj, u1, v1)

        # Update best
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
        # If no improvement, stop early
        if not improved:
            break
    return adj, best_count

def _k5_seed_adj():
    """
    Build a C6-free seed from 8 disjoint K5's (40 vertices).
    Returns adj list.
    """
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _anneal_from(adj, deadline, rng):
    """
    Simulated annealing on top of adj:
    - propose add/swap/remove to move toward higher edge count while C6-free.
    Accept improves always; accept neutral/swap with probability to escape.
    Returns best_adj, best_count.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()

    t = deadline
    min_temp = 0.01
    # Time-based schedule
    start = time.time()
    total = max(1e-6, t - start)

    while time.time() < t:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total)
        temp = max(min_temp, 1.0 - frac)

        # Choose move type
        r = rng.random()
        if r < 0.55:
            # Add a random edge
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.85:
            # Swap: remove a random existing edge, try to add a random edge
            e1 = rng.choice(all_edges)
            u1, v1 = e1
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            added = False
            for _ in range(15):
                e2 = rng.choice(all_edges)
                u2, v2 = e2
                if (cur[u2] >> v2) & 1:
                    continue
                if _creates_c6(cur, u2, v2):
                    continue
                _add_edge(cur, u2, v2)
                added = True
                move = ("swap", (u1, v1), (u2, v2))
                break
            if not added:
                _add_edge(cur, u1, v1)
                continue
            delta = 0
        else:
            # Remove a random edge (escape)
            e1 = rng.choice(all_edges)
            u1, v1 = e1
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            delta = -1
            move = ("remove", u1, v1)

        new_count = cur_count + delta
        if new_count > best_count:
            best = cur[:]
            best_count = new_count
        cur_count = new_count

        # Accept/reject
        if delta > 0:
            continue
        # delta <= 0: accept with probability
        p = 1.0 if delta == 0 else min(1.0, temp * 0.5)
        if rng.random() < p:
            continue
        else:
            # revert
            if move[0] == "add":
                _remove_edge(cur, move[1], move[2])
            elif move[0] == "swap":
                (u2, v2), (u1, v1) = move[2], move[1]
                _remove_edge(cur, u2, v2)
                _add_edge(cur, u1, v1)
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
            cur_count = _count_edges(cur)
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(12345)
    all_edges = _make_all_edges()

    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed + local search + anneal
    seed = _k5_seed_adj()
    # Greedy add from random order
    order = all_edges[:]
    rng.shuffle(order)
    for u, v in order:
        if (seed[u] >> v) & 1:
            continue
        if _creates_c6(seed, u, v):
            continue
        _add_edge(seed, u, v)
    seed, c = _local_search(seed, max_iter=150, deadline=deadline, rng=rng)
    refresh_best(seed)
    if time.time() < deadline:
        seed, c = _anneal_from(seed, deadline, rng)
        refresh_best(seed)

    # Phase 2: Multiple random-start greedy builds with local search / anneal
    # Use time chunks: each iteration gets a bounded time slice
    while time.time() < deadline:
        # Allocate time for this iteration
        remaining = deadline - time.time()
        if remaining < 5:
            break
        iter_deadline = time.time() + min(remaining, 25)

        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
        adj, c = _local_search(adj, max_iter=80, deadline=iter_deadline, rng=rng)
        refresh_best(adj)
        if time.time() < iter_deadline:
            adj, c = _anneal_from(adj, iter_deadline, rng)
            refresh_best(adj)

    # Final polish: try greedy fill on the best solution one more time
    if time.time() < deadline - 2:
        polish_adj = best_adj[:]
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if time.time() >= deadline - 1:
                break
            if (polish_adj[u] >> v) & 1:
                continue
            if _creates_c6(polish_adj, u, v):
                continue
            _add_edge(polish_adj, u, v)
        refresh_best(polish_adj)

    # Reconstruct edge list from best_adj
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
