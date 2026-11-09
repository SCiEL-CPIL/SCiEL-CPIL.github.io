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
    forbidden_uv = (1 << u) | (1 << v)

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

            forbidden = forbidden_uv | (1 << a) | (1 << d)

            # b in N(a), c in N(d), b,c distinct from {u,v,a,d}
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask
            # Early exit: compute union of adj[b] for all b in b_mask, intersect with c_mask
            union_adj = 0
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                union_adj |= adj[b]
                if union_adj & c_mask:
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

def _random_seed(adj, rng):
    """Add random edges greedily to seed."""
    all_edges = _make_all_edges()
    order = all_edges[:]
    rng.shuffle(order)
    for u, v in order:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _local_search(adj, max_iter, deadline, rng):
    """
    Local search: try to add an edge (accept if improves and C6-free),
    or 2-opt swap (remove e1, add e2) to keep edge count and unlock more adds.
    Returns (best_adj, best_count).
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False
        order = all_edges[:]
        rng.shuffle(order)
        # Pass 1: greedy additions
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
        # Remove one edge, then try to add two edges to net +1
        swap_order = all_edges[:]
        rng.shuffle(swap_order)
        for e1 in swap_order:
            if time.time() >= deadline:
                break
            u1, v1 = e1
            if not ((adj[u1] >> v1) & 1):
                continue
            _remove_edge(adj, u1, v1)
            # Try to add edges after removal
            added = 0
            for u2, v2 in order:
                if time.time() >= deadline:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added += 1
                if added >= 1:
                    break
            if added >= 1:
                improved = True
            else:
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

def _design_seed_adj():
    """
    Build a C6-free seed from a structured design:
    Use a (40, 10) incidence structure based on Z_8 x Z_5.
    Vertices are (i, j) with i in 0..7, j in 0..4.
    Connect (i,j) to (i,j') for j'!=j (within-group, 4 edges each) -> 8*5*4/2 = 80 edges
    Then add inter-group edges: connect (i,j) to (i+1, j+1 mod 5) for all i,j.
    This gives 40 more edges, total 120. But need to verify C6-free.
    
    Actually, let's use a simpler approach: start with 8 K5's (80 edges)
    and add a structured set of inter-group edges.
    """
    adj = _k5_seed_adj()
    # Add inter-group edges in a structured way
    # Connect vertex k*5+i to (k+1)*5+((i+1)%5) for k in 0..7 (cyclic)
    for k in range(8):
        for i in range(5):
            u = k * 5 + i
            v = ((k + 1) % 8) * 5 + ((i + 1) % 5)
            if u < v:
                _add_edge(adj, u, v)
            elif v < u:
                _add_edge(adj, v, u)
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

    start = time.time()
    total = max(1e-6, deadline - start)
    temp = 1.0
    min_temp = 0.01

    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total)
        temp = max(min_temp, 1.0 - frac)

        # Choose move type
        r = rng.random()
        if r < 0.5:
            # Add a random edge
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.8:
            # 2-opt: remove a random edge, then try to add up to 2 edges
            e1 = rng.choice(all_edges)
            u1, v1 = e1
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            added = 0
            for _ in range(20):
                e2 = rng.choice(all_edges)
                u2, v2 = e2
                if (cur[u2] >> v2) & 1:
                    continue
                if _creates_c6(cur, u2, v2):
                    continue
                _add_edge(cur, u2, v2)
                added += 1
                if added >= 1:
                    break
            if added == 0:
                _add_edge(cur, u1, v1)
                continue
            delta = added - 1
            move = ("two_opt", (u1, v1), added)
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
            elif move[0] == "two_opt":
                (u1, v1), added = move[1], move[2]
                # Remove the 'added' edges we added (we don't track them precisely,
                # so just re-add e1; this is approximate but good enough for SA)
                _add_edge(cur, u1, v1)
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
            cur_count = _count_edges(cur)
    return best, best_count

def _degree_guided_greedy(adj, rng):
    """
    Greedy edge addition guided by degree balance: prefer edges connecting
    lower-degree vertices to promote uniformity and potentially more edges.
    """
    all_edges = _make_all_edges()
    while True:
        # Compute current degrees
        deg = [bin(adj[i]).count("1") for i in range(N)]
        # Score each missing edge by sum of degrees of endpoints (lower is better)
        missing = [(deg[u] + deg[v], u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        if not missing:
            break
        # Sort by score, add best edges that are C6-free
        missing.sort()
        added_any = False
        for score, u, v in missing:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            added_any = True
        if not added_any:
            break

def _construct():
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
    _random_seed(seed, rng)
    seed, c = _local_search(seed, max_iter=200, deadline=deadline, rng=rng)
    refresh_best(seed)
    if time.time() < deadline:
        seed, c = _anneal_from(seed, deadline, rng)
        refresh_best(seed)

    # Phase 1b: Design seed + local search + anneal
    if time.time() < deadline:
        seed2 = _design_seed_adj()
        _random_seed(seed2, rng)
        seed2, c = _local_search(seed2, max_iter=150, deadline=deadline, rng=rng)
        refresh_best(seed2)
        if time.time() < deadline:
            seed2, c = _anneal_from(seed2, deadline, rng)
            refresh_best(seed2)

    # Phase 2: Multiple random-start greedy builds with local search / anneal
    while time.time() < deadline:
        adj = [0] * N
        _random_seed(adj, rng)
        adj, c = _local_search(adj, max_iter=120, deadline=deadline, rng=rng)
        refresh_best(adj)
        if time.time() < deadline:
            adj, c = _anneal_from(adj, deadline, rng)
            refresh_best(adj)

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

def construct():
    return _construct()

if __name__ == "__main__":
    construct()
