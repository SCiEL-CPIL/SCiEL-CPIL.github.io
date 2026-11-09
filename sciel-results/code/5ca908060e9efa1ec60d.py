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

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            # b in N(a), c in N(d), b,c distinct from {u,v,a,d}
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Any b in b_mask connected to any c in c_mask?
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

def _spectral_seed(adj, target_deg, rng):
    """
    Build a near-regular C6-free graph by starting from a random regular-ish
    structure. Use degree balance as a spectral proxy: graphs with uniform
    degrees have more 'spectral spread' available for adding edges.
    
    Strategy: 
    1. Start with a random d-regular graph (approximate)
    2. Greedily add edges preferring lower-degree vertices
    """
    all_edges = _make_all_edges()
    deg = [bin(adj[i]).count("1") for i in range(N)]
    
    # First pass: add edges to reach target degree, preferring balanced
    remaining = all_edges[:]
    rng.shuffle(remaining)
    
    for u, v in remaining:
        if (adj[u] >> v) & 1:
            continue
        # Only add if both vertices are below target degree
        if deg[u] >= target_deg or deg[v] >= target_deg:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        deg[u] += 1
        deg[v] += 1
    
    # Second pass: fill remaining capacity with degree-balanced additions
    missing = [(deg[u] + deg[v], u, v) for (u, v) in all_edges 
               if not ((adj[u] >> v) & 1)]
    missing.sort()
    for score, u, v in missing:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _local_search(adj, max_iter, deadline, rng):
    """
    Local search: try to add an edge (accept if improves and C6-free),
    or swap (remove e1, add e2) to keep edge count and unlock more adds.
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
        # Pass 1: additions
        for u, v in order:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        # Pass 2: swaps (remove one, add one) -- keep count, may enable more
        for _ in range(5):
            if time.time() >= deadline:
                break
            order = all_edges[:]
            rng.shuffle(order)
            for e1 in order:
                u1, v1 = e1
                if not ((adj[u1] >> v1) & 1):
                    continue
                _remove_edge(adj, u1, v1)
                # Try adding some random edge
                for _t in range(15):
                    u2, v2 = rng.choice(all_edges)
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    improved = True
                    break
                else:
                    # No add was made, restore e1
                    _add_edge(adj, u1, v1)
                    continue
                if improved:
                    break
            if improved:
                break
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
        if r < 0.6:
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
            # Swap: remove a random edge, add a random edge
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

def _polish_pass(adj, deadline, rng):
    """
    Final high-precision polish: aggressively try all possible single-edge
    additions and swaps to squeeze out maximum edges.
    """
    all_edges = _make_all_edges()
    best_count = _count_edges(adj)
    
    # Multiple passes of greedy addition
    for _ in range(3):
        if time.time() >= deadline:
            break
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if time.time() >= deadline:
                    return adj, best_count
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                improved = True
                break
    
    # Try swaps to unlock more additions
    for _ in range(50):
        if time.time() >= deadline:
            break
        u1, v1 = rng.choice(all_edges)
        if not ((adj[u1] >> v1) & 1):
            continue
        _remove_edge(adj, u1, v1)
        added = False
        for _ in range(30):
            u2, v2 = rng.choice(all_edges)
            if (adj[u2] >> v2) & 1:
                continue
            if _creates_c6(adj, u2, v2):
                continue
            _add_edge(adj, u2, v2)
            added = True
            break
        if not added:
            _add_edge(adj, u1, v1)
        else:
            # After a successful swap, try to add more
            for u, v in all_edges:
                if time.time() >= deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
    
    cnt = _count_edges(adj)
    if cnt > best_count:
        best_count = cnt
    return adj, best_count

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

    # Phase 2: Spectral-style seed (near-regular) + local search
    if time.time() < deadline:
        adj = [0] * N
        _spectral_seed(adj, target_deg=5, rng=rng)
        adj, c = _local_search(adj, max_iter=150, deadline=deadline, rng=rng)
        refresh_best(adj)
        if time.time() < deadline:
            adj, c = _anneal_from(adj, deadline, rng)
            refresh_best(adj)

    # Phase 3: Multiple random-start greedy builds with local search / anneal
    while time.time() < deadline:
        adj = [0] * N
        _random_seed(adj, rng)
        adj, c = _local_search(adj, max_iter=120, deadline=deadline, rng=rng)
        refresh_best(adj)
        if time.time() < deadline:
            adj, c = _anneal_from(adj, deadline, rng)
            refresh_best(adj)

    # Phase 4: Final polish on best found
    if time.time() < deadline:
        best_adj, best_count = _polish_pass(best_adj, deadline, rng)

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
