import time
import random
import numpy as np

N = 40
DEADLINE = 115  # safety margin under 120s limit

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    """
    nu = adj[u]
    nv = adj[v]

    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

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

def _local_search(adj, max_iter, deadline, rng):
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False
        random.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        
        for _ in range(2):
            if time.time() >= deadline:
                break
            random.shuffle(all_edges)
            for e1 in all_edges:
                u1, v1 = e1
                if not ((adj[u1] >> v1) & 1):
                    continue
                _remove_edge(adj, u1, v1)
                added = False
                for _t in range(8):
                    u2, v2 = rng.choice(all_edges)
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added = True
                    improved = True
                    break
                if not added:
                    _add_edge(adj, u1, v1)
        
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
        if not improved:
            break
    return adj, best_count

def _k5_seed_adj():
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _perturb(adj, rng, k=3):
    edges = []
    for i in range(N):
        m = adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append((i, j))
            m ^= lsb
    
    if not edges:
        return adj[:]
        
    new_adj = adj[:]
    for _ in range(k):
        if not edges: break
        idx = rng.randrange(len(edges))
        u, v = edges[idx]
        edges.pop(idx)
        _remove_edge(new_adj, u, v)
    return new_adj

def _greedy_fill(adj, all_edges, rng):
    order = all_edges[:]
    rng.shuffle(order)
    for u, v in order:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
    return adj

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

    # Phase 1: K5 seed + local search
    seed = _k5_seed_adj()
    order = all_edges[:]
    rng.shuffle(order)
    _greedy_from_order(seed, order)
    seed, c = _local_search(seed, max_iter=200, deadline=deadline, rng=rng)
    refresh_best(seed)

    # Phase 2: Iterated Local Search
    while time.time() < deadline:
        k = rng.randint(2, 5)
        perturbed = _perturb(best_adj, rng, k)
        perturbed = _greedy_fill(perturbed, all_edges, rng)
        perturbed, c = _local_search(perturbed, max_iter=100, deadline=deadline, rng=rng)
        refresh_best(perturbed)

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
