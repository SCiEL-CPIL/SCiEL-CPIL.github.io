import time
import random
import numpy as np

N = 40
DEADLINE = 115  # Safety margin under 150s kill

def _popcount(x):
    return bin(x).count("1")

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]

    # a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            # Neighbors of a, excluding forbidden
            na = adj[a] & ~forbidden
            # Neighbors of d, excluding forbidden
            nd = adj[d] & ~forbidden
            
            if not na or not nd:
                continue

            # Check if any b in na is connected to any c in nd
            # Iterate over the smaller set to minimize checks
            if _popcount(na) < _popcount(nd):
                m = na
                while m:
                    lsb_b = m & -m
                    b = lsb_b.bit_length() - 1
                    m ^= lsb_b
                    if adj[b] & nd:
                        return True
            else:
                m = nd
                while m:
                    lsb_c = m & -m
                    c = lsb_c.bit_length() - 1
                    m ^= lsb_c
                    if adj[c] & na:
                        return True

    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += _popcount(adj[i])
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _greedy_from_order(adj, order, deadline):
    """Try adding edges in the given order if it keeps C6-free."""
    count = 0
    for u, v in order:
        if time.time() > deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count

def _local_search(adj, deadline, rng):
    """
    Local search: try to add an edge, or swap (remove e1, add e2).
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    current_edges = set((min(u,v), max(u,v)) for u in range(N) if adj[u] & (1<<v) for v in range(N) if u < v)
    # Reconstruct current edge set properly
    current_edges = set()
    for u in range(N):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                current_edges.add((u, v))
            m ^= lsb
    
    it = 0
    while time.time() < deadline and it < 1000:
        it += 1
        improved = False
        
        # 1. Try to add a random non-edge
        non_edges = [(u, v) for u, v in all_edges if (u, v) not in current_edges]
        if non_edges:
            # Try a few random additions
            for _ in range(20):
                if time.time() > deadline:
                    break
                u, v = rng.choice(non_edges)
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                current_edges.add((u, v))
                non_edges.remove((u, v))
                improved = True
                break
        
        # 2. Try to swap: remove one edge, add one edge
        if not improved and current_edges:
            for _ in range(20):
                if time.time() > deadline:
                    break
                e1 = rng.choice(list(current_edges))
                u1, v1 = e1
                _remove_edge(adj, u1, v1)
                current_edges.remove(e1)
                
                # Try to add a random non-edge
                non_edges = [(u, v) for u, v in all_edges if (u, v) not in current_edges]
                added = False
                if non_edges:
                    for _ in range(5):
                        u2, v2 = rng.choice(non_edges)
                        if _creates_c6(adj, u2, v2):
                            continue
                        _add_edge(adj, u2, v2)
                        current_edges.add((u2, v2))
                        added = True
                        break
                
                if not added:
                    # Revert
                    _add_edge(adj, u1, v1)
                    current_edges.add(e1)
                else:
                    improved = True
                    # Note: count is same, but structure changed. 
                    # We don't update best_count here as it's not higher, 
                    # but it might enable future improvements.
        
        # Update best
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
        
        if not improved:
            # If stuck, break or do a more expensive move?
            # For now, break to save time for other strategies
            if it > 50:
                break

    return adj, best_count

def _anneal_from(adj, deadline, rng):
    """
    Simulated Annealing.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()
    
    # Current edge set for fast lookup
    cur_edges_set = set()
    for u in range(N):
        m = cur[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                cur_edges_set.add((u, v))
            m ^= lsb

    start = time.time()
    total_time = max(1e-6, deadline - start)
    temp = 1.0
    
    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total_time)
        temp = max(0.01, 1.0 - frac)

        r = rng.random()
        if r < 0.5:
            # Add
            # Pick random non-edge
            non_edges = [(u, v) for u, v in all_edges if (u, v) not in cur_edges_set]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            cur_edges_set.add((u, v))
            delta = 1
            move = ("add", u, v)
        elif r < 0.9:
            # Swap
            if not cur_edges_set:
                continue
            e1 = rng.choice(list(cur_edges_set))
            u1, v1 = e1
            _remove_edge(cur, u1, v1)
            cur_edges_set.remove(e1)
            
            non_edges = [(u, v) for u, v in all_edges if (u, v) not in cur_edges_set]
            added = False
            if non_edges:
                for _ in range(10):
                    u2, v2 = rng.choice(non_edges)
                    if _creates_c6(cur, u2, v2):
                        continue
                    _add_edge(cur, u2, v2)
                    cur_edges_set.add((u2, v2))
                    added = True
                    move = ("swap", (u1, v1), (u2, v2))
                    break
            
            if not added:
                _add_edge(cur, u1, v1)
                cur_edges_set.add(e1)
                continue
            delta = 0
        else:
            # Remove
            if not cur_edges_set:
                continue
            e1 = rng.choice(list(cur_edges_set))
            u1, v1 = e1
            _remove_edge(cur, u1, v1)
            cur_edges_set.remove(e1)
            delta = -1
            move = ("remove", u1, v1)

        new_count = cur_count + delta
        if new_count > best_count:
            best = cur[:]
            best_count = new_count
        cur_count = new_count

        # Accept/Reject
        if delta > 0:
            continue
        p = 1.0 if delta == 0 else min(1.0, temp * 0.5)
        if rng.random() < p:
            continue
        else:
            # Revert
            if move[0] == "add":
                _remove_edge(cur, move[1], move[2])
                cur_edges_set.remove((move[1], move[2]))
            elif move[0] == "swap":
                (u1, v1), (u2, v2) = move[1], move[2]
                _remove_edge(cur, u2, v2)
                cur_edges_set.remove((u2, v2))
                _add_edge(cur, u1, v1)
                cur_edges_set.add((u1, v1))
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
                cur_edges_set.add((move[1], move[2]))
            cur_count = _count_edges(cur)
            
    return best, best_count

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

    # Phase 1: K5 Seed
    seed = _k5_seed_adj()
    order = all_edges[:]
    rng.shuffle(order)
    _greedy_from_order(seed, order, deadline)
    refresh_best(seed)
    
    seed, c = _local_search(seed, deadline, rng)
    refresh_best(seed)
    
    if time.time() < deadline:
        seed, c = _anneal_from(seed, deadline, rng)
        refresh_best(seed)

    # Phase 2: Random Restarts
    while time.time() < deadline:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_from_order(adj, order, deadline)
        refresh_best(adj)
        
        if time.time() < deadline:
            adj, c = _local_search(adj, deadline, rng)
            refresh_best(adj)
            
        if time.time() < deadline:
            adj, c = _anneal_from(adj, deadline, rng)
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
