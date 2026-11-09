import time
import random
import numpy as np

N = 40
DEADLINE = 115  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    adj: list of int bitmasks.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We look for a path u-a-b-c-d-v
    # a is in N(u), d is in N(v)
    # b is in N(a), c is in N(d)
    # b-c is an edge
    # u, v, a, b, c, d are distinct
    
    # Iterate over a in N(u)
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        if a == v: continue
        
        # Iterate over d in N(v)
        d_mask = nv
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            if d == u or d == a: continue
            
            # Forbidden vertices for b and c: u, v, a, d
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            # b in N(a) \ forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            
            # c in N(d) \ forbidden
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue
            
            # Check if any b in b_mask is connected to any c in c_mask
            # We can iterate over b and check adj[b] & c_mask
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

def _local_search(adj, deadline, rng):
    """
    Local search with add and swap moves.
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    current = adj[:]
    current_count = best_count
    
    while time.time() < deadline:
        improved = False
        random.shuffle(all_edges)
        
        # Pass 1: Try to add edges
        for u, v in all_edges:
            if (current[u] >> v) & 1:
                continue
            if _creates_c6(current, u, v):
                continue
            _add_edge(current, u, v)
            current_count += 1
            improved = True
            if current_count > best_count:
                best_count = current_count
                adj = current[:]
        
        # Pass 2: Try swaps (remove one, add one) to escape local optima
        # We do this a few times to allow multiple swaps
        for _ in range(3):
            if time.time() >= deadline:
                break
            random.shuffle(all_edges)
            for e1 in all_edges:
                u1, v1 = e1
                if not ((current[u1] >> v1) & 1):
                    continue
                
                _remove_edge(current, u1, v1)
                current_count -= 1
                
                added = False
                # Try to add a random edge
                for _ in range(20):
                    u2, v2 = rng.choice(all_edges)
                    if (current[u2] >> v2) & 1:
                        continue
                    if _creates_c6(current, u2, v2):
                        continue
                    _add_edge(current, u2, v2)
                    current_count += 1
                    added = True
                    improved = True
                    break
                
                if not added:
                    # Revert
                    _add_edge(current, u1, v1)
                    current_count += 1
                else:
                    # If we added, we might have improved best count? No, count is same.
                    # But structure changed.
                    if current_count > best_count:
                        best_count = current_count
                        adj = current[:]
                        
        if not improved:
            break
            
    return adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)
    
    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed
    seed = _k5_seed_adj()
    seed, c = _local_search(seed, deadline, rng)
    refresh_best(seed)

    # Phase 2: Multiple random restarts with greedy init
    while time.time() < deadline:
        adj = [0] * N
        # Greedy init: add edges in random order if C6-free
        all_edges = _make_all_edges()
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
        
        adj, c = _local_search(adj, deadline, rng)
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
