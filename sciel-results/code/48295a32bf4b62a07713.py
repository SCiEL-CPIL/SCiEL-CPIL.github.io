import time
import random
import sys

N = 40
DEADLINE = 110  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    Faster: precompute N2[u] = set of vertices at distance exactly 2 from u.
    A C6 u-a-b-c-d-v exists iff there's a common neighbor of some a~u and d~v
    that isn't u,v,a,d, AND that common neighbor connects back properly.
    Actually simpler: for each a in N(u)\{v}, d in N(v)\{u,a},
    check if N(a) ∩ N(d) contains a vertex b≠c that forms the middle.
    """
    nu = adj[u]
    nv = adj[v]

    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        na = adj[a]
        
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            if d == a:
                continue

            nd = adj[d]
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            # We need b in N(a), c in N(d), b~c, b!=c, none in {u,v,a,d}
            # b in N(a)\{u,v,a,d}, c in N(d)\{u,v,a,d}, b~c, b!=c
            b_mask = na & ~forbidden
            if not b_mask:
                continue
            c_mask = nd & ~forbidden
            if not c_mask:
                continue

            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if adj[b] & (c_mask & ~(1 << b)):
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

def _greedy_build(adj, order, deadline):
    """
    Adds edges from order if possible. Returns count added.
    """
    count = 0
    for u, v in order:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count

def _k5_seed_adj():
    """
    Create a K5-free seed: partition vertices into 8 groups of 5,
    connect all within each group (K5), plus some cross-group edges.
    K5 is C6-free, and we can add cross edges carefully.
    """
    adj = [0] * N
    # 8 groups of 5 vertices: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39
    for g in range(8):
        base = g * 5
        for i in range(5):
            for j in range(i + 1, 5):
                _add_edge(adj, base + i, base + j)
    return adj

def _anneal_from(adj, deadline, rng):
    """
    Local search to maximize edges while maintaining C6-free property.
    Moves: 
    1. Add a random missing edge if valid (improvement)
    2. 2-opt swap: remove one edge, add two edges (net +1)
    3. Greedy fill after any change
    """
    current_count = _count_edges(adj)
    best_count = current_count
    best_adj = adj[:]
    
    all_edges = _make_all_edges()
    n_edges = len(all_edges)
    
    # Precompute edge index for quick lookup
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to add a random missing edge
            # Sample a few candidates
            found = False
            for _ in range(10):
                u, v = rng.choice(all_edges)
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                current_count += 1
                found = True
                break
            if not found:
                # Try a more systematic search for any addable edge
                for _ in range(50):
                    idx = rng.randint(0, n_edges - 1)
                    u, v = all_edges[idx]
                    if (adj[u] >> v) & 1:
                        continue
                    if _creates_c6(adj, u, v):
                        continue
                    _add_edge(adj, u, v)
                    current_count += 1
                    found = True
                    break
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
        elif move_type < 0.8:
            # 2-opt: remove one edge, try to add two
            # Build list of existing edges
            existing_edges = []
            for u in range(N):
                m = adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        existing_edges.append((u, v))
                    m ^= lsb
            
            if not existing_edges:
                continue
            
            u1, v1 = rng.choice(existing_edges)
            _remove_edge(adj, u1, v1)
            
            added = 0
            added_edges = []
            for _ in range(40):
                u2, v2 = rng.choice(all_edges)
                if (u2, v2) == (u1, v1):
                    continue
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added_edges.append((u2, v2))
                added += 1
                if added >= 2:
                    break
            
            if added >= 2:
                current_count += added - 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            elif added == 1:
                # Neutral swap, keep it
                pass
            else:
                # Revert: add back the removed edge, remove added edges
                for u2, v2 in added_edges:
                    _remove_edge(adj, u2, v2)
                _add_edge(adj, u1, v1)
        
        else:
            # Greedy fill: try to add any valid edge
            for _ in range(100):
                idx = rng.randint(0, n_edges - 1)
                u, v = all_edges[idx]
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                break
    
    return best_adj, best_count

def _greedy_fill(adj, deadline):
    """Repeatedly try to add any valid edge until none found or deadline."""
    all_edges = _make_all_edges()
    n_edges = len(all_edges)
    count = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(n_edges):
            if time.time() >= deadline:
                break
            u, v = all_edges[idx]
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            count += 1
            changed = True
    return count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    all_edges = _make_all_edges()
    
    best_adj = [0] * N
    best_count = 0
    
    # Phase 0: Initial greedy builds with different orderings
    for trial in range(5):
        if time.time() >= deadline - 5:
            break
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_build(adj, order, deadline)
        current_count = _count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Phase 1: K5-based seed
    if time.time() < deadline - 5:
        adj = _k5_seed_adj()
        _greedy_fill(adj, deadline)
        current_count = _count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Phase 2: Local search from best and other starting points
    remaining = deadline - 5
    chunks = 4
    chunk_time = remaining / chunks
    
    for i in range(chunks):
        if time.time() >= deadline - 2:
            break
        chunk_deadline = start_time + DEADLINE - 2 - (chunks - 1 - i) * chunk_time
        
        if i == 0:
            # Start from best
            adj = best_adj[:]
        elif i == 1:
            # K5 seed
            adj = _k5_seed_adj()
            _greedy_fill(adj, chunk_deadline)
        else:
            # Random greedy
            adj = [0] * N
            order = all_edges[:]
            rng.shuffle(order)
            _greedy_build(adj, order, chunk_deadline)
        
        adj, current_count = _anneal_from(adj, chunk_deadline, rng)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Early exit
        if best_count >= 105:
            break
    
    # Final greedy polish
    if time.time() < deadline - 1:
        _greedy_fill(best_adj, deadline - 0.5)
        best_count = _count_edges(best_adj)
    
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
