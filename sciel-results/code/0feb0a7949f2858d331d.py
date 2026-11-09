import time
import random
import sys

N = 40
DEADLINE = 115  # Safety margin

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

            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask (b != c)
            # We iterate over the smaller mask to reduce checks
            if b_mask.bit_count() < c_mask.bit_count():
                # Check if any neighbor of b is in c_mask
                bm = b_mask
                while bm:
                    lsb_b = bm & -bm
                    b = lsb_b.bit_length() - 1
                    bm ^= lsb_b
                    # c must be in c_mask and not b
                    if adj[b] & (c_mask & ~(1 << b)):
                        return True
            else:
                # Check if any neighbor of c is in b_mask
                cm = c_mask
                while cm:
                    lsb_c = cm & -cm
                    c = lsb_c.bit_length() - 1
                    cm ^= lsb_c
                    # b must be in b_mask and not c
                    if adj[c] & (b_mask & ~(1 << c)):
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

def _greedy_build(adj, order, deadline):
    """
    Adds edges from order if possible.
    """
    for u, v in order:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _local_search(adj, deadline, rng):
    """
    Local search to maximize edges.
    1. Try to add any valid edge.
    2. If stuck, try 2-opt swaps: remove one edge, add two new ones.
    3. If stuck, try 3-opt swaps: remove two edges, add three new ones.
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    
    # Precompute current edges list for quick access? 
    # We'll just scan all_edges each time, it's 780 edges, fast enough.
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Simple additions
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
            best_count += 1
            
        if improved:
            continue

        # Pass 2: 2-opt swaps (remove 1, add 2)
        current_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
        if not current_edges:
            break
        rng.shuffle(current_edges)
        
        for u1, v1 in current_edges:
            if time.time() >= deadline:
                break
            
            _remove_edge(adj, u1, v1)
            
            # Try to add 2 edges
            added = []
            # Shuffle all_edges to try random additions
            shuffled_edges = all_edges[:]
            rng.shuffle(shuffled_edges)
            for u2, v2 in shuffled_edges:
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added.append((u2, v2))
                if len(added) == 2:
                    break
            
            if len(added) == 2:
                best_count += 1
                improved = True
                # After 2-opt success, we might be able to add more, so break to restart loop
                break
            else:
                # Revert
                _add_edge(adj, u1, v1)
                # If we added 1, we are neutral, but we didn't improve. 
                # We should probably revert to keep the search focused on improvements.
                if len(added) == 1:
                    _remove_edge(adj, added[0][0], added[0][1])

        if improved:
            continue

        # Pass 3: 3-opt swaps (remove 2, add 3)
        # This is more expensive. Only do if we have time.
        if time.time() > deadline - 5:
            break
            
        current_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
        if len(current_edges) < 2:
            break
            
        # Pick two random edges to remove
        e1, e2 = rng.sample(current_edges, 2)
        
        _remove_edge(adj, e1[0], e1[1])
        _remove_edge(adj, e2[0], e2[1])
        
        added = []
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        for u3, v3 in shuffled_edges:
            if (adj[u3] >> v3) & 1:
                continue
            if _creates_c6(adj, u3, v3):
                continue
            _add_edge(adj, u3, v3)
            added.append((u3, v3))
            if len(added) == 3:
                break
        
        if len(added) == 3:
            best_count += 1
            improved = True
            break
        else:
            # Revert
            _add_edge(adj, e1[0], e1[1])
            _add_edge(adj, e2[0], e2[1])
            for u, v in added:
                _remove_edge(adj, u, v)

        if not improved:
            # Stuck in local optimum. Try a random kick.
            # Remove a random edge and do a quick greedy fill.
            if current_edges:
                u1, v1 = rng.choice(current_edges)
                _remove_edge(adj, u1, v1)
                
                shuffled_edges = all_edges[:]
                rng.shuffle(shuffled_edges)
                for u, v in shuffled_edges:
                    if (adj[u] >> v) & 1:
                        continue
                    if _creates_c6(adj, u, v):
                        continue
                    _add_edge(adj, u, v)
                    break
                
                new_count = _count_edges(adj)
                if new_count > best_count:
                    best_count = new_count
                    improved = True
                else:
                    # Revert kick
                    # We added one edge and removed one. If count didn't improve, we are back to same count but different structure.
                    # We can keep this structure as it might allow future improvements.
                    # So we don't revert, we just continue.
                    pass
            else:
                break

    return adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    all_edges = _make_all_edges()
    
    best_adj = [0] * N
    best_count = 0
    
    # Run multiple iterations
    while time.time() < deadline - 2:
        adj = [0] * N
        
        # Randomized greedy build
        order = all_edges[:]
        rng.shuffle(order)
        
        _greedy_build(adj, order, deadline)
        
        current_count = _count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Local search
        if time.time() < deadline - 5:
            adj, current_count = _local_search(adj, deadline, rng)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
                
        # Early exit if we hit the known max (105) or higher
        if best_count >= 105:
            break

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
