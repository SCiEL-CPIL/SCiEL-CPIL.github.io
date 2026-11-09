import time
import random
import numpy as np

N = 40
DEADLINE = 115  # Safety margin

def _creates_c6_fast(adj, u, v):
    """
    Returns True if adding edge (u, v) creates a C6.
    This is equivalent to checking if there is a simple path of length 5 between u and v.
    Path: u - a - b - c - d - v
    """
    u_bit = 1 << u
    v_bit = 1 << v
    
    # Layer 1: Neighbors of u
    a_mask = adj[u] & ~v_bit
    if not a_mask:
        return False
        
    # Layer 2: Neighbors of a, excluding u and a itself (implicitly by set logic)
    # We need to track visited sets to ensure simplicity.
    # Let S1 = {u}
    # A1 = N(u) \ {v}
    # A2 = N(A1) \ (S1 U A1)
    # A3 = N(A2) \ (S1 U A1 U A2)
    # A4 = N(A3) \ (S1 U A1 U A2 U A3)
    # A5 = N(A4) \ (S1 U A1 U A2 U A3 U A4)
    # If v in A5, return True.
    
    S1 = u_bit
    A1 = a_mask
    visited = S1 | A1
    
    # Calculate A2
    A2 = 0
    m = A1
    while m:
        lsb = m & -m
        a = lsb.bit_length() - 1
        m ^= lsb
        A2 |= adj[a]
    A2 &= ~visited
    
    if not A2:
        return False
        
    visited |= A2
    
    # Calculate A3
    A3 = 0
    m = A2
    while m:
        lsb = m & -m
        b = lsb.bit_length() - 1
        m ^= lsb
        A3 |= adj[b]
    A3 &= ~visited
    
    if not A3:
        return False
        
    visited |= A3
    
    # Calculate A4
    A4 = 0
    m = A3
    while m:
        lsb = m & -m
        c = lsb.bit_length() - 1
        m ^= lsb
        A4 |= adj[c]
    A4 &= ~visited
    
    if not A4:
        return False
        
    visited |= A4
    
    # Calculate A5
    A5 = 0
    m = A4
    while m:
        lsb = m & -m
        d = lsb.bit_length() - 1
        m ^= lsb
        A5 |= adj[d]
    A5 &= ~visited
    
    return bool(A5 & v_bit)

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

def _greedy_fill(adj, all_edges, rng):
    """
    Greedily add edges in random order if they don't create a C6.
    """
    count = 0
    for _pass in range(5): # Multiple passes to catch edges missed in previous random order
        order = all_edges[:]
        rng.shuffle(order)
        added_in_pass = 0
        for u, v in order:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6_fast(adj, u, v):
                continue
            _add_edge(adj, u, v)
            count += 1
            added_in_pass += 1
        if added_in_pass == 0:
            break
    return count

def _local_search(adj, max_iter, deadline, rng, all_edges):
    """
    Local search: Try to add edges, then try edge swaps (remove 1, add 2) to increase density.
    """
    best_count = _count_edges(adj)
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False
        
        # Phase 1: Try to add any missing edge
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6_fast(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        
        # Phase 2: Edge Swaps
        # Try to remove an edge e1 and add two edges e2, e3
        # This allows escaping local optima where no single edge can be added.
        for _swap_iter in range(5):
            if time.time() >= deadline:
                break
            
            # Pick a random edge to remove
            present_edges = []
            for i in range(N):
                m = adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if j > i:
                        present_edges.append((i, j))
                    m ^= lsb
            
            if not present_edges:
                break
                
            u1, v1 = rng.choice(present_edges)
            _remove_edge(adj, u1, v1)
            
            added = 0
            # Try to add 2 random edges
            for _t in range(20):
                if time.time() >= deadline:
                    break
                u2, v2 = rng.choice(all_edges)
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6_fast(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added += 1
                if added == 2:
                    break
            
            if added < 1:
                # Rollback if we couldn't add anything
                _add_edge(adj, u1, v1)
            elif added == 1:
                # Net 0 change, but might help future moves? 
                # Usually we want net +1. If we added 1, we are at same count.
                # If we added 2, we are at +1.
                # If added == 1, we removed 1, added 1. Count is same.
                # We accept it if it changes the structure (random walk).
                # But to maximize, we prefer +1.
                # Let's only accept if added == 2 for now, or accept added==1 with some probability?
                # For simplicity, accept if added >= 1 (structure change) or added == 2 (improvement).
                # Actually, if added==1, we are at same edge count. This is a sideways move.
                # If added==2, we improved.
                improved = (added == 2)
                if added == 1:
                    # Sideways move, accept to escape local minima
                    pass 
            else: # added == 2
                improved = True
        
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            
        if not improved:
            # Try a few more iterations before giving up?
            # If no improvement in add phase and no improvement in swap phase (net +1), stop.
            # But sideways moves might help.
            # Let's break if no net improvement.
            break
            
    return adj, best_count

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

    # Start with a random sparse graph or empty
    # Empty graph is safe
    adj = [0] * N
    
    # Initial Greedy
    _greedy_fill(adj, all_edges, rng)
    refresh_best(adj)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturb
        perturbed = best_adj[:]
        # Remove a few random edges
        k = rng.randint(2, 5)
        for _ in range(k):
            present_edges = []
            for i in range(N):
                m = perturbed[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if j > i:
                        present_edges.append((i, j))
                    m ^= lsb
            if not present_edges:
                break
            u, v = rng.choice(present_edges)
            _remove_edge(perturbed, u, v)
            
        # Greedy Fill
        _greedy_fill(perturbed, all_edges, rng)
        
        # Local Search
        perturbed, c = _local_search(perturbed, max_iter=50, deadline=deadline, rng=rng, all_edges=all_edges)
        refresh_best(perturbed)
        
        # If we hit the theoretical max (105), we can stop early if confident
        if best_count >= 105:
            # Verify if 105 is indeed the max. 
            # If we find 105, it's likely optimal.
            # We can continue to try to find 106, but it's unlikely.
            pass

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
