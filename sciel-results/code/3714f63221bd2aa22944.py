"""Combined program: fast C6 check + structured greedy + local search + swaps."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 145.0  # safety margin before 150s kill

def _check_c6_through_edge(adj, u, v, n=40):
    """Check if adding edge (u,v) creates a C6.
    A C6 using edge (u,v): u-v-x1-x2-x3-x4-u where all 6 distinct.
    Equivalent: need path of length 4 from u to v through 4 distinct intermediate vertices.
    u - a - b - c - d - v where a,d in N(u)\\{v}, b in N(a)\\{u,d}, c in N(d)\\{v,b}, b-c edge, all distinct.
    """
    # For each a in N(u) \ {v}:
    a_mask = adj[u] & ~(1 << v)
    while a_mask:
        a = (a_mask & (-a_mask)).bit_length() - 1
        a_mask &= a_mask - 1
        if a == u:
            continue
        # For each d in N(u) \ {v, a}:
        d_mask = adj[u] & ~(1 << v) & ~(1 << a)
        while d_mask:
            d = (d_mask & (-d_mask)).bit_length() - 1
            d_mask &= d_mask - 1
            if d == u or d == v:
                continue
            # For each b in N(a) \ {u, d, v}:
            b_mask = adj[a] & ~(1 << u) & ~(1 << d) & ~(1 << v)
            while b_mask:
                b = (b_mask & (-b_mask)).bit_length() - 1
                b_mask &= b_mask - 1
                if b == u or b == v or b == a or b == d:
                    continue
                # For each c in N(d) \ {u, v, b, a}:
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << b) & ~(1 << a)
                while c_mask:
                    c = (c_mask & (-c_mask)).bit_length() - 1
                    c_mask &= c_mask - 1
                    if c == u or c == v or c == a or c == b or c == d:
                        continue
                    # Check b-c edge
                    if adj[b] & (1 << c):
                        return False  # C6 found
    return True  # no C6 through this edge

def _is_c6_free(adj_bits, n=40):
    """Full check: verify no C6 exists in the graph."""
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                if not _check_c6_through_edge(adj_bits, i, j, n):
                    return False
    return True

def _count_edges(adj_bits, n=40):
    count = 0
    for i in range(n):
        count += bin(adj_bits[i]).count('1')
    return count // 2

def _get_edges(adj_bits, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=5.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _check_c6_through_edge(adj, u, v, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _local_search(adj, n=40, max_time=20.0):
    """Try to add non-edges one at a time using incremental check."""
    start_time = time.time()
    current = _count_edges(adj, n)
    
    while time.time() - start_time < max_time:
        improved = False
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > max_time:
                return adj, current
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _check_c6_through_edge(adj, u, v, n):
                current += 1
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if not improved:
            break
    return adj, current

def _swap_search(adj, n=40, max_time=15.0):
    """1-swap: remove one edge, add one non-edge. Accept if C6-free (same edge count)."""
    start_time = time.time()
    current = _count_edges(adj, n)
    
    while time.time() - start_time < max_time:
        improved = False
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(cur_edges)
        random.shuffle(non_edges)
        
        for ru, rv in cur_edges:
            if time.time() - start_time > max_time:
                break
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            
            for au, av in non_edges:
                if time.time() - start_time > max_time:
                    break
                if adj[au] & (1 << av):
                    continue
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                if _check_c6_through_edge(adj, au, av, n):
                    improved = True
                else:
                    adj[au] &= ~(1 << av)
                    adj[av] &= ~(1 << au)
            
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
        
        if not improved:
            break
    
    return adj, current

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def _structured_greedy(n=40, seed=None, max_time=10.0):
    """Greedy that prioritizes intra-cluster edges first, then inter-cluster."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    
    # Intra-cluster edges (8 clusters of 5)
    intra_edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                intra_edges.append((base+a, base+b))
    
    # Inter-cluster edges
    inter_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if i // 5 != j // 5]
    
    # Add intra-cluster edges first (always safe)
    for u, v in intra_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Shuffle inter-cluster edges and try to add them
    random.shuffle(inter_edges)
    for u, v in inter_edges:
        if time.time() - start_time > max_time:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _check_c6_through_edge(adj, u, v, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return adj, _count_edges(adj, n)

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from structured greedy (intra + inter)
    adj = _structured_greedy(n, seed=42, max_time=min(8.0, DEADLINE - 40))
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search to add more edges
    if time.time() - start_time < DEADLINE - 25:
        adj_ls, count_ls = _local_search(adj, n, max_time=min(12.0, DEADLINE - (time.time() - start_time) - 15))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random restarts with structured greedy
    for attempt in range(25):
        remaining = DEADLINE - (time.time() - start_time) - 8
        if remaining < 2:
            break
        mt = min(2.0, remaining / 4)
        adj_g, count_g = _structured_greedy(n, seed=200 + attempt, max_time=mt)
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
        # Quick local search on good results
        if count_g >= best_edges - 2 and time.time() - start_time < DEADLINE - 8:
            adj_l, count_l = _local_search(adj_g, n, max_time=min(3.0, DEADLINE - (time.time() - start_time) - 5))
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Phase 4: Swap search to escape local optima
    if time.time() - start_time < DEADLINE - 10:
        adj_sw, count_sw = _swap_search(best_adj, n, max_time=min(8.0, DEADLINE - (time.time() - start_time) - 5))
        if count_sw >= best_edges:
            best_adj = adj_sw
            best_edges = count_sw
        # Follow up with local search after swaps
        if time.time() - start_time < DEADLINE - 5:
            adj_f, count_f = _local_search(best_adj, n, max_time=min(4.0, DEADLINE - (time.time() - start_time) - 2))
            if count_f > best_edges:
                best_edges = count_f
                best_adj = adj_f[:]
    
    # Phase 5: Final local search
    if time.time() - start_time < DEADLINE - 3:
        adj_f, count_f = _local_search(best_adj, n, max_time=min(5.0, DEADLINE - (time.time() - start_time) - 1))
        if count_f > best_edges:
            best_edges = count_f
            best_adj = adj_f[:]
    
    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges(best_adj, n)
# EVOLVE-BLOCK-END
