"""Combined program: fast incremental C6 check + greedy + local search + 2-opt swaps."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 145.0  # safety margin before 150s kill

def _check_c6_through_edge(adj, u, v, n=40):
    """Check if adding edge (u,v) creates a C6. Only checks paths through (u,v).
    A C6 using edge (u,v) as one side: u-v-a-b-c-d-u or u-v-a-b-c-d-v... 
    Actually: the C6 has u and v as adjacent vertices. The other 4 vertices form
    a path of length 4 from u to v avoiding v and u respectively.
    Path: u - x1 - x2 - x3 - x4 - v where all distinct.
    Equivalently: u - x4 - x3 - x2 - x1 - v.
    So we need: x1 in N(v)\\{u}, x4 in N(u)\\{v}, x2 in N(x1)\\{v,x4},
    x3 in N(x4)\\{u,x2}, and x2-x3 is an edge, all 6 distinct.
    """
    # x1 in N(v) \ {u}
    x1_mask = adj[v] & ~(1 << u)
    while x1_mask:
        x1 = (x1_mask & (-x1_mask)).bit_length() - 1
        x1_mask &= x1_mask - 1
        # x4 in N(u) \ {v, x1}
        x4_mask = adj[u] & ~(1 << v) & ~(1 << x1)
        while x4_mask:
            x4 = (x4_mask & (-x4_mask)).bit_length() - 1
            x4_mask &= x4_mask - 1
            # x2 in N(x1) \ {v, x4}
            x2_mask = adj[x1] & ~(1 << v) & ~(1 << x4)
            while x2_mask:
                x2 = (x2_mask & (-x2_mask)).bit_length() - 1
                x2_mask &= x2_mask - 1
                if x2 == u:
                    continue
                # x3 in N(x4) \ {u, x2}
                x3_mask = adj[x4] & ~(1 << u) & ~(1 << x2)
                while x3_mask:
                    x3 = (x3_mask & (-x3_mask)).bit_length() - 1
                    x3_mask &= x3_mask - 1
                    if x3 == v or x3 == x1:
                        continue
                    # Check x2 - x3 edge
                    if adj[x2] & (1 << x3):
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

def _two_opt_swap(adj, n=40, max_time=15.0):
    """2-opt: remove two edges, add two non-edges. Try to find a net +0 or +1 swap.
    Actually just do 1-swap (remove one, add one) but with incremental check for the add."""
    start_time = time.time()
    current = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current
    
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
                    # Keep the swap (same edge count) as it may help future additions
                    improved = True
                    if current > best_count:
                        best_count = current
                        best_adj = adj[:]
                else:
                    adj[au] &= ~(1 << av)
                    adj[av] &= ~(1 << au)
            
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
        
        if not improved:
            break
    
    if best_count > current:
        return best_adj, best_count
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

def _try_structures(n=40):
    """Try some known good structures and return the best."""
    candidates = []
    
    # 8x K5
    adj1 = _known_good(n)
    candidates.append(adj1)
    
    # 4x K7 + 4x K4 (4*21 + 4*6 = 84+24 = 108 edges? No, K7 has 21 edges, K4 has 6)
    # 4*21 + 4*6 = 108... but we only have 40 vertices: 4*7 + 4*4 = 28+16 = 44 > 40. Too many.
    # Try 3x K7 + 5x K5 = 3*21 + 5*10 = 63+50 = 113? But 3*7+5*5=21+25=46 > 40.
    # 2x K7 + 6x K5 = 2*21 + 6*10 = 42+60 = 102. Vertices: 14+30=44 > 40.
    # 1x K7 + 7x K5 = 21+70 = 91. Vertices: 7+35=42 > 40.
    # Let's just use 8x K5 for now.
    
    # Try a different partition: 5x K8 = 5*28 = 140 edges. Vertices: 5*8=40. 
    # But K8 contains C6! So that won't work.
    
    # What about blow-ups? Take a C6-free base graph and blow up vertices.
    # The complete bipartite K_{m,n} is C6-free iff min(m,n) <= 2 or max(m,n) <= 2... 
    # Actually K_{m,n} has C4 if m,n >= 2, and C6 if m,n >= 3.
    # So K_{2,n} is C6-free. 
    # Disjoint union of K_{2,n_i} with sum of (2+n_i) = 40.
    # Edges = sum 2*n_i.
    # Maximize sum 2*n_i subject to sum (2+n_i) = 40, i.e., 2k + sum n_i = 40 where k is number of components.
    # sum n_i = 40 - 2k, edges = 2(40-2k) = 80-4k. Best when k is small.
    # k=1: n=38, edges=76. k=2: n sum=36, edges=72. Worse.
    # So K_{2,38} gives 76 edges. Worse than 100.
    
    return candidates

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search from 8xK5
    if time.time() - start_time < DEADLINE - 30:
        adj_ls, count_ls = _local_search(adj, n, max_time=min(15.0, DEADLINE - (time.time() - start_time) - 15))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random greedy restarts
    for attempt in range(40):
        remaining = DEADLINE - (time.time() - start_time) - 10
        if remaining < 2:
            break
        mt = min(1.5, remaining / 3)
        adj_g, count_g = _random_greedy(n, seed=100 + attempt, max_time=mt)
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
        # Quick local search on good greedy results
        if count_g >= best_edges - 3 and time.time() - start_time < DEADLINE - 10:
            adj_l, count_l = _local_search(adj_g, n, max_time=min(3.0, DEADLINE - (time.time() - start_time) - 5))
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Phase 4: 1-swap search on best (to escape local optima)
    if time.time() - start_time < DEADLINE - 8:
        adj_sw, count_sw = _two_opt_swap(best_adj, n, max_time=min(8.0, DEADLINE - (time.time() - start_time) - 3))
        if count_sw >= best_edges:
            best_adj = adj_sw
            best_edges = count_sw
    
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
