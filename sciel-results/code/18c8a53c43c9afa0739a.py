"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score_bitmask(adj_bits, n, edge_count):
    """
    Compute max triangle degree sum using bitmasks.
    adj_bits: list of n integers, each a bitmask of neighbors.
    Returns max degree sum over all triangles.
    """
    deg = [bin(adj_bits[i]).count('1') for i in range(n)]
    max_sum = 0
    for i in range(n):
        ai = adj_bits[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                # common neighbors of i and j that are > j
                common = ai & adj_bits[j] & (~((1 << (j + 1)) - 1))
                while common:
                    lsb = common & (-common)
                    k = lsb.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb
    return max_sum

def _build_adj_bits(edges, n):
    """Build bitmask adjacency representation from edge list."""
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits

def _sa_search(n, edges, deadline, max_iter=200000):
    """
    Simulated annealing with bitmask scoring.
    Edge swaps: remove one edge, add another (maintaining edge count).
    """
    start_time = time.time()
    random.seed(42)
    
    current_edges = list(edges)
    adj_bits = _build_adj_bits(current_edges, n)
    current_score = _score_bitmask(adj_bits, n, len(current_edges))
    best_edges = list(current_edges)
    best_score = current_score
    best_adj_bits = list(adj_bits)
    
    # Precompute all possible edge pairs as (u,v) with u<v
    all_edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
    
    # Temperature schedule
    T0 = 3.0
    T_end = 0.01
    n_iter = min(max_iter, 200000)
    
    for it in range(n_iter):
        if time.time() - start_time > deadline - 5:
            break
        
        T = T0 * (T_end / T0) ** (it / n_iter)
        
        # Pick random edge to remove
        idx = random.randrange(len(current_edges))
        u1, v1 = current_edges[idx]
        e1 = (min(u1, v1), max(u1, v1))
        
        # Pick random edge to add
        attempts = 0
        while attempts < 200:
            u2, v2 = all_edge_pairs[random.randrange(len(all_edge_pairs))]
            e2 = (min(u2, v2), max(u2, v2))
            if e2 not in edge_set and e2 != e1:
                break
            attempts += 1
        else:
            continue
        
        # Apply swap to bitmask
        adj_bits[u1] ^= (1 << v1)
        adj_bits[v1] ^= (1 << u1)
        adj_bits[u2] |= (1 << v2)
        adj_bits[v2] |= (1 << u2)
        
        new_score = _score_bitmask(adj_bits, n, len(current_edges))
        
        delta = new_score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            # Accept
            current_edges[idx] = [u2, v2]
            edge_set.remove(e1)
            edge_set.add(e2)
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = list(current_edges)
                best_adj_bits = list(adj_bits)
        else:
            # Revert
            adj_bits[u2] ^= (1 << v2)
            adj_bits[v2] ^= (1 << u2)
            adj_bits[u1] |= (1 << v1)
            adj_bits[v1] |= (1 << u1)
    
    return best_edges, best_score

def _local_polish(n, edges, deadline):
    """
    Exhaustive single-swap local search with bitmask scoring.
    Repeats until no improvement found or deadline.
    """
    start_time = time.time()
    current_edges = list(edges)
    adj_bits = _build_adj_bits(current_edges, n)
    current_score = _score_bitmask(adj_bits, n, len(current_edges))
    best_edges = list(current_edges)
    best_score = current_score
    
    all_edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
    
    improved = True
    while improved and time.time() - start_time < deadline - 3:
        improved = False
        for i in range(len(current_edges)):
            if time.time() - start_time > deadline - 3:
                break
            u1, v1 = current_edges[i]
            e1 = (min(u1, v1), max(u1, v1))
            
            for u2, v2 in all_edge_pairs:
                e2 = (min(u2, v2), max(u2, v2))
                if e2 in edge_set or e2 == e1:
                    continue
                
                # Try swap
                adj_bits[u1] ^= (1 << v1)
                adj_bits[v1] ^= (1 << u1)
                adj_bits[u2] |= (1 << v2)
                adj_bits[v2] |= (1 << u2)
                
                new_score = _score_bitmask(adj_bits, n, len(current_edges))
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = list(current_edges)
                    current_edges[i] = [u2, v2]
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    adj_bits[u2] ^= (1 << v2)
                    adj_bits[v2] ^= (1 << u2)
                    adj_bits[u1] |= (1 << v1)
                    adj_bits[v1] |= (1 << u1)
    
    return best_edges, best_score

def _multi_start_sa(n, deadline):
    """
    Run SA from multiple different initial configurations.
    """
    start_time = time.time()
    best_overall = None
    best_score_overall = float('inf')
    
    # Multiple initial configurations
    configs = []
    
    # Config 1: K_{6,10} + matching in part of size 10
    e1 = [[u, v] for u in range(6) for v in range(6, 16)]
    e1 += [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    configs.append(e1)
    
    # Config 2: K_{8,8} + one edge
    e2 = [[u, v] for u in range(8) for v in range(8, 16)]
    e2.append([0, 1])
    configs.append(e2)
    
    # Config 3: K_{7,9} + 2 edges in part of size 9
    e3 = [[u, v] for u in range(7) for v in range(7, 16)]
    e3 += [[7, 8], [9, 10]]
    configs.append(e3)
    
    # Config 4: K_{5,11} + 10 edges in part of size 11
    e4 = [[u, v] for u in range(5) for v in range(5, 16)]
    # Add 10 edges among vertices 5..15
    inner = list(range(5, 16))
    random.seed(123)
    all_inner = [(inner[a], inner[b]) for a in range(len(inner)) for b in range(a+1, len(inner))]
    random.shuffle(all_inner)
    e4 += [list(x) for x in all_inner[:10]]
    configs.append(e4)
    
    # Config 5: Random graph with 65 edges
    random.seed(777)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    e5 = [list(x) for x in all_pairs[:65]]
    configs.append(e5)
    
    # Time budget per config
    time_per_config = (deadline - 10 - start_time) / len(configs)
    
    for ci, cfg in enumerate(configs):
        cfg_deadline = start_time + time_per_config * (ci + 1)
        # Ensure at least some time
        if cfg_deadline < start_time + 2:
            cfg_deadline = start_time + 2
        
        # Run SA
        edges, score = _sa_search(n, cfg, cfg_deadline, max_iter=100000)
        
        if score < best_score_overall:
            best_score_overall = score
            best_overall = list(edges)
    
    return best_overall, best_score_overall

def construct():
    n = 16
    deadline = time.time() + 140  # 140s budget, 10s safety margin
    
    # Phase 1: Multi-start simulated annealing
    edges, score = _multi_start_sa(n, deadline)
    
    # Phase 2: Local polish
    edges, score = _local_polish(n, edges, deadline)
    
    # Phase 3: Another SA round starting from polished result (if time)
    remaining = deadline - 3 - time.time()
    if remaining > 5:
        edges2, score2 = _sa_search(n, edges, deadline - 3, max_iter=50000)
        if score2 < score:
            edges, score = edges2, score2
        edges, score = _local_polish(n, edges, deadline)
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
