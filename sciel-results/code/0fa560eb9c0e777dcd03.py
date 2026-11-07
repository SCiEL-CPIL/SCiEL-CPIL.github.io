"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from functools import lru_cache

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj, adj_mask

def _score_subset(vertices, adj_mask):
    """Score: number of distinct degrees in induced subgraph. Uses bitmask for speed."""
    mask = 0
    for v in vertices:
        mask |= (1 << v)
    degs = set()
    for v in vertices:
        d = (adj_mask[v] & mask).bit_count()
        degs.add(d)
    return len(degs)

def _greedy_construct(adj, adj_mask, n, min_vertices=8, target_size=None):
    """Greedy construction: add vertices that maximize distinct degree count."""
    if target_size is None:
        target_size = min_vertices
    
    current = [0]
    current_mask = 1
    
    for _ in range(target_size - 1):
        best_v = None
        best_score = -1
        for v in range(n):
            if (current_mask >> v) & 1:
                continue
            new_mask = current_mask | (1 << v)
            degs = set()
            for x in range(n):
                if (new_mask >> x) & 1:
                    d = (adj_mask[x] & new_mask).bit_count()
                    degs.add(d)
            score = len(degs)
            if score > best_score:
                best_score = score
                best_v = v
        if best_v is not None:
            current.append(best_v)
            current_mask |= (1 << best_v)
    
    return current

def _local_search_1swap(current, adj_mask, n, deadline):
    """Local search with 1-swap moves using bitmask."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        best_swap = None
        best_swap_score = _score_subset(current, adj_mask)
        
        for out_v in current:
            out_bit = 1 << out_v
            for in_v in range(n):
                in_bit = 1 << in_v
                if (current_mask & in_bit):
                    continue
                new_mask = (current_mask & ~out_bit) | in_bit
                degs = set()
                for x in range(n):
                    if (new_mask >> x) & 1:
                        d = (adj_mask[x] & new_mask).bit_count()
                        degs.add(d)
                score = len(degs)
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (out_v, in_v)
        
        if best_swap is not None:
            out_v, in_v = best_swap
            current = [v for v in current if v != out_v]
            current.append(in_v)
            improved = True
    
    return current

def _local_search_2swap(current, adj_mask, n, deadline):
    """Local search with 2-swap moves using bitmask."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        best_swap = None
        best_swap_score = _score_subset(current, adj_mask)
        
        # 1-swap
        for out_v in current:
            out_bit = 1 << out_v
            for in_v in range(n):
                in_bit = 1 << in_v
                if (current_mask & in_bit):
                    continue
                new_mask = (current_mask & ~out_bit) | in_bit
                degs = set()
                for x in range(n):
                    if (new_mask >> x) & 1:
                        d = (adj_mask[x] & new_mask).bit_count()
                        degs.add(d)
                score = len(degs)
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (out_v, in_v, -1, -1)
        
        # 2-swap
        for out1_idx, out1 in enumerate(current):
            for out2_idx in range(out1_idx + 1, len(current)):
                out2 = current[out2_idx]
                out_bits = (1 << out1) | (1 << out2)
                base_mask = current_mask & ~out_bits
                for in1 in range(n):
                    if (current_mask >> in1) & 1:
                        continue
                    for in2 in range(in1 + 1, n):
                        if (current_mask >> in2) & 1:
                            continue
                        new_mask = base_mask | (1 << in1) | (1 << in2)
                        degs = set()
                        for x in range(n):
                            if (new_mask >> x) & 1:
                                d = (adj_mask[x] & new_mask).bit_count()
                                degs.add(d)
                        score = len(degs)
                        if score > best_swap_score:
                            best_swap_score = score
                            best_swap = (out1, out2, in1, in2)
        
        if best_swap is not None:
            out1, out2, in1, in2 = best_swap
            if in1 >= 0:
                current = [v for v in current if v != out1]
                current.append(in1)
            else:
                current = [v for v in current if v != out1 and v != out2]
                current.append(in1)
                current.append(in2)
            improved = True
    
    return current

def _exhaustive_best_size_k(adj_mask, n, k, deadline=None):
    """Exhaustive search over all C(n,k) subsets to find best distinct degree count."""
    best_vertices = None
    best_score = -1
    
    for subset in itertools.combinations(range(n), k):
        if deadline is not None and time.time() > deadline:
            break
        score = _score_subset(subset, adj_mask)
        if score > best_score:
            best_score = score
            best_vertices = list(subset)
    
    return best_vertices, best_score

def _multi_start_local_search(adj, adj_mask, n, min_vertices, deadline, n_starts=20, seed=42):
    """Multi-start local search with diverse initializations."""
    rng = random.Random(seed)
    all_vertices = list(range(n))
    best_overall = None
    best_overall_score = -1
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Diverse initialization
        if start_idx == 0:
            # Greedy start
            current = _greedy_construct(adj, adj_mask, n, min_vertices)
        elif start_idx == 1:
            # Random start
            current = rng.sample(all_vertices, min_vertices)
        elif start_idx == 2:
            # Random start with size min_vertices+1
            current = rng.sample(all_vertices, min(min_vertices+1, n))
        elif start_idx == 3:
            # Random start with size min_vertices+2
            current = rng.sample(all_vertices, min(min_vertices+2, n))
        elif start_idx == 4:
            # Random start with size min_vertices+3
            current = rng.sample(all_vertices, min(min_vertices+3, n))
        elif start_idx == 5:
            # Random start with size min_vertices+4
            current = rng.sample(all_vertices, min(min_vertices+4, n))
        else:
            # Random start with varying sizes
            size = min_vertices + (start_idx % 5)
            size = min(size, n)
            current = rng.sample(all_vertices, size)
        
        # Local search with 1-swap
        current = _local_search_1swap(current, adj_mask, n, deadline)
        score = _score_subset(current, adj_mask)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = list(current)
        
        # Try 2-swap if time permits
        if time.time() < deadline - 3:
            current2 = _local_search_2swap(current, adj_mask, n, deadline - 3)
            score2 = _score_subset(current2, adj_mask)
            if score2 > best_overall_score:
                best_overall_score = score2
                best_overall = list(current2)
    
    return best_overall, best_overall_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    adj, adj_mask = _build_adj(n, edges)
    
    # Start with greedy construction
    current = _greedy_construct(adj, adj_mask, n, min_vertices)
    current_score = _score_subset(current, adj_mask)
    
    # Try exhaustive search for sizes 8 through 12
    # C(17,8)=24310, C(17,9)=24310, C(17,10)=19448, C(17,11)=12376, C(17,12)=6188
    for k in [8, 9, 10, 11, 12]:
        if time.time() < deadline - 5:
            best_k, score_k = _exhaustive_best_size_k(adj_mask, n, k, deadline - 5)
            if best_k is not None and score_k > current_score:
                current = best_k
                current_score = score_k
    
    # Multi-start local search with diverse initializations
    if time.time() < deadline - 15:
        ms_best, ms_score = _multi_start_local_search(adj, adj_mask, n, min_vertices, deadline - 15, n_starts=50, seed=42)
        if ms_best is not None and ms_score > current_score:
            current = ms_best
            current_score = ms_score
    
    # Second multi-start with different seed
    if time.time() < deadline - 10:
        ms_best2, ms_score2 = _multi_start_local_search(adj, adj_mask, n, min_vertices, deadline - 10, n_starts=30, seed=123)
        if ms_best2 is not None and ms_score2 > current_score:
            current = ms_best2
            current_score = ms_score2
    
    # Final local search polish with 1-swap
    if time.time() < deadline:
        current = _local_search_1swap(current, adj_mask, n, deadline)
        current_score = _score_subset(current, adj_mask)
    
    # Try 2-swap polish if time permits
    if time.time() < deadline - 3:
        current2 = _local_search_2swap(current, adj_mask, n, deadline - 3)
        if _score_subset(current2, adj_mask) > current_score:
            current = current2
    
    # Ensure we have at least min_vertices
    if len(current) < min_vertices:
        current_set = set(current)
        for v in range(n):
            if v not in current_set and len(current) < min_vertices:
                current.append(v)
                current_set.add(v)
    
    # Return as list of Python ints
    return [int(v) for v in current]

# EVOLVE-BLOCK-END
