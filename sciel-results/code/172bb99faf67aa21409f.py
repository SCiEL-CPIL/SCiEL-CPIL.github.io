"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj_bitmask():
    n = P['n']
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _score_bitmask(mask, adj_mask, min_size=8):
    """Fast score: count distinct degrees in induced subgraph using bitmasks."""
    if bin(mask).count('1') < min_size:
        return -1
    distinct = set()
    for v in range(P['n']):
        if mask & (1 << v):
            deg = bin(adj_mask[v] & mask).count('1')
            distinct.add(deg)
    return len(distinct)

def _local_search(mask, adj_mask, deadline, best_score, best_subset):
    """Aggressive local search: add, remove, swap vertices."""
    current = mask
    current_score = _score_bitmask(current, adj_mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try adding each vertex
        for v in range(P['n']):
            if not (current & (1 << v)):
                new_mask = current | (1 << v)
                score = _score_bitmask(new_mask, adj_mask)
                if score > current_score:
                    current = new_mask
                    current_score = score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_subset = [v for v in range(P['n']) if current & (1 << v)]
        
        # Try removing each vertex (if size > 8)
        if bin(current).count('1') > 8:
            for v in range(P['n']):
                if current & (1 << v):
                    new_mask = current & ~(1 << v)
                    score = _score_bitmask(new_mask, adj_mask)
                    if score > current_score:
                        current = new_mask
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_subset = [v for v in range(P['n']) if current & (1 << v)]
        
        # Try swaps
        for v_in in range(P['n']):
            if not (current & (1 << v_in)):
                continue
            for v_out in range(P['n']):
                if current & (1 << v_out):
                    continue
                new_mask = (current & ~(1 << v_in)) | (1 << v_out)
                score = _score_bitmask(new_mask, adj_mask)
                if score > current_score:
                    current = new_mask
                    current_score = score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_subset = [v for v in range(P['n']) if current & (1 << v)]
        
        if not improved:
            break
    
    return current, current_score, best_score, best_subset

def _random_restart(mask, adj_mask, deadline, best_score, best_subset, n_random=20):
    """Random restarts with local search."""
    for _ in range(n_random):
        if time.time() > deadline:
            break
        # Create random subset of size 8-12
        size = random.randint(8, 12)
        vertices = random.sample(range(P['n']), size)
        rand_mask = 0
        for v in vertices:
            rand_mask |= (1 << v)
        
        # Local search from random start
        _, _, best_score, best_subset = _local_search(rand_mask, adj_mask, deadline, best_score, best_subset)
    
    return best_score, best_subset

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    n = P['n']
    adj_mask = _build_adj_bitmask()
    
    best_score = 0
    best_subset = list(range(8))  # fallback
    
    # Strategy 1: Exhaustive search for sizes 8-10 (feasible)
    for size in range(8, 11):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            score = _score_bitmask(mask, adj_mask)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                if best_score == size:
                    # Can't do better than size distinct degrees for size vertices
                    return [int(v) for v in best_subset]
    
    # Strategy 2: Local search from best found
    if time.time() < deadline:
        start_mask = 0
        for v in best_subset:
            start_mask |= (1 << v)
        _, _, best_score, best_subset = _local_search(start_mask, adj_mask, deadline, best_score, best_subset)
    
    # Strategy 3: Random restarts
    if time.time() < deadline:
        best_score, best_subset = _random_restart(0, adj_mask, deadline, best_score, best_subset, n_random=50)
    
    # Ensure we return at least 8 vertices
    if len(best_subset) < 8:
        best_subset = list(range(8))
    
    # Convert to ensure Python int types
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
