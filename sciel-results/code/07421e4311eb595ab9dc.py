"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _has_triangle_circulant_fast(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle.
    Triangle exists if there exist x,y in S such that x+y mod n is in S or -S.
    """
    S_set = frozenset(S)
    neg_S = frozenset({(-x) % n for x in S})
    for x in S:
        for y in S:
            z = (x + y) % n
            if z in S_set or z in neg_S:
                return True
    return False

def _max_independent_set_size(n, adj):
    """
    Compute alpha of a graph given its adjacency list.
    Uses branch and bound with bitmasks.
    """
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size, depth):
        # Upper bound
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidates for better pruning
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(candidates & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1, depth + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size, depth + 1)
    
    dfs(full_mask, 0, 0)
    return best[0]

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant_fast(n, S):
        return False, 0
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i:
                adj[i].append(j)
    
    # Run exact B&B
    alpha = _max_independent_set_size(n, adj)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _build_edges(n, S):
    """Build edge list from circulant connection set."""
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        
        # Strategy 1: Systematic enumeration of symmetric S sets
        # For each possible size k, try random subsets of {1, ..., floor(n/2)}
        half = n // 2
        available = list(range(1, half + 1))
        
        # Try different target sizes, from larger to smaller (higher degree = smaller alpha)
        max_k = half  # Maximum possible symmetric S size
        min_k = 3
        
        # Try sizes from larger to smaller
        for k in range(min(max_k, 15), min_k - 1, -1):
            if time.time() > deadline - 10:
                    break
            
            # Number of random attempts for this size
            num_attempts = 100 if n <= 35 else 50
            
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                
                # Random subset of available of size k
                if k > len(available):
                    continue
                    
                S_half = set(random.sample(available, k))
                # Make symmetric: S = S_half ∪ {-x mod n for x in S_half}
                S = set()
                for x in S_half:
                    S.add(x)
                    S.add((-x) % n)
                
                # Remove duplicates (when x = -x mod n, i.e., x = n/2 for even n)
                S = set(S)
                
                # Quick triangle check
                if _has_triangle_circulant_fast(n, S):
                    continue
                
                # Verify alpha
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Fallback: Known small graphs
    if best_n < 19:
        # Use a known triangle-free graph with alpha < 10
        # The 26-vertex graph from the previous version
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
