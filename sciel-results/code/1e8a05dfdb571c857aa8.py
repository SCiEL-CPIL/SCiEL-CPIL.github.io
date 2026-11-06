"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _has_triangle_circulant(n, S):
    """Check if circulant graph C_n(S) has a triangle.
    Triangle: vertices 0, a, b with a-b, b, a all in S (or equivalent).
    Equivalently: exist a,b in S with a!=b, a!=0, b!=0, and a+b mod n in S,
    with a+b not equal to a or b (i.e., a,b,a+b distinct and nonzero).
    """
    S_set = set(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            s = (a + b) % n
            if s in S_set and s != a and s != b and s != 0:
                return True
    return False

def _alpha_circulant(n, S):
    """Calculate independence number of C_n(S) using branch and bound with bitmasks."""
    # Build adjacency masks
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    # Precompute: for each vertex, the set of vertices NOT adjacent to it (excluding itself)
    # non_adj[v] = all_mask ^ adj_mask[v] ^ (1 << v)
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = all_mask ^ adj_mask[v] ^ (1 << v)
    
    best = 0
    # Time limit for this computation
    start_t = time.time()
    time_limit = 3.0  # seconds per alpha computation
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size):
        nonlocal best
        # Upper bound: current_size + remaining candidates
        rem = popcount(candidates)
        if current_size + rem <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        
        # Check time
        if time.time() - start_t > time_limit:
            return
        
        # Pick vertex with max degree in candidate subgraph for better pruning
        # Try a few heuristic choices
        temp = candidates
        max_deg = -1
        best_v_bit = 0
        # Sample up to 8 vertices to find high-degree one
        count = 0
        while temp and count < 8:
            t_bit = temp & -temp
            t = t_bit.bit_length() - 1
            deg = popcount(adj_mask[t] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v_bit = t_bit
            temp ^= t_bit
            count += 1
        
        if best_v_bit == 0:
            best_v_bit = candidates & -candidates
        
        v = best_v_bit.bit_length() - 1
        
        # Option 1: Include v - new candidates are non-neighbors of v in current candidates
        new_cand = candidates & non_adj[v]
        dfs(new_cand, current_size + 1)
        
        # Option 2: Exclude v
        dfs(candidates & ~best_v_bit, current_size)

    dfs(all_mask, 0)
    return best

def _circulant_to_edges(n, S):
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
    return edges

def _greedy_sum_free(n, order):
    """Build a sum-free set by greedily adding elements in the given order."""
    S = set()
    for c in order:
        if c == 0:
            continue
        new_S = set(S)
        new_S.add(c)
        new_S.add((n - c) % n)
        
        is_sum_free = True
        for a in new_S:
            for b in new_S:
                if a == b:
                    continue
                s = (a + b) % n
                if s in new_S and s != a and s != b and s != 0:
                    is_sum_free = False
                    break
            if not is_sum_free:
                break
        
        if is_sum_free:
            S = new_S
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid circulant graph for n from 40 down to 20
    for n in range(40, 20, -1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Structured sets to try first
        structured_sets = []
        
        if n == 40:
            # Powers of 2
            structured_sets.append([1, 2, 4, 8, 16])
            structured_sets.append([1, 2, 4, 8, 16, 32])
            structured_sets.append([1, 3, 9, 27])
            structured_sets.append([1, 2, 4, 7, 14, 28])
            structured_sets.append([1, 2, 4, 8, 17])
            structured_sets.append([1, 2, 3, 5, 8])
            structured_sets.append([1, 2, 4, 8, 16, 33])
            structured_sets.append([1, 2, 4, 8, 16, 31])
            structured_sets.append([1, 2, 4, 8, 16, 34])
            structured_sets.append([1, 2, 4, 8, 16, 35])
            structured_sets.append([1, 2, 4, 8, 16, 36])
            structured_sets.append([1, 2, 4, 8, 16, 37])
            structured_sets.append([1, 2, 4, 8, 16, 38])
            structured_sets.append([1, 2, 4, 8, 16, 39])
            structured_sets.append([1, 2, 4, 8, 16, 20])
            structured_sets.append([1, 2, 4, 8, 16, 24])
            structured_sets.append([1, 2, 4, 8, 16, 28])
            structured_sets.append([1, 2, 4, 8, 16, 30])
            structured_sets.append([1, 2, 4, 8, 16, 34])
            structured_sets.append([1, 2, 4, 8, 16, 38])
        elif n == 39:
            structured_sets.append([1, 2, 4, 8, 16])
            structured_sets.append([1, 2, 4, 8, 17])
            structured_sets.append([1, 2, 4, 8, 16, 32])
            structured_sets.append([1, 3, 9, 27])
            structured_sets.append([1, 2, 3, 5, 8])
            structured_sets.append([1, 2, 4, 7, 14, 28])
            structured_sets.append([1, 2, 4, 8, 16, 33])
            structured_sets.append([1, 2, 4, 8, 16, 31])
            structured_sets.append([1, 2, 4, 8, 16, 34])
            structured_sets.append([1, 2, 4, 8, 16, 35])
            structured_sets.append([1, 2, 4, 8, 16, 36])
            structured_sets.append([1, 2, 4, 8, 16, 37])
            structured_sets.append([1, 2, 4, 8, 16, 38])
            structured_sets.append([1, 2, 4, 8, 16, 20])
            structured_sets.append([1, 2, 4, 8, 16, 24])
            structured_sets.append([1, 2, 4, 8, 16, 28])
            structured_sets.append([1, 2, 4, 8, 16, 30])
        elif n == 38:
            structured_sets.append([1, 2, 4, 8, 16])
            structured_sets.append([1, 2, 4, 8, 17])
            structured_sets.append([1, 2, 4, 8, 16, 32])
            structured_sets.append([1, 3, 9, 27])
            structured_sets.append([1, 2, 3, 5, 8])
            structured_sets.append([1, 2, 4, 7, 14, 28])
            structured_sets.append([1, 2, 4, 8, 16, 31])
            structured_sets.append([1, 2, 4, 8, 16, 33])
            structured_sets.append([1, 2, 4, 8, 16, 34])
            structured_sets.append([1, 2, 4, 8, 16, 35])
            structured_sets.append([1, 2, 4, 8, 16, 36])
            structured_sets.append([1, 2, 4, 8, 16, 37])
        elif n == 37:
            structured_sets.append([1, 2, 4, 8, 16])
            structured_sets.append([1, 2, 4, 8, 17])
            structured_sets.append([1, 2, 4, 8, 16, 32])
            structured_sets.append([1, 3, 9, 27])
            structured_sets.append([1, 2, 3, 5, 8])
            structured_sets.append([1, 2, 4, 7, 14, 28])
            structured_sets.append([1, 2, 4, 8, 16, 31])
            structured_sets.append([1, 2, 4, 8, 16, 33])
            structured_sets.append([1, 2, 4, 8, 16, 34])
            structured_sets.append([1, 2, 4, 8, 16, 35])
            structured_sets.append([1, 2, 4, 8, 16, 36])
        
        # Try structured sets
        for S_test in structured_sets:
            if time.time() > deadline:
                break
            S_test = [s % n for s in S_test if s % n != 0]
            S_test = list(set(S_test))
            if len(S_test) < 2:
                continue
            if _has_triangle_circulant(n, S_test):
                continue
            alpha = _alpha_circulant(n, S_test)
            if alpha < 10:
                best_n = n
                best_edges = _circulant_to_edges(n, S_test)
                found = True
                break
        
        if found:
            break
        
        # Random search with greedy construction
        max_attempts = 3000 if n >= 38 else 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random order for greedy construction
            candidates = list(range(1, n))
            random.shuffle(candidates)
            
            S = _greedy_sum_free(n, candidates)
            
            if len(S) < 2:
                continue
            
            if _has_triangle_circulant(n, list(S)):
                continue
            
            alpha = _alpha_circulant(n, list(S))
            if alpha < 10:
                best_n = n
                best_edges = _circulant_to_edges(n, list(S))
                found = True
                break
        
        if found:
            break
    
    # Fallback: If no circulant graph found, use the known Andrásfai graph (n=26)
    if best_n == 0:
        n0 = 26
        edges0 = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        best_edges = edges0
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
