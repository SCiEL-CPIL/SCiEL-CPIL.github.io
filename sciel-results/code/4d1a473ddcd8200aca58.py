import time
import random

def _popcount(x):
    return bin(x).count('1')

def _has_K4_bitset(adj_bits, n):
    # adj_bits[i] is an int with bits set for neighbors
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_bits[i] & (1 << j)):
                continue
            # Triangle i,j,k
            # Common neighbors of i and j
            common_ij = adj_bits[i] & adj_bits[j]
            # k must be > j
            # We can just iterate k
            for k in range(j+1, n):
                if not (common_ij & (1 << k)):
                    continue
                # i,j,k is a triangle
                # Common neighbors of i,j,k
                common_ijk = common_ij & adj_bits[k]
                # l must be > k
                # Check if any l > k is in common_ijk
                # Mask out bits <= k
                mask_l = ~((1 << (k+1)) - 1)
                if common_ijk & mask_l:
                    return True
    return False

def _has_indep_6_bitset(adj_bits, n):
    # Look for K6 in complement
    # comp[i] = ~adj_bits[i] & mask_all
    mask_all = (1 << n) - 1
    # Remove self bits from adj_bits? 
    # adj_bits[i] should not have bit i set.
    
    comp = [0]*n
    for i in range(n):
        # ~adj_bits[i] will have 1s everywhere except neighbors and i (if adj_bits[i] has 0 at i)
        # We need to ensure bit i is 0 in comp[i]
        comp[i] = (~adj_bits[i]) & mask_all
        comp[i] &= ~(1 << i)
        
    def find_clique(candidates, size):
        if size == 0:
            return True
        if _popcount(candidates) < size:
            return False
        # Pick a vertex with high degree in the induced subgraph?
        # Or just pick the first one
        # Heuristic: pick vertex v in candidates
        # To speed up, pick v that maximizes common neighbors?
        # For now, pick first set bit
        # v = (candidates & -candidates).bit_length() - 1
        # Better: iterate
        # We can prune if max possible clique size is small
        
        # Simple backtracking
        # Pick a vertex v
        # Try to include v
        # v = lowest set bit
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        new_cands = candidates & comp[v]
        if find_clique(new_cands, size - 1):
            return True
        # Exclude v
        candidates &= ~v_bit
        if find_clique(candidates, size):
            return True
        return False

    return find_clique(mask_all, 6)

def _is_valid_bitset(adj_bits, n):
    return not _has_K4_bitset(adj_bits, n) and not _has_indep_6_bitset(adj_bits, n)

def _edges_from_bitset(adj_bits, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _local_search_bitset(adj_bits, n, deadline):
    curr = adj_bits[:]
    
    while time.time() < deadline:
        # Check K4
        if _has_K4_bitset(curr, n):
            # Find a K4 and remove an edge
            # Re-implement to return the K4
            # For speed, we can just pick a random edge and remove it?
            # No, that's slow.
            # Let's find one K4
            k4 = _find_K4(curr, n)
            if k4:
                u, v, w, x = k4
                pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                ru, rv = random.choice(pairs)
                curr[ru] &= ~(1 << rv)
                curr[rv] &= ~(1 << ru)
            continue
            
        # Check Indep 6
        if _has_indep_6_bitset(curr, n):
            # Find one and add an edge
            indep = _find_indep_6(curr, n)
            if indep:
                u, v = random.sample(indep, 2)
                curr[u] |= (1 << v)
                curr[v] |= (1 << u)
            continue
            
        # Valid
        return curr
    return None

def _find_K4(adj_bits, n):
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_bits[i] & (1 << j)): continue
            common_ij = adj_bits[i] & adj_bits[j]
            for k in range(j+1, n):
                if not (common_ij & (1 << k)): continue
                common_ijk = common_ij & adj_bits[k]
                mask_l = ~((1 << (k+1)) - 1)
                cands = common_ijk & mask_l
                if cands:
                    l_bit = cands & -cands
                    l = l_bit.bit_length() - 1
                    return (i, j, k, l)
    return None

def _find_indep_6(adj_bits, n):
    mask_all = (1 << n) - 1
    comp = [0]*n
    for i in range(n):
        comp[i] = (~adj_bits[i]) & mask_all
        comp[i] &= ~(1 << i)
        
    path = []
    def backtrack(candidates, size):
        if size == 0:
            return True
        if _popcount(candidates) < size:
            return False
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        path.append(v)
        new_cands = candidates & comp[v]
        if backtrack(new_cands, size - 1):
            return True
        path.pop()
        candidates &= ~v_bit
        if backtrack(candidates, size):
            return True
        return False
        
    if backtrack(mask_all, 6):
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=35
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        n = 35
        p = 0.5
        adj_bits = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    
        res = _local_search_bitset(adj_bits, n, deadline)
        if res is not None:
            if _is_valid_bitset(res, n):
                return {'n': n, 'edges': _edges_from_bitset(res, n)}
                
    # Try n=34
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        n = 34
        adj_bits = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        res = _local_search_bitset(adj_bits, n, deadline)
        if res is not None:
            if _is_valid_bitset(res, n):
                return {'n': n, 'edges': _edges_from_bitset(res, n)}
                
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
