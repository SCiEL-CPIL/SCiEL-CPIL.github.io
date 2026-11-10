import time
import sys
from math import ceil, log2

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute popcount
    popcount = [0] * (1 << n)
    for m in range(1, 1 << n):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
    
    is_valid_part = [False] * (1 << n)
    for m in range(1, 1 << n):
        is_valid_part[m] = is_clique[m] or is_independent[m]
    
    # Find max clique and max independent set sizes for lower bound
    max_clique_size = 0
    max_indep_size = 0
    for m in range(1, 1 << n):
        if is_clique[m]:
            max_clique_size = max(max_clique_size, popcount[m])
        if is_independent[m]:
            max_indep_size = max(max_indep_size, popcount[m])
    
    lb = max(ceil(n / max(max_clique_size, max_indep_size)), 1)
    
    # For each vertex, precompute valid parts containing it
    # valid_parts_with[v] = list of (size, mask) for valid parts containing v
    valid_parts_with = [[] for _ in range(n)]
    for m in range(1, 1 << n):
        if is_valid_part[m]:
            sz = popcount[m]
            # For each vertex in m, add this part
            mm = m
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                valid_parts_with[v].append((sz, m))
                mm ^= lsb
    
    # Sort each list by size descending
    for v in range(n):
        valid_parts_with[v].sort(key=lambda x: -x[0])
    
    # Backtracking with memoization on (remaining_mask, k_used)
    # We want to find if we can partition remaining_mask into (k_limit - k_used) valid parts
    memo = {}
    
    def can_partition(remaining_mask, k_used, k_limit):
        if remaining_mask == 0:
            return True
        if k_used == k_limit:
            return False
        
        key = (remaining_mask, k_used)
        if key in memo:
            return memo[key]
        
        # Pick the smallest vertex in remaining_mask
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        
        rest = remaining_mask ^ lsb
        
        # Try all valid parts containing v that are subsets of remaining_mask
        # Use the precomputed list but filter for subset of remaining_mask
        for sz, S in valid_parts_with[v]:
            if S & remaining_mask != S:
                continue
            if not is_valid_part[S]:
                continue
            new_remaining = remaining_mask ^ S
            if can_partition(new_remaining, k_used + 1, k_limit):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    # Reconstruct the partition
    def find_partition(remaining_mask, k_used, k_limit):
        if remaining_mask == 0:
            return []
        if k_used == k_limit:
            return None
        
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        
        for sz, S in valid_parts_with[v]:
            if S & remaining_mask != S:
                continue
            if not is_valid_part[S]:
                continue
            new_remaining = remaining_mask ^ S
            sub = find_partition(new_remaining, k_used + 1, k_limit)
            if sub is not None:
                verts = []
                m = S
                while m:
                    l = m & -m
                    verts.append(l.bit_length() - 1)
                    m ^= l
                return [verts] + sub
        
        return None
    
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    for k in range(lb, n + 1):
        if time.time() - start_time > deadline:
            break
        
        memo.clear()
        if can_partition(full_mask, 0, k):
            partition = find_partition(full_mask, 0, k)
            if partition is not None:
                # Verify and return
                result = []
                for part in partition:
                    result.append([int(v) for v in part])
                return result
    
    # Fallback: greedy
    return _greedy_construct()

def _greedy_construct():
    """Fallback greedy constructor."""
    n = P['n']
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            is_cli = True
            is_ind = True
            for u in part:
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    is_ind = False
                else:
                    is_cli = False
            if is_cli or is_ind:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    
    return parts

# EVOLVE-BLOCK-END
