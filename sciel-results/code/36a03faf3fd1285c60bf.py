import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _bit_count(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

def _is_regular_subset(adj_bits, subset, mask):
    k = len(subset)
    if k < 6:
        return False
    target_deg = -1
    for v in subset:
        deg = _bit_count(adj_bits[v] & mask)
        if target_deg == -1:
            target_deg = deg
        elif deg != target_deg:
            return False
    return True

def _precompute_subsets(n):
    """
    Precompute subsets of size 6 and 7, and map edges to subsets.
    """
    subsets_6 = []
    subsets_7 = []
    
    # Generate all 6-subsets
    for comb in combinations(range(n), 6):
        subsets_6.append(comb)
        
    # Generate all 7-subsets
    for comb in combinations(range(n), 7):
        subsets_7.append(comb)
        
    # Map edge (u,v) to indices of subsets containing it
    # Use a dictionary or list of lists
    # edge_index = u * n + v ? No, just use a dict with tuple keys
    # Or a 2D array? n=21, so 21x21 is small.
    edge_to_6 = [[[] for _ in range(n)] for __ in range(n)]
    edge_to_7 = [[[] for _ in range(n)] for __ in range(n)]
    
    for idx, subset in enumerate(subsets_6):
        for i in range(6):
            u = subset[i]
            for j in range(i+1, 6):
                v = subset[j]
                edge_to_6[u][v].append(idx)
                edge_to_6[v][u].append(idx)
                
    for idx, subset in enumerate(subsets_7):
        for i in range(7):
            u = subset[i]
            for j in range(i+1, 7):
                v = subset[j]
                edge_to_7[u][v].append(idx)
                edge_to_7[v][u].append(idx)
                
    return subsets_6, subsets_7, edge_to_6, edge_to_7

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_result = None
    
    for n_target in [21, 20]:
        if time.time() > deadline - 10.0:
            break
        
        # Precompute for this n
        subsets_6, subsets_7, edge_to_6, edge_to_7 = _precompute_subsets(n_target)
        
        max_restarts = 20
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Initialize random graph
            adj_bits = [0] * n_target
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if random.random() < 0.5:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            
            # Precompute masks for subsets
            masks_6 = [0] * len(subsets_6)
            for i, subset in enumerate(subsets_6):
                m = 0
                for v in subset:
                    m |= (1 << v)
                masks_6[i] = m
                
            masks_7 = [0] * len(subsets_7)
            for i, subset in enumerate(subsets_7):
                m = 0
                for v in subset:
                    m |= (1 << v)
                masks_7[i] = m
                
            # Initialize validity arrays
            valid_6 = [False] * len(subsets_6)
            valid_7 = [False] * len(subsets_7)
            
            for i, subset in enumerate(subsets_6):
                valid_6[i] = _is_regular_subset(adj_bits, subset, masks_6[i])
                
            for i, subset in enumerate(subsets_7):
                valid_7[i] = _is_regular_subset(adj_bits, subset, masks_7[i])
                
            # Local search
            max_steps = 2000
            for step in range(max_steps):
                if time.time() > deadline - 1.0:
                    break
                    
                # Find violations
                viol_6 = [i for i, v in enumerate(valid_6) if v]
                viol_7 = [i for i, v in enumerate(valid_7) if v]
                
                if not viol_6 and not viol_7:
                    # Check larger sizes
                    violation_found = False
                    for k in range(8, n_target + 1):
                        for subset in combinations(range(n_target), k):
                            mask = 0
                            for v in subset:
                                mask |= (1 << v)
                            if _is_regular_subset(adj_bits, subset, mask):
                                # Found violation
                                u, v = random.sample(list(subset), 2)
                                # Flip edge
                                if adj_bits[u] & (1 << v):
                                    adj_bits[u] &= ~(1 << v)
                                    adj_bits[v] &= ~(1 << u)
                                else:
                                    adj_bits[u] |= (1 << v)
                                    adj_bits[v] |= (1 << u)
                                
                                # Update valid_6 and valid_7
                                for idx in edge_to_6[u][v]:
                                    valid_6[idx] = _is_regular_subset(adj_bits, subsets_6[idx], masks_6[idx])
                                for idx in edge_to_7[u][v]:
                                    valid_7[idx] = _is_regular_subset(adj_bits, subsets_7[idx], masks_7[idx])
                                
                                violation_found = True
                                break
                        if violation_found:
                            break
                    
                    if not violation_found:
                        # Valid graph!
                        edges = []
                        for u in range(n_target):
                            for v in range(u + 1, n_target):
                                if adj_bits[u] & (1 << v):
                                    edges.append([u, v])
                        
                        if best_result is None or n_target > best_result[0]:
                            best_result = (n_target, edges)
                        found_valid = True
                        break
                    else:
                        # Violation found and fixed, continue loop
                        continue
                else:
                    # Pick a violation from 6 or 7
                    if viol_6:
                        idx = random.choice(viol_6)
                        subset = subsets_6[idx]
                    else:
                        idx = random.choice(viol_7)
                        subset = subsets_7[idx]
                        
                    u, v = random.sample(list(subset), 2)
                    
                    # Flip edge
                    if adj_bits[u] & (1 << v):
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    else:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                    
                    # Update valid_6 and valid_7
                    for idx6 in edge_to_6[u][v]:
                        valid_6[idx6] = _is_regular_subset(adj_bits, subsets_6[idx6], masks_6[idx6])
                    for idx7 in edge_to_7[u][v]:
                        valid_7[idx7] = _is_regular_subset(adj_bits, subsets_7[idx7], masks_7[idx7])
        
        if found_valid:
            break
        
    if best_result is not None:
        n_found, edges_found = best_result
        return {'n': n_found, 'edges': edges_found}
    else:
        return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
