import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    K = 18  # B&B on first K elements
    
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Precompute conflict masks for fast bitset checking
    conflict_masks = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        seen = set()
        # x is first: x, y, z with z = 2y - x, y > x
        for y in range(x + 1, N + 1):
            z = 2 * y - x
            if z <= N and z != x and z != y:
                mask = (1 << min(y, z)) | (1 << max(y, z))
                if mask not in seen:
                    seen.add(mask)
                    conflict_masks[x].append(mask)
        # x is middle: y, x, z with y < x < z, y + z = 2x
        for y in range(1, x):
            z = 2 * x - y
            if z > x and z <= N:
                mask = (1 << min(y, z)) | (1 << max(y, z))
                if mask not in seen:
                    seen.add(mask)
                    conflict_masks[x].append(mask)
        # x is last: y, z, x with y < z < x, z = (y+x)/2
        for y in range(1, x):
            s = y + x
            if s % 2 == 0:
                z = s // 2
                if 1 <= z < x and z != y:
                    mask = (1 << min(y, z)) | (1 << max(y, z))
                    if mask not in seen:
                        seen.add(mask)
                        conflict_masks[x].append(mask)
    
    def can_add(x, bits):
        for mask in conflict_masks[x]:
            if (bits & mask) == mask:
                return False
        return True
    
    # Suffix sum for upper bound
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    best_set = set()
    best_score = 0.0
    
    # Greedy initialization
    in_bits = 0
    greedy_set = set()
    for x in range(1, N + 1):
        if can_add(x, in_bits):
            in_bits |= (1 << x)
            greedy_set.add(x)
    greedy_score = sum(weights[x] for x in greedy_set)
    best_score = greedy_score
    best_set = greedy_set.copy()
    
    # B&B on first K elements, greedy completion for rest
    def complete_greedily(bits):
        for x in range(K + 1, N + 1):
            if can_add(x, bits):
                bits |= (1 << x)
        return bits
    
    def bb(idx, bits, cur_sum):
        nonlocal best_set, best_score
        
        if time.time() > deadline:
            return
        
        if idx > K:
            final_bits = complete_greedily(bits)
            final_score = 0.0
            for i in range(1, N + 1):
                if (final_bits >> i) & 1:
                    final_score += weights[i]
            if final_score > best_score:
                best_score = final_score
                best_set = set()
                for i in range(1, N + 1):
                    if (final_bits >> i) & 1:
                        best_set.add(i)
            return
        
        # Upper bound
        ub = cur_sum + suffix_sum[idx]
        if ub <= best_score:
            return
        
        x = idx
        
        # Try include
        if can_add(x, bits):
            bb(idx + 1, bits | (1 << x), cur_sum + weights[x])
        
        # Try exclude
        bb(idx + 1, bits, cur_sum)
    
    bb(1, 0, 0.0)
    
    return list(best_set)
