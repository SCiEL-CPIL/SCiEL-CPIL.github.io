"""
Alternative strategy: Greedy construction with multiple randomized heuristics,
followed by a targeted local search that focuses on adding elements in order
of increasing "conflict count" (fewest constraints first), and finally a
randomized perturbation phase to escape local optima.
"""

import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Precompute all 4-APs and their conflict structure
    # For each element, store the list of masks representing the other 3 elements
    # of each 4-AP containing that element.
    conflicts = [[] for _ in range(N)]
    
    # Also precompute for each element, the count of 4-APs it participates in
    ap_count = [0] * N
    
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For a: others are b, c, e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(mask_a)
            ap_count[a - 1] += 1
            
            # For b: others are a, c, e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(mask_b)
            ap_count[b - 1] += 1
            
            # For c: others are a, b, e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(mask_c)
            ap_count[c - 1] += 1
            
            # For e: others are a, b, c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(mask_e)
            ap_count[e - 1] += 1
    
    def is_valid_addition(x, S):
        """Check if element x (1-indexed) can be added to bitmask S."""
        idx = x - 1
        for m in conflicts[idx]:
            if (S & m) == m:
                return False
        return True
    
    def greedy_by_order(order, S=0):
        """
        Greedy construction using a given order of elements.
        Returns the resulting bitmask.
        """
        for x in order:
            if is_valid_addition(x, S):
                S |= (1 << (x - 1))
        return S
    
    def local_search_add(S):
        """
        Try to add any remaining elements that can be added.
        Returns the updated bitmask.
        """
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if not (S & (1 << (x - 1))):
                    if is_valid_addition(x, S):
                        S |= (1 << (x - 1))
                        changed = True
        return S
    
    def local_search_swap(S, max_swaps=50):
        """
        Try to improve S by removing one element and adding two.
        This is a 1-2 exchange move.
        """
        improved = True
        swaps = 0
        while improved and swaps < max_swaps:
            improved = False
            # Try removing each element in S
            for x in range(1, N + 1):
                if S & (1 << (x - 1)):
                    S_without_x = S & ~(1 << (x - 1))
                    # Try adding pairs of elements not in S_without_x
                    for y in range(1, N + 1):
                        if S_without_x & (1 << (y - 1)):
                            continue
                        if not is_valid_addition(y, S_without_x):
                            continue
                        S_with_y = S_without_x | (1 << (y - 1))
                        for z in range(y + 1, N + 1):
                            if S_with_y & (1 << (z - 1)):
                                continue
                            if not is_valid_addition(z, S_with_y):
                                continue
                            S_with_yz = S_with_y | (1 << (z - 1))
                            # If we can add both, we've improved by 1
                            S = S_with_yz
                            improved = True
                            swaps += 1
                            break
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    # Strategy: Multiple greedy restarts with different orderings
    best_S = 0
    best_size = 0
    
    # Ordering 1: Natural order (1, 2, ..., 512)
    order1 = list(range(1, N + 1))
    S1 = greedy_by_order(order1)
    S1 = local_search_add(S1)
    size1 = bin(S1).count('1')
    if size1 > best_size:
        best_size = size1
        best_S = S1
    
    # Ordering 2: Reverse order (512, 511, ..., 1)
    order2 = list(range(N, 0, -1))
    S2 = greedy_by_order(order2)
    S2 = local_search_add(S2)
    size2 = bin(S2).count('1')
    if size2 > best_size:
        best_size = size2
        best_S = S2
    
    # Ordering 3: Random order
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        order3 = list(range(1, N + 1))
        random.shuffle(order3)
        S3 = greedy_by_order(order3)
        S3 = local_search_add(S3)
        size3 = bin(S3).count('1')
        if size3 > best_size:
            best_size = size3
            best_S = S3
    
    # Ordering 4: Order by increasing AP count (least constrained first)
    order4 = sorted(range(1, N + 1), key=lambda x: ap_count[x - 1])
    S4 = greedy_by_order(order4)
    S4 = local_search_add(S4)
    size4 = bin(S4).count('1')
    if size4 > best_size:
        best_size = size4
        best_S = S4
    
    # Ordering 5: Order by decreasing AP count (most constrained first)
    order5 = sorted(range(1, N + 1), key=lambda x: -ap_count[x - 1])
    S5 = greedy_by_order(order5)
    S5 = local_search_add(S5)
    size5 = bin(S5).count('1')
    if size5 > best_size:
        best_size = size5
        best_S = S5
    
    # Ordering 6: Randomized AP count order
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        # Perturb the AP count order
        order6 = list(range(1, N + 1))
        # Weighted shuffle: elements with lower AP count more likely to come first
        weights = [1.0 / (ap_count[x - 1] + 1) for x in order6]
        # Simple approach: sort by ap_count + random noise
        noisy_order = [(ap_count[x - 1] + random.random() * 5, x) for x in order6]
        noisy_order.sort()
        order6 = [x for _, x in noisy_order]
        S6 = greedy_by_order(order6)
        S6 = local_search_add(S6)
        size6 = bin(S6).count('1')
        if size6 > best_size:
            best_size = size6
            best_S = S6
    
    # Local search with swaps on the best set
    if time.time() < deadline - 10:
        S_swap = local_search_swap(best_S, max_swaps=30)
        S_swap = local_search_add(S_swap)
        size_swap = bin(S_swap).count('1')
        if size_swap > best_size:
            best_size = size_swap
            best_S = S_swap
    
    # Final: try adding any remaining elements
    best_S = local_search_add(best_S)
    best_size = bin(best_S).count('1')
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
    
    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(result):
        return [1]
        
    return sorted(result)

# EVOLVE-BLOCK-END
