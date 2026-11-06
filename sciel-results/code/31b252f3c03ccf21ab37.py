import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute compatibility bitsets
    # can[i] is a bitset of j (1..N) such that lcm(i, j) <= N
    # We use 0-indexed bits: bit j-1 for number j
    can = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                mask |= (1 << (j - 1))
                continue
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << (j - 1))
        can[i] = mask
    
    # Helper to check if x is compatible with set S (represented as bitset)
    def is_compatible(x, S_mask):
        # S_mask has bits set for numbers in S
        # We need to check if all y in S are compatible with x
        # i.e., (can[x] & S_mask) == S_mask
        return (can[x] & S_mask) == S_mask
    
    # Greedy construction
    def greedy_construct(order):
        S_mask = 0
        S_list = []
        for x in order:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        return S_list, S_mask
    
    # Local search: try to add elements
    def local_search_add(S_list, S_mask):
        improved = True
        while improved:
            improved = False
            # Try to add any x
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        S_list.append(x)
                        improved = True
                        break
        return S_list, S_mask
    
    # Local search: try swaps (remove one, add two compatible ones)
    def local_search_swap(S_list, S_mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline:
                    break
                # Remove S_list[i]
                removed = S_list[i]
                new_mask = S_mask & ~(1 << (removed - 1))
                
                # Find all candidates compatible with new_mask
                candidates = []
                for x in range(1, N + 1):
                    if x == removed:
                        continue
                    if not (new_mask & (1 << (x - 1))):  # not already in set
                        if is_compatible(x, new_mask):
                            candidates.append(x)
                
                # Try to find two compatible candidates
                if len(candidates) >= 2:
                    found_pair = False
                    for c1_idx in range(len(candidates)):
                        c1 = candidates[c1_idx]
                        for c2_idx in range(c1_idx + 1, len(candidates)):
                            c2 = candidates[c2_idx]
                            # Check if c1 and c2 are compatible
                            if (can[c1] & (1 << (c2 - 1))) != 0:
                                # Add both
                                new_mask |= (1 << (c1 - 1))
                                new_mask |= (1 << (c2 - 1))
                                new_list = S_list[:]
                                new_list.pop(i)
                                new_list.append(c1)
                                new_list.append(c2)
                                S_list = new_list
                                S_mask = new_mask
                                improved = True
                                found_pair = True
                                break
                        if found_pair:
                            break
                if found_pair:
                    break  # Restart local search with new S
        return S_list, S_mask
    
    best_size = 0
    best_set = []
    best_mask = 0
    
    # Strategy 1: Start with divisors of highly composite numbers
    # These are known to form valid sets
    for M in [360, 420, 480, 240, 120, 72, 60, 48, 36, 24, 480, 360, 420]:
        if time.time() > deadline:
            break
        divs = [i for i in range(1, N + 1) if M % i == 0]
        S_list, S_mask = greedy_construct(divs)
        S_list, S_mask = local_search_add(S_list, S_mask)
        S_list, S_mask = local_search_swap(S_list, S_mask)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            best_mask = S_mask
    
    # Strategy 2: Random restarts with greedy + local search
    for _ in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_list, S_mask = greedy_construct(order)
        S_list, S_mask = local_search_add(S_list, S_mask)
        S_list, S_mask = local_search_swap(S_list, S_mask)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            best_mask = S_mask
    
    # Strategy 3: Try natural and reverse orders
    for order in [range(1, N + 1), range(N, 0, -1)]:
        if time.time() > deadline:
            break
        S_list, S_mask = greedy_construct(list(order))
        S_list, S_mask = local_search_add(S_list, S_mask)
        S_list, S_mask = local_search_swap(S_list, S_mask)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            best_mask = S_mask
    
    # Strategy 4: Start from best_set and try more swaps
    if best_set and time.time() < deadline:
        S_list, S_mask = best_set[:], best_mask
        S_list, S_mask = local_search_swap(S_list, S_mask)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            best_mask = S_mask
    
    # Final polish: try to add any remaining compatible elements
    if best_set:
        S_list, S_mask = best_set[:], best_mask
        S_list, S_mask = local_search_add(S_list, S_mask)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            best_mask = S_mask
    
    return best_set if best_set else [1]
