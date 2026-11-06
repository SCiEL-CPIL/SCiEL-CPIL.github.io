import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    # Precompute compatibility bitsets
    # can[i] is a bitset of j (1..N) such that lcm(i, j) <= N
    # We use 0-indexed bits: bit j-1 for number j
    can = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << (j - 1))
        can[i] = mask
    
    # Helper to check if x is compatible with set S (represented as bitset)
    def is_compatible(x, S_mask):
        return (can[x] & S_mask) == S_mask
    
    # Greedy construction
    def greedy_construct(order):
        S_mask = 0
        S_list = []
        for x in order:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        return S_list
    
    # Enhanced local search with add, swap, and 2-for-1 moves
    def local_search(S_list, max_iters=200):
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Phase 1: Try to add any x
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        S_list.append(x)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Phase 2: Try swap - remove x, add y (net zero, but may enable future adds)
            # Only useful if it enables subsequent adds. Try: for each x in S, remove it,
            # then greedily add all compatible elements. If result is larger, keep it.
            best_swap_size = len(S_list)
            best_swap_list = None
            for x in S_list:
                if time.time() > deadline:
                    break
                S_mask_temp = S_mask & ~(1 << (x - 1))
                S_list_temp = [y for y in S_list if y != x]
                # Try to add any compatible element to S_list_temp
                for y in range(1, N + 1):
                    if not (S_mask_temp & (1 << (y - 1))):
                        if is_compatible(y, S_mask_temp):
                            S_mask_temp |= (1 << (y - 1))
                            S_list_temp.append(y)
                            if len(S_list_temp) > best_swap_size:
                                best_swap_size = len(S_list_temp)
                                best_swap_list = S_list_temp[:]
                            # Continue adding more
                            for z in range(1, N + 1):
                                if not (S_mask_temp & (1 << (z - 1))):
                                    if is_compatible(z, S_mask_temp):
                                        S_mask_temp |= (1 << (z - 1))
                                        S_list_temp.append(z)
                            if len(S_list_temp) > best_swap_size:
                                best_swap_size = len(S_list_temp)
                                best_swap_list = S_list_temp[:]
                            break
            if best_swap_list is not None and len(best_swap_list) > len(S_list):
                S_list = best_swap_list
                S_mask = 0
                for x in S_list:
                    S_mask |= (1 << (x - 1))
                improved = True
                continue
            
            # Phase 3: Try 2-for-1 - remove x, add two elements y, z
            for x in S_list:
                if time.time() > deadline:
                    break
                S_mask_temp = S_mask & ~(1 << (x - 1))
                S_list_temp = [y for y in S_list if y != x]
                # Find two elements that can be added
                added = 0
                for y in range(1, N + 1):
                    if added >= 2:
                        break
                    if not (S_mask_temp & (1 << (y - 1))):
                        if is_compatible(y, S_mask_temp):
                            S_mask_temp |= (1 << (y - 1))
                            S_list_temp.append(y)
                            added += 1
                if added >= 2 and len(S_list_temp) > len(S_list):
                    S_list = S_list_temp
                    S_mask = S_mask_temp
                    improved = True
                    break
        
        return S_list
    
    best_size = 0
    best_set = []
    
    import random
    
    # 1. Natural order
    s1 = greedy_construct(range(1, N + 1))
    s1 = local_search(s1)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1[:]
        
    # 2. Reverse order
    s2 = greedy_construct(range(N, 0, -1))
    s2 = local_search(s2)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2[:]
        
    # 3. Random orders with enhanced local search
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
    # 4. Try divisors of highly composite numbers
    for M in [360, 420, 480, 240, 120, 72, 60, 48, 36, 24, 480, 432, 420, 360, 336, 300, 280, 252, 240, 210]:
        if time.time() > deadline - 3:
            break
        divs = [i for i in range(1, N + 1) if M % i == 0]
        s = local_search(divs)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # 5. Final polish: try to add any remaining compatible elements
    if time.time() < deadline - 1:
        S_mask = 0
        for x in best_set:
            S_mask |= (1 << (x - 1))
        for x in range(1, N + 1):
            if not (S_mask & (1 << (x - 1))):
                if is_compatible(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    best_set.append(x)
    
    return best_set
