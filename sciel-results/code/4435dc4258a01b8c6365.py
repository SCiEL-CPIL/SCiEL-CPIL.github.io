import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 512
    # Precompute compatibility bitsets
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

    def is_compatible(x, S_mask):
        return (can[x] & S_mask) == S_mask

    def greedy_construct(order):
        S_mask = 0
        S_list = []
        for x in order:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        return S_list

    def local_search(S_list):
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
        
        improved = True
        # Limit iterations to prevent infinite loops in case of bugs, though logic should terminate
        max_iter = 100
        iter_count = 0
        
        while improved and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # 1. Try to add any x
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        S_list.append(x)
                        improved = True
                        break
            
            if improved:
                continue

            # 2. Try to swap (remove 1, add 1) to change composition
            # This helps escape local minima where no add is possible but a different set of same size is.
            swapped = False
            # Randomize order of u to break symmetry
            u_indices = list(range(len(S_list)))
            random.shuffle(u_indices)
            for idx in u_indices:
                u = S_list[idx]
                S_mask_without_u = S_mask ^ (1 << (u - 1))
                for v in range(1, N + 1):
                    if not (S_mask & (1 << (v - 1))):
                        if is_compatible(v, S_mask_without_u):
                            # Swap u with v
                            S_mask = S_mask_without_u | (1 << (v - 1))
                            S_list[idx] = v
                            swapped = True
                            break
                if swapped:
                    break
            
            if swapped:
                improved = True
                continue

            # 3. Try to remove 1, add 2 (increase size by 1)
            # This is the key move to increase cardinality.
            # Optimize: Only do this if we haven't improved in a while or randomly?
            # For now, try it if stuck.
            # To make it faster, we can limit the search for v, w.
            
            found_triple = False
            u_indices = list(range(len(S_list)))
            random.shuffle(u_indices)
            
            for idx in u_indices:
                u = S_list[idx]
                S_mask_without_u = S_mask ^ (1 << (u - 1))
                
                # Collect candidates v that are compatible with S \ {u}
                candidates = []
                for v in range(1, N + 1):
                    if not (S_mask & (1 << (v - 1))):
                        if is_compatible(v, S_mask_without_u):
                            candidates.append(v)
                
                if len(candidates) < 2:
                    continue
                
                # Check pairs in candidates for compatibility
                # We can break early if we find a valid pair
                for i in range(len(candidates)):
                    v = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        w = candidates[j]
                        # Check if v and w are compatible
                        if (can[v] >> (w - 1)) & 1:
                            # Valid move: remove u, add v, w
                            S_mask = S_mask_without_u | (1 << (v - 1)) | (1 << (w - 1))
                            S_list.pop(idx)
                            S_list.append(v)
                            S_list.append(w)
                            found_triple = True
                            break
                    if found_triple:
                        break
                if found_triple:
                    break
            
            if found_triple:
                improved = True
                continue

        return S_list

    best_size = 0
    best_set = []
    
    # 1. Structural seeds: Divisors of highly composite numbers
    # These provide dense initial cliques.
    for M in [504, 420, 360, 480, 336, 240, 210, 180, 168, 120, 108, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2, 1]:
        if time.time() > deadline:
            break
        divs = [i for i in range(1, N + 1) if M % i == 0]
        s = local_search(divs)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # 2. Greedy with different orderings
    # Order by degree (number of compatible elements) - ascending (hard first) or descending (easy first)?
    # Let's try both.
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(can[i]).count('1')
    
    order_asc = sorted(range(1, N + 1), key=lambda x: degrees[x])
    order_desc = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    
    if time.time() < deadline:
        s = greedy_construct(order_asc)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    if time.time() < deadline:
        s = greedy_construct(order_desc)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # 3. Random restarts
    # Use the best set found so far as a starting point for some restarts?
    # Or just random greedy.
    random.seed(42)
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
        
        # Also try starting from the current best and perturbing?
        # This is "Simulated Annealing" like.
        if best_set and random.random() < 0.1:
             # Perturb best set
             s_perturbed = best_set[:]
             if s_perturbed:
                 # Remove a random element
                 idx = random.randint(0, len(s_perturbed) - 1)
                 s_perturbed.pop(idx)
                 # Add a random compatible element if possible?
                 # Or just run local search on the perturbed set
                 s = local_search(s_perturbed)
                 if len(s) > best_size:
                     best_size = len(s)
                     best_set = s[:]

    return best_set
