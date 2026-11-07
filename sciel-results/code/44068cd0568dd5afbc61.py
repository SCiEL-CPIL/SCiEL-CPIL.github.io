import math
import time
import sys

sys.setrecursionlimit(3000)

def construct():
    deadline = time.time() + 88.0  # Safety margin
    
    n = 256
    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # Precompute factor masks for coprime check
    # We need to know which numbers conflict with which.
    # Two numbers conflict if gcd > 1.
    
    # For B&B, we need conflict masks based on a specific ordering.
    # Order by value descending.
    values = list(range(1, n + 1))
    values.sort(reverse=True)
    
    # Map value to index in sorted order
    val_to_idx = {v: i for i, v in enumerate(values)}
    
    # conflict_mask[i] is a bitmask of indices j such that values[i] and values[j] are not coprime.
    # We only need to consider j > i for the forward search? 
    # No, in B&B we process sequentially. 
    # If we decide to include values[i], we must exclude all values[j] (j > i) that conflict with it.
    # So conflict_mask[i] should contain bits for j > i where gcd(values[i], values[j]) > 1.
    
    conflict_masks = [0] * n
    for i in range(n):
        v1 = values[i]
        mask = 0
        for j in range(i + 1, n):
            v2 = values[j]
            if math.gcd(v1, v2) != 1:
                mask |= (1 << j)
        conflict_masks[i] = mask
    
    # Initial potential sum is sum of all values
    total_sum = sum(values)
    
    best_sum = 0
    best_solution_mask = 0  # Bitmask of selected indices in sorted order
    
    # To track the actual solution for local search later
    # We can reconstruct from the mask at the end.
    
    def dfs(idx, current_sum, potential_sum, available_mask):
        nonlocal best_sum, best_solution_mask, deadline
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + potential_sum <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution_mask = available_mask # This is wrong, available_mask is empty at the end.
                # We need to track the selected mask separately.
            return
        
        # Option 1: Include values[idx] if it is available
        if available_mask & (1 << idx):
            # It is available
            new_current_sum = current_sum + values[idx]
            
            # Calculate new potential sum
            # We remove values[idx] and all its conflicts from available_mask
            conflicts = conflict_masks[idx]
            # The set of nodes to remove is (1 << idx) | conflicts
            # But we must only remove those that are currently in available_mask
            to_remove = (1 << idx) | conflicts
            to_remove &= available_mask
            
            # Sum of values to be removed from potential
            # We can precompute a list of values and just sum them? 
            # Or maintain a separate array?
            # For speed, let's just iterate bits in to_remove? 
            # n=256, so at most 256 bits.
            remove_sum = 0
            m = to_remove
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                remove_sum += values[bit_idx]
                m ^= lsb
                
            new_potential_sum = potential_sum - remove_sum
            new_available_mask = available_mask & ~to_remove
            
            # We need to track the selected mask to reconstruct the solution.
            # Let's pass selected_mask as an argument or update a global one?
            # Passing is safer for recursion.
            
            dfs(idx + 1, new_current_sum, new_potential_sum, new_available_mask, 
                selected_mask | (1 << idx))
            
        # Option 2: Exclude values[idx]
        # Only if it's available? No, if it's not available, it's already excluded.
        # But we must advance the index.
        # If it was available, we remove it from potential and available.
        # If it was not available, we just advance.
        
        if available_mask & (1 << idx):
            new_potential_sum = potential_sum - values[idx]
            new_available_mask = available_mask & ~(1 << idx)
            dfs(idx + 1, current_sum, new_potential_sum, new_available_mask, selected_mask)
        else:
            # If not available, just advance
            dfs(idx + 1, current_sum, potential_sum, available_mask, selected_mask)

    # We need to modify dfs to take selected_mask
    def dfs(idx, current_sum, potential_sum, available_mask, selected_mask):
        nonlocal best_sum, best_solution_mask, deadline
        
        if time.time() > deadline:
            return
            
        if current_sum + potential_sum <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution_mask = selected_mask
            return
        
        # Option 1: Include values[idx]
        if available_mask & (1 << idx):
            new_current_sum = current_sum + values[idx]
            
            conflicts = conflict_masks[idx]
            to_remove = (1 << idx) | conflicts
            to_remove &= available_mask
            
            remove_sum = 0
            m = to_remove
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                remove_sum += values[bit_idx]
                m ^= lsb
                
            new_potential_sum = potential_sum - remove_sum
            new_available_mask = available_mask & ~to_remove
            new_selected_mask = selected_mask | (1 << idx)
            
            dfs(idx + 1, new_current_sum, new_potential_sum, new_available_mask, new_selected_mask)
            
        # Option 2: Exclude values[idx]
        if available_mask & (1 << idx):
            new_potential_sum = potential_sum - values[idx]
            new_available_mask = available_mask & ~(1 << idx)
            dfs(idx + 1, current_sum, new_potential_sum, new_available_mask, selected_mask)
        else:
            dfs(idx + 1, current_sum, potential_sum, available_mask, selected_mask)

    # Start B&B
    initial_available_mask = (1 << n) - 1
    dfs(0, 0, total_sum, initial_available_mask, 0)
    
    # Reconstruct solution from best_solution_mask
    solution = []
    for i in range(n):
        if best_solution_mask & (1 << i):
            solution.append(values[i])
            
    # If B&B didn't find anything (shouldn't happen, 1 is always valid), fallback
    if not solution:
        solution = [1]
        
    # If B&B timed out, we might have a suboptimal solution.
    # Run local search to improve it.
    
    # Check if we likely finished (heuristic: if best_sum is high, or time is left)
    # Actually, just always run local search if time permits.
    
    if time.time() < deadline - 5.0:
         # Local search logic from original code, adapted
         
         # Precompute factor masks for local search (using original indices 1..256)
         is_prime_ls = [True] * (n + 1)
         is_prime_ls[0] = False
         is_prime_ls[1] = False
         for i in range(2, int(math.isqrt(n)) + 1):
             if is_prime_ls[i]:
                 for j in range(i * i, n + 1, i):
                     is_prime_ls[j] = False
                 
         primes = [i for i in range(2, n + 1) if is_prime_ls[i]]
         prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}
         
         factor_mask = [0] * (n + 1)
         factor_mask[1] = 0
         for num in range(2, n + 1):
             x = num
             mask = 0
             for p in primes:
                 if p * p > x:
                     break
                 if x % p == 0:
                     mask |= prime_to_bit[p]
                     while x % p == 0:
                         x //= p
             if x > 1:
                 mask |= prime_to_bit[x]
             factor_mask[num] = mask
             
         selected = set(solution)
         selected_masks = {num: factor_mask[num] for num in selected}
         current_sum = sum(solution)
         
         def can_add(num, sel_set, sel_masks):
             mask = factor_mask[num]
             for sm in sel_masks.values():
                 if mask & sm:
                     return False
             return True
         
         # 1-for-1 swaps
         improved = True
         while improved and time.time() < deadline - 2.0:
             improved = False
             for rem in list(selected):
                 if time.time() > deadline - 2.0:
                     break
                 rem_mask = factor_mask[rem]
                 new_sum = current_sum - rem
                 for add in range(1, n + 1):
                     if add in selected:
                         continue
                     add_mask = factor_mask[add]
                     ok = True
                     for sel_num, sel_mask in selected_masks.items():
                         if sel_num == rem:
                             continue
                         if add_mask & sel_mask:
                             ok = False
                             break
                     if ok and new_sum + add > current_sum:
                         selected.remove(rem)
                         del selected_masks[rem]
                         selected.add(add)
                         selected_masks[add] = add_mask
                         current_sum = new_sum + add
                         improved = True
                         break
                 if improved:
                     break
                     
         # 1-for-2 swaps
         improved = True
         while improved and time.time() < deadline - 1.0:
             improved = False
             for rem in list(selected):
                 if time.time() > deadline - 1.0:
                     break
                 rem_mask = factor_mask[rem]
                 new_sum = current_sum - rem
                 for add1 in range(1, n + 1):
                     if add1 in selected:
                         continue
                     add1_mask = factor_mask[add1]
                     ok1 = True
                     for sel_num, sel_mask in selected_masks.items():
                         if sel_num == rem:
                             continue
                         if add1_mask & sel_mask:
                             ok1 = False
                             break
                     if not ok1:
                         continue
                     for add2 in range(add1 + 1, n + 1):
                         if add2 in selected:
                             continue
                         add2_mask = factor_mask[add2]
                         ok2 = True
                         if add1_mask & add2_mask:
                             ok2 = False
                         else:
                             for sel_num, sel_mask in selected_masks.items():
                                 if sel_num == rem:
                                     continue
                                 if add2_mask & sel_mask:
                                     ok2 = False
                                     break
                         if ok2 and new_sum + add1 + add2 > current_sum:
                             selected.remove(rem)
                             del selected_masks[rem]
                             selected.add(add1)
                             selected_masks[add1] = add1_mask
                             selected.add(add2)
                             selected_masks[add2] = add2_mask
                             current_sum = new_sum + add1 + add2
                             improved = True
                             break
                     if improved:
                         break
                 if improved:
                     break

         solution = sorted(selected)

    # Final verification
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if math.gcd(solution[i], solution[j]) != 1:
                return [1]
                
    return [int(x) for x in solution]

# EVOLVE-BLOCK-END
