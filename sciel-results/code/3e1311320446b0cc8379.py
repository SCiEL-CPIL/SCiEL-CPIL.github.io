import numpy as np
from math import lcm, gcd
import time
import sys
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    weights = [1.0 / x for x in nums]
    
    # Build conflict bitsets for fast checking
    conflict = [0] * n
    for i in range(n):
        a = nums[i]
        bits = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            g = gcd(a, b)
            if (a // g) * b <= 256:
                bits |= (1 << j)
        conflict[i] = bits
    
    def sol_weight(sol):
        s = 0.0
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            s += weights[i]
            x ^= lsb
        return s
    
    def valid(sol):
        x = sol
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            if sol & conflict[i]:
                return False
            x ^= lsb
        return True
    
    # Initial Solution: Greedy by ascending value (smallest first, highest weight)
    def make_initial_greedy():
        sol = 0
        for i in range(n):
            if (sol & conflict[i]) == 0:
                sol |= (1 << i)
        return sol
    
    best_sol = make_initial_greedy()
    best_w = sol_weight(best_sol)
    
    # Local Search with 1-Add, 1-Swap, 2-Add moves
    def local_search(sol, w, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-12
            best_move = None
            
            # 1-Add
            for j in range(n):
                if sol & (1 << j):
                    continue
                if (sol & conflict[j]) == 0:
                    gain = weights[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (sol | (1 << j), gain)
            
            # 1-Swap
            for i in range(n):
                if not (sol & (1 << i)):
                    continue
                base = sol ^ (1 << i)
                for j in range(n):
                    if base & (1 << j):
                        continue
                    if (base & conflict[j]) == 0:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (base | (1 << j), gain)
            
            # 2-Add
            if time.time() < deadline - 0.5:
                not_in_sol = [j for j in range(n) if not (sol & (1 << j))]
                compatible_with_sol = [j for j in not_in_sol if (sol & conflict[j]) == 0]
                
                for idx_j in range(len(compatible_with_sol)):
                    j = compatible_with_sol[idx_j]
                    for idx_k in range(idx_j + 1, len(compatible_with_sol)):
                        k = compatible_with_sol[idx_k]
                        if (conflict[j] & (1 << k)) != 0:
                            continue
                        gain = weights[j] + weights[k]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (sol | (1 << j) | (1 << k), gain)
            
            if best_move is not None:
                sol, w = best_move[0], w + best_gain
                improved = True
        
        return sol, w
    
    best_sol, best_w = local_search(best_sol, best_w, deadline)
    
    # Key improvement: Identify a small "core" of high-weight mutually compatible elements,
    # then solve the remaining subproblem exactly using bitmask DP
    
    # Sort elements by weight descending (i.e., by value ascending)
    order_by_weight = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    # Try different core sizes: pick top k elements by weight that form an independent set
    # Then solve the rest exactly
    
    # First, find a good core using greedy
    def find_core(k_max=8):
        """Find a set of up to k_max high-weight elements that are mutually compatible."""
        core = []
        used = 0
        for i in order_by_weight:
            if len(core) >= k_max:
                break
            if (used & conflict[i]) == 0:
                core.append(i)
                used |= (1 << i)
        return core, used
    
    # Bitmask DP for exact solution on a subset of elements
    def exact_solve(indices, deadline):
        """Solve exactly for the given list of indices using bitmask DP."""
        m = len(indices)
        if m == 0:
            return 0, 0
        
        # Build local conflict masks
        local_conflict = [0] * m
        for i in range(m):
            for j in range(m):
                if i != j and (conflict[indices[i]] & (1 << indices[j])):
                    local_conflict[i] |= (1 << j)
        
        # DP: dp[mask] = max weight using subset 'mask' of {0..m-1}
        # But this is 2^m which is too large for m > 20 or so.
        # Use branch and bound instead.
        
        # Branch and bound with memoization
        from functools import lru_cache
        
        # Sort indices by weight descending for better pruning
        sorted_idx = sorted(range(m), key=lambda i: weights[indices[i]], reverse=True)
        
        best_val = 0.0
        best_mask = 0
        
        def dfs(pos, current_mask, current_w):
            nonlocal best_val, best_mask
            
            if time.time() > deadline - 1.0:
                return
            
            if pos == m:
                if current_w > best_val:
                    best_val = current_w
                    best_mask = current_mask
                return
            
            i = sorted_idx[pos]
            
            # Option 1: Don't include i
            dfs(pos + 1, current_mask, current_w)
            
            # Option 2: Include i if compatible
            if (current_mask & local_conflict[i]) == 0:
                new_mask = current_mask | (1 << i)
                dfs(pos + 1, new_mask, current_w + weights[indices[i]])
        
        # Use iterative approach with pruning
        # For small m, use full DP
        if m <= 15:
            # Full bitmask DP
            size = 1 << m
            dp = [0.0] * size
            choice = [0] * size  # 0 = not included, 1 = included
            
            for mask in range(1, size):
                # Find lowest set bit
                lsb = mask & (-mask)
                i = lsb.bit_length() - 1
                prev_mask = mask ^ lsb
                
                # Option: don't include i
                val_excl = dp[prev_mask]
                
                # Option: include i if no conflict
                val_incl = 0.0
                if (prev_mask & local_conflict[i]) == 0:
                    val_incl = dp[prev_mask] + weights[indices[i]]
                
                if val_incl >= val_excl:
                    dp[mask] = val_incl
                    choice[mask] = 1
                else:
                    dp[mask] = val_excl
                    choice[mask] = 0
            
            # Reconstruct
            mask = size - 1
            result_mask = 0
            while mask:
                lsb = mask & (-mask)
                i = lsb.bit_length() - 1
                if choice[mask]:
                    result_mask |= lsb
                mask ^= lsb
            
            return dp[size - 1], result_mask
        else:
            # Branch and bound for larger m
            # Use a simpler heuristic-based approach
            best_val = 0.0
            best_mask = 0
            
            # Greedy initialization
            greedy_mask = 0
            for i in sorted_idx:
                if (greedy_mask & local_conflict[i]) == 0:
                    greedy_mask |= (1 << i)
            greedy_w = sum(weights[indices[i]] for i in range(m) if greedy_mask & (1 << i))
            best_val = greedy_w
            best_mask = greedy_mask
            
            # Local search on the exact subproblem
            for _ in range(100):
                if time.time() > deadline - 1.0:
                    break
                improved = False
                
                # 1-Add
                for i in range(m):
                    if best_mask & (1 << i):
                        continue
                    if (best_mask & local_conflict[i]) == 0:
                        new_mask = best_mask | (1 << i)
                        new_w = best_val + weights[indices[i]]
                        if new_w > best_val:
                            best_val = new_w
                            best_mask = new_mask
                            improved = True
                
                # 1-Swap
                for i in range(m):
                    if not (best_mask & (1 << i)):
                        continue
                    base = best_mask ^ (1 << i)
                    base_w = best_val - weights[indices[i]]
                    for j in range(m):
                        if base & (1 << j):
                            continue
                        if (base & local_conflict[j]) == 0:
                            new_mask = base | (1 << j)
                            new_w = base_w + weights[indices[j]]
                            if new_w > best_val:
                                best_val = new_w
                                best_mask = new_mask
                                improved = True
                
                if not improved:
                    break
            
            return best_val, best_mask
    
    # Strategy: Try different core sizes and solve the rest exactly
    for core_size in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5.0:
            break
        
        # Find a core of size core_size
        core = []
        core_mask = 0
        for i in order_by_weight:
            if len(core) >= core_size:
                break
            if (core_mask & conflict[i]) == 0:
                core.append(i)
                core_mask |= (1 << i)
        
        if len(core) < core_size:
            continue
        
        # The remaining elements are those not in core and not conflicting with core
        remaining = []
        for i in range(n):
            if core_mask & (1 << i):
                continue
            if (core_mask & conflict[i]) == 0:
                remaining.append(i)
        
        if not remaining:
            continue
        
        # Solve the remaining subproblem exactly
        rem_val, rem_mask = exact_solve(remaining, deadline)
        
        # Reconstruct the solution
        full_sol = core_mask
        for i in range(len(remaining)):
            if rem_mask & (1 << i):
                full_sol |= (1 << remaining[i])
        
        full_w = sol_weight(full_sol)
        
        if full_w > best_w:
            best_w = full_w
            best_sol = full_sol
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        best_sol, best_w = local_search(best_sol, best_w, deadline)
    
    # Convert bitset back to list
    result = [nums[i] for i in range(n) if best_sol & (1 << i)]
    
    # Final Validation
    if len(result) > 0:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if lcm(result[i], result[j]) <= 256:
                    return list(range(129, 257))
    
    return result
