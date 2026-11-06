import time
import math
from math import gcd
from fractions import Fraction
import sys
import random

def construct():
    start_time = time.time()
    # Safety margin: return before 90s wall-clock limit. 
    # We target 85s to be safe.
    deadline = start_time + 85.0

    # Candidates 2..256
    all_candidates = list(range(2, 257))
    n = len(all_candidates)
    
    # Precompute inverse values
    inv = {x: 1.0/x for x in all_candidates}
    
    # Precompute compatibility matrix
    # compat[i][j] is True if lcm(candidates[i], candidates[j]) > 256
    # We represent the graph as a list of bitsets (integers) for conflicts.
    # conflict_mask[i] has bit j set if candidates[i] and candidates[j] are INCOMPATIBLE.
    
    conflict_mask = [0] * n
    
    # To speed up construction, we can use the property that conflict depends on gcd.
    # lcm(a,b) = a*b/gcd(a,b).
    # Conflict if a*b/gcd(a,b) <= 256.
    
    for i in range(n):
        ai = all_candidates[i]
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            aj = all_candidates[j]
            # Optimization: if ai > aj, we might have already computed? 
            # No, let's just compute. 255^2 is small.
            g = gcd(ai, aj)
            if (ai * aj) // g <= 256:
                mask_i |= (1 << j)
        conflict_mask[i] = mask_i

    # We want to maximize sum(1/x) subject to independent set constraint.
    
    # Strategy:
    # The graph is sparse for large numbers. Dense for small numbers.
    # We split into "Core" (small, high degree) and "Tail" (large, low degree).
    # Let's define Core as numbers 2..16 (15 numbers).
    # Tail as 17..256 (240 numbers).
    
    core_indices = list(range(0, 15)) # indices 0..14 correspond to values 2..16
    tail_indices = list(range(15, n)) # indices 15..254 correspond to values 17..256
    
    core_values = [all_candidates[i] for i in core_indices]
    tail_values = [all_candidates[i] for i in tail_indices]
    
    # Precompute conflict masks within core, within tail, and between core and tail.
    # core_conflict_mask[i] (i in 0..14) : bits set for j in 0..14 if conflict
    core_conflict = [0] * len(core_indices)
    for i in range(len(core_indices)):
        global_i = core_indices[i]
        mask = 0
        for j in range(len(core_indices)):
            if i == j: continue
            global_j = core_indices[j]
            if conflict_mask[global_i] & (1 << global_j):
                mask |= (1 << j)
        core_conflict[i] = mask
        
    # tail_conflict_mask[i] (i in 0..239) : bits set for j in 0..239 if conflict
    tail_conflict = [0] * len(tail_indices)
    for i in range(len(tail_indices)):
        global_i = tail_indices[i]
        mask = 0
        for j in range(len(tail_indices)):
            if i == j: continue
            global_j = tail_indices[j]
            if conflict_mask[global_i] & (1 << global_j):
                mask |= (1 << j)
        tail_conflict[i] = mask
        
    # cross_conflict_mask[i] (i in 0..14) : bits set for j in 0..239 if conflict
    # This is used to determine which tail nodes are available given a core selection.
    cross_conflict = [0] * len(core_indices)
    for i in range(len(core_indices)):
        global_i = core_indices[i]
        mask = 0
        for j in range(len(tail_indices)):
            global_j = tail_indices[j]
            if conflict_mask[global_i] & (1 << global_j):
                mask |= (1 << j)
        cross_conflict[i] = mask

    best_global_sum = 0.0
    best_global_set = []
    
    # Upper bound for tail: sum of all inverses in tail
    # This is a loose bound, but we can tighten it based on available nodes.
    # For B&B on core, we need a fast upper bound for the tail given a core mask.
    
    def solve_tail_greedy(available_mask):
        """
        Solves MWIS on tail given a bitmask of available nodes.
        Uses greedy initialization + limited local search.
        Returns (sum, list_of_indices_in_tail)
        """
        if available_mask == 0:
            return 0.0, []
        
        # Get list of available indices
        # Since tail size is 240, iterating is fast.
        available = []
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            available.append(idx)
            m ^= lsb
            
        # Sort by weight descending (value ascending)
        # tail_values[idx] is the number. We want small numbers (high weight).
        available.sort(key=lambda idx: tail_values[idx])
        
        current_set = []
        current_sum = 0.0
        current_mask = 0
        
        # Greedy
        for idx in available:
            # Check compatibility with current_set
            # If idx conflicts with any in current_set, skip.
            # conflict_mask_tail[idx] & current_mask == 0
            if (tail_conflict[idx] & current_mask) == 0:
                current_set.append(idx)
                current_sum += inv[tail_values[idx]]
                current_mask |= (1 << idx)
                
        # Local Search: 1-swap
        # Try to improve by swapping one out, one in.
        # Limit iterations to keep it fast.
        improved = True
        max_ls_iters = 3
        ls_iter = 0
        while improved and ls_iter < max_ls_iters and time.time() < deadline - 5.0:
            improved = False
            ls_iter += 1
            for i in range(len(current_set)):
                out_idx = current_set[i]
                # Try all available indices not in current_set
                # Optimization: only consider in_idx with weight > weight(out_idx)
                out_val = tail_values[out_idx]
                for in_idx in available:
                    if in_idx in current_set:
                        continue
                    in_val = tail_values[in_idx]
                    if in_val >= out_val:
                        continue # No gain from swap if weight is lower or equal
                    # Check if in_idx is compatible with (current_set - {out_idx})
                    # Remove out_idx from mask
                    temp_mask = current_mask & ~(1 << out_idx)
                    if (tail_conflict[in_idx] & temp_mask) == 0:
                        # Swap
                        # Update sum
                        new_sum = current_sum - inv[out_val] + inv[in_val]
                        if new_sum > current_sum:
                            # Perform swap
                            current_set[i] = in_idx
                            current_sum = new_sum
                            current_mask = temp_mask | (1 << in_idx)
                            improved = True
                            break # Restart scan? Or continue? 
                            # Breaking inner loop and restarting outer is safer but slower.
                            # Let's just break inner and continue outer to find more swaps in this pass.
                            # Actually, changing current_set[i] affects subsequent checks in this pass.
                            # It's a heuristic, so it's fine.
                
        return current_sum, current_set

    def enumerate_core(idx, core_mask, core_set_list, core_sum):
        nonlocal best_global_sum, best_global_set
        
        if time.time() > deadline:
            return
            
        # Upper Bound
        # Current core sum + Max possible tail sum
        # Tighten tail bound: only consider tail nodes compatible with current core_mask
        # available_tail_mask = ~ (union of cross_conflict for selected core nodes)
        
        blocked_tail_mask = 0
        m = core_mask
        while m:
            lsb = m & -m
            c_idx = lsb.bit_length() - 1
            blocked_tail_mask |= cross_conflict[c_idx]
            m ^= lsb
            
        # Full tail mask is (1 << len(tail_indices)) - 1
        full_tail_mask = (1 << len(tail_indices)) - 1
        available_tail_mask = full_tail_mask & ~blocked_tail_mask
        
        # Upper bound for tail: sum of inverses of available tail nodes
        # This is expensive to compute exactly every time.
        # Approximation: sum of all tail inverses. (Loose)
        # Better: Precompute prefix sums? No, mask changes.
        # Let's use a loose bound: sum of all tail inverses.
        # If this loose bound doesn't prune, we lose time.
        # Let's try a slightly tighter bound: 
        # We know that if a node is blocked, we lose its weight.
        # But computing sum of available is O(N).
        # Given N=240, and depth 15, total nodes in B&B tree is up to 2^15=32768.
        # 32768 * 240 ~ 8M ops. This is acceptable in Python if optimized? 
        # Maybe not in pure Python loops.
        
        # Let's stick to the loose bound first: max_tail_sum_global
        # If core_sum + max_tail_sum_global <= best_global_sum: return
        
        max_tail_sum_global = sum(inv[x] for x in tail_values)
        
        if core_sum + max_tail_sum_global <= best_global_sum:
            return
            
        if idx == len(core_indices):
            # Solve tail
            tail_sum, tail_set_indices = solve_tail_greedy(available_tail_mask)
            total_sum = core_sum + tail_sum
            
            if total_sum > best_global_sum:
                best_global_sum = total_sum
                # Reconstruct set
                # core_set_list contains global indices? No, local core indices.
                # We need global values.
                final_set = []
                for c_idx in core_set_list:
                    final_set.append(core_values[c_idx])
                for t_idx in tail_set_indices:
                    final_set.append(tail_values[t_idx])
                best_global_set = final_set
            return
        
        c_idx = core_indices[idx]
        c_val = core_values[idx]
        
        # Branch 1: Include c_idx
        # Check compatibility with current core
        if (core_conflict[c_idx] & core_mask) == 0:
            new_mask = core_mask | (1 << idx)
            new_list = core_set_list + [idx]
            new_sum = core_sum + inv[c_val]
            enumerate_core(idx + 1, new_mask, new_list, new_sum)
            
        # Branch 2: Exclude c_idx
        enumerate_core(idx + 1, core_mask, core_set_list, core_sum)

    # Start enumeration
    # To improve pruning, we can sort core indices by "difficulty" or "weight".
    # High weight (small value) nodes are more likely to be in the optimal solution.
    # Including them early might lead to better pruning?
    # Or excluding them?
    # Usually, including high weight nodes first helps find a good incumbent early.
    
    # Let's start with the current order (2, 3, ..., 16).
    
    # Initial incumbent: Greedy on whole graph?
    # Let's just run the B&B.
    
    enumerate_core(0, 0, [], 0.0)
    
    # If we didn't find anything (should not happen), return empty or best effort
    if not best_global_set:
        # Fallback: single element with max weight (min value)
        best_global_set = [2]
        
    return best_global_set

if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Sum: {sum(1/x for x in res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if (res[i]*res[j])//gcd(res[i], res[j]) <= 256:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print(f"Valid: {valid}")
