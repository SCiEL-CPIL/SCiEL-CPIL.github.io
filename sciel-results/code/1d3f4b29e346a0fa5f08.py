import time
import random
from math import gcd
from functools import reduce

# LCM of 1..24
def _lcm(a, b):
    return a * b // gcd(a, b)

LCM_24 = reduce(_lcm, range(1, 25))

# Precompute values
VALS = [0] + [LCM_24 // i for i in range(1, 25)]

def _can_form_target(target, others):
    """
    Check if 'target' can be formed by a subset sum of 'others'.
    'others' is a list of integers.
    Uses Meet-in-the-Middle.
    """
    n = len(others)
    if n == 0:
        return target == 0
    
    # Pruning
    if target < 0:
        return False
    if target == 0:
        return True
    
    total_sum = sum(others)
    if total_sum < target:
        return False
    if min(others) > target:
        return False
    
    # If n is small, simple DFS might be faster, but MITM is robust.
    # Split into two halves
    mid = n // 2
    left = others[:mid]
    right = others[mid:]
    
    def gen_sums(arr):
        sums = set([0])
        for val in arr:
            new_sums = {s + val for s in sums}
            sums.update(new_sums)
        return sums
    
    # Optimization: If one half is very small, iterate directly
    if len(left) <= 15 and len(right) <= 15:
        sums_left = gen_sums(left)
        sums_right = gen_sums(right)
        
        # Check intersection
        # For each s in sums_left, check if target - s is in sums_right
        # To speed up, make the smaller set the one we iterate over
        if len(sums_left) < len(sums_right):
            for s in sums_left:
                if target - s in sums_right:
                    return True
        else:
            for s in sums_right:
                if target - s in sums_left:
                    return True
        return False
    else:
        # Fallback for larger n (unlikely in this problem as max n=23)
        # Simple DFS with pruning
        others_sorted = sorted(others, reverse=True)
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + others_sorted[i]
            
        def dfs(idx, current_sum):
            if current_sum == target:
                return True
            if current_sum > target:
                return False
            if idx == n:
                return False
            if current_sum + (prefix_sums[n] - prefix_sums[idx]) < target:
                return False
            
            if dfs(idx + 1, current_sum + others_sorted[idx]):
                return True
            if dfs(idx + 1, current_sum):
                return True
            return False
        
        return dfs(0, 0)

def _is_valid(S):
    """
    Check if set S is reciprocal-subsum-free.
    """
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return True
    
    # For each x in S, check if 1/x is a sum of 1/y for y in S\{x}
    # In integer terms: VALS[x] is a subset sum of {VALS[y] for y in S, y != x}
    for i, x in enumerate(S_list):
        target = VALS[x]
        others = [VALS[y] for j, y in enumerate(S_list) if j != i]
        if _can_form_target(target, others):
            return False
    return True

def construct():
    deadline = time.time() + 50
    best_size = 0
    best_set = set()
    
    # Strategy:
    # 1. Try to construct a large set greedily.
    # 2. Use local search to improve.
    
    # Initial heuristic: Start with larger numbers (smaller reciprocals)
    # They are "safer" to pack densely.
    
    def try_local_search(initial_set):
        current_set = set(initial_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements
            candidates_add = [x for x in range(1, 25) if x not in current_set]
            random.shuffle(candidates_add)
            for x in candidates_add:
                if time.time() > deadline:
                    break
                new_set = current_set | {x}
                if _is_valid(new_set):
                    current_set = new_set
                    improved = True
                    break # Restart adding loop with new set
            
            # Try swapping (1 out, 1 in)
            if not improved and len(current_set) > 0:
                candidates_swap_out = list(current_set)
                random.shuffle(candidates_swap_out)
                for x_out in candidates_swap_out:
                    if time.time() > deadline:
                        break
                    candidates_swap_in = [x for x in range(1, 25) if x not in current_set]
                    random.shuffle(candidates_swap_in)
                    for x_in in candidates_swap_in:
                        new_set = (current_set - {x_out}) | {x_in}
                        if _is_valid(new_set):
                            # Only swap if it helps? Or just accept?
                            # Since we want max size, swap keeps size same.
                            # But it might open up space for additions.
                            # Let's accept swaps that lead to a "better" structure?
                            # Hard to define. Let's just accept random valid swaps.
                            current_set = new_set
                            improved = True
                            break
                    if improved:
                        break
    
        return current_set

    # Run multiple random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Random initial set
        # Bias towards larger numbers?
        # Let's try a mix.
        init_size = random.randint(10, 18)
        initial_elements = random.sample(range(1, 25), init_size)
        
        # Greedy filter to ensure it's valid start
        valid_init = set()
        for x in initial_elements:
            if _is_valid(valid_init | {x}):
                valid_init.add(x)
        
        result_set = try_local_search(valid_init)
        
        if len(result_set) > best_size:
            best_size = len(result_set)
            best_set = result_set
            
            # If we found a very large set, we can stop early?
            # Max is 24. If we find 20+, it's good.
            if best_size >= 20:
                # Continue searching for 21, 22...
                pass

    # Final check
    if not best_set:
        # Fallback: return a known small valid set
        # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} is likely valid?
        # Let's just return [1]
        return [1]

    return list(best_set)
