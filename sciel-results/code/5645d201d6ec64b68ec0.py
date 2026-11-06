import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 512
    target_size = 48
    
    # Bitset DP for subset sums
    # dp[k] is a bitmask where bit j is 1 if sum j is achievable with k elements
    def _subset_sums_dp(arr):
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp

    def _is_valid(S):
        """
        Check if no element in S is the mean of a subset of >=2 other elements.
        S is a list or set of integers.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True
        
        # For each element a, check if a is mean of subset of S\{a}
        for i, a in enumerate(S_list):
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
            
            # Compute subset sums of others
            dp = _subset_sums_dp(others)
            sum_others = sum(others)
            
            # Check k from 2 to len(others)
            # Target sum = k * a
            # We need (dp[k] >> (k*a)) & 1 to be 1
            for k in range(2, len(others) + 1):
                target = k * a
                if target > sum_others:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def _greedy_init():
        """
        Fast greedy to get a starting valid set.
        Order: Try numbers that are less likely to be means.
        Large numbers are less likely to be means of smaller numbers?
        Or numbers with many divisors?
        Let's try descending order first, then random.
        """
        # Try a few deterministic orders
        orders = [
            list(range(n, 0, -1)),
            list(range(1, n + 1)),
        ]
        
        # Add a random order
        random.seed(42)
        rand_order = list(range(1, n + 1))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
        best_set = []
        
        for order in orders:
            S = []
            for x in order:
                if len(S) >= target_size:
                    break
                # Check if x can be added
                # 1. x is not mean of subset of S
                # 2. No a in S is mean of subset of S U {x} including x
                
                # For speed in init, we just do full check on S U {x}
                # Since S is small initially, this is fine.
                new_S = S + [x]
                if _is_valid(new_S):
                    S = new_S
            
            if len(S) > len(best_set):
                best_set = S
                if len(best_set) == target_size:
                    break
        
        return best_set

    # Initialize
    current_set = _greedy_init()
    best_set = list(current_set)
    best_size = len(best_set)
    
    if best_size == target_size:
        return best_set

    # Local Search
    # We maintain current_set as a list for easy manipulation
    # We will use a set for O(1) membership checks for candidates
    
    # Pre-compute all candidates
    all_numbers = set(range(1, n + 1))
    
    # State
    cur_set = set(current_set)
    cur_list = list(cur_set)
    
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        # Try to add
        added = False
        # Shuffle candidates to randomize
        candidates = [x for x in all_numbers if x not in cur_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if len(cur_set) >= target_size:
                break
            new_set = cur_set | {c}
            if _is_valid(new_set):
                cur_set = new_set
                cur_list = list(cur_set)
                added = True
                if len(cur_set) > best_size:
                    best_size = len(cur_set)
                    best_set = list(cur_set)
                    no_improve_count = 0
                    if best_size == target_size:
                        return best_set
                else:
                    # Size didn't increase? Impossible if we added.
                    # Wait, if we added, size increased by 1.
                    pass
                break
        
        if added:
            continue
        
        # If no addition, try swap
        # Remove one, add one
        if not cur_list:
            break
            
        # Pick random element to remove
        idx = random.randint(0, len(cur_list) - 1)
        r = cur_list[idx]
        
        # Pick random candidate to add
        candidates = [x for x in all_numbers if x not in cur_set]
        if not candidates:
            break
        random.shuffle(candidates)
        
        # Try a few swaps
        swap_done = False
        # Limit swaps per step to avoid wasting time
        max_swaps_to_try = 100
        for _ in range(max_swaps_to_try):
            if not candidates:
                break
            c = candidates.pop()
            
            new_set = (cur_set - {r}) | {c}
            if _is_valid(new_set):
                cur_set = new_set
                cur_list = list(cur_set)
                swap_done = True
                break
        
        if not swap_done:
            # If we couldn't find a valid swap, maybe we are stuck.
            # Try a "kick": remove 2, add 1? Or just restart greedy from scratch?
            # Let's try a few random restarts of the greedy if stuck for too long.
            no_improve_count += 1
            if no_improve_count > 50:
                # Restart with a new random seed
                random.seed(int(time.time() * 1000) % 10000)
                new_init = _greedy_init()
                if len(new_init) > len(cur_set):
                    cur_set = set(new_init)
                    cur_list = list(cur_set)
                    no_improve_count = 0
                else:
                    # Perturb current set
                    # Remove a random element to break symmetry
                    if cur_list:
                        r = random.choice(cur_list)
                        cur_set.remove(r)
                        cur_list = list(cur_set)
                    no_improve_count = 0

    return best_set
