import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    n = 512
    max_size = 48

    # Precompute 3-term AP conflicts for heuristic
    # conflict_count[x] = number of pairs (a, c) such that a + c = 2x
    conflict_count = [0] * (n + 1)
    for x in range(1, n + 1):
        # a < x < c, a + c = 2x => c = 2x - a
        # a must be >= 1, c <= n
        # a < x => 2x - a > x => a < x
        # c <= n => 2x - a <= n => a >= 2x - n
        # Also a != x (distinct)
        min_a = max(1, 2 * x - n)
        max_a = x - 1
        if min_a <= max_a:
            conflict_count[x] = max_a - min_a + 1

    def build_dp(arr):
        """Build DP table for subset sums by cardinality."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp

    def is_valid_with_dp(arr, dp, x):
        """
        Check if x can be added to arr (with precomputed dp for arr) to form a valid set.
        Returns True if valid.
        """
        m = len(arr)
        if m == 0:
            return True
        
        total_sum = sum(arr)
        
        # Check 1: x is not the mean of any subset of arr of size >= 2
        # Subset S of arr, |S| = k >= 2, sum(S) = k * x
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: No element a in arr is the mean of a subset of arr U {x} of size >= 2 that includes x.
        # Subset T = S U {x}, S subset of arr, |T| = |S| + 1 = k >= 2 => |S| = k-1 >= 1.
        # Mean(T) = a => sum(S) + x = k * a => sum(S) = k * a - x.
        # We need to check if (k*a - x) is a sum of (k-1) elements from arr.
        for a in arr:
            # k ranges from 2 to m+1 (max size of subset from arr U {x})
            # |S| = k-1 ranges from 1 to m
            for t in range(1, m + 1):
                # t is |S|
                k = t + 1
                target = k * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                # Check if target is in dp[t]
                if (dp[t] >> target) & 1:
                    return False
        return True

    def greedy_init(seed):
        random.seed(seed)
        selected = []
        dp = [1] # dp for empty set
        
        # Create candidates sorted by conflict count (low conflict first)
        # Add some randomness to break ties or avoid deterministic traps
        candidates = list(range(1, n + 1))
        # Shuffle first to randomize tie-breaking, then stable sort by conflict count?
        # Or just sort by conflict count, then randomize the order of same-conflict items.
        # Let's just shuffle and then sort by conflict count. Python's sort is stable.
        random.shuffle(candidates)
        candidates.sort(key=lambda x: conflict_count[x])
        
        for x in candidates:
            if len(selected) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
            if is_valid_with_dp(selected, dp, x):
                selected.append(x)
                dp = build_dp(selected)
        
        return selected, dp

    def local_search(initial_set, initial_dp, time_limit):
        current_set = list(initial_set)
        current_dp = initial_dp
        best_set = list(current_set)
        best_size = len(current_set)
        
        used = set(current_set)
        unused = [x for x in range(1, n + 1) if x not in used]
        
        # Cache for DP of (current_set - {x})
        # Key: x (the removed element)
        # Value: dp_table
        # We clear this cache whenever current_set changes.
        dp_cache = {}
        
        def get_dp_prime(x_remove):
            if x_remove in dp_cache:
                return dp_cache[x_remove]
            A_prime = [x for x in current_set if x != x_remove]
            dp_prime = build_dp(A_prime)
            dp_cache[x_remove] = dp_prime
            return dp_prime

        while time.time() < time_limit:
            m = len(current_set)
            improved = False
            
            # 1. Try to Add
            if unused and m < max_size:
                # Try a few random additions
                for _ in range(5):
                    if not unused: break
                    idx = random.randint(0, len(unused) - 1)
                    x = unused[idx]
                    if is_valid_with_dp(current_set, current_dp, x):
                        current_set.append(x)
                        current_dp = build_dp(current_set)
                        used.add(x)
                        unused.remove(x)
                        dp_cache.clear()
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        improved = True
                        break
                if improved:
                    continue

            # 2. Try to Swap
            if unused and m > 0:
                # Try a few random swaps
                for _ in range(10):
                    i = random.randint(0, m - 1)
                    x_remove = current_set[i]
                    j = random.randint(0, len(unused) - 1)
                    x_add = unused[j]
                    
                    dp_prime = get_dp_prime(x_remove)
                    # Note: is_valid_with_dp expects the array without x_remove
                    # We pass the list without x_remove
                    A_prime_list = [x for idx, x in enumerate(current_set) if idx != i]
                    
                    if is_valid_with_dp(A_prime_list, dp_prime, x_add):
                        # Accept
                        current_set[i] = x_add
                        current_dp = build_dp(current_set)
                        used.remove(x_remove)
                        used.add(x_add)
                        unused.remove(x_add)
                        unused.append(x_remove)
                        dp_cache.clear()
                        improved = True
                        break
                if improved:
                    continue

            # 3. If stuck, try a random restart of the local search state?
            # Or just shuffle unused to change the random selection order
            if random.random() < 0.01:
                random.shuffle(unused)

        return best_set

    best_overall = []
    best_size = 0
    
    # Run several greedy inits + local searches
    num_trials = 50
    for i in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        initial_set, initial_dp = greedy_init(i)
        
        if len(initial_set) > best_size:
            best_size = len(initial_set)
            best_overall = list(initial_set)
            
        # Allocate time for local search
        # Give more time to larger sets
        time_left = deadline - time.time() - 5.0
        if time_left < 0: break
        
        # Heuristic: if set is large, search more
        weight = len(initial_set) / 40.0
        time_alloc = min(3.0, max(0.5, time_left * weight / 10.0))
        
        improved_set = local_search(initial_set, initial_dp, time.time() + time_alloc)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_overall = list(improved_set)
            
        if best_size >= max_size:
            break

    # Final greedy fill
    if len(best_overall) < max_size:
        used = set(best_overall)
        dp = build_dp(best_overall)
        # Try to add any remaining valid numbers
        candidates = [x for x in range(1, n + 1) if x not in used]
        random.shuffle(candidates)
        for x in candidates:
            if len(best_overall) >= max_size:
                break
            if is_valid_with_dp(best_overall, dp, x):
                best_overall.append(x)
                dp = build_dp(best_overall)

    return best_overall
