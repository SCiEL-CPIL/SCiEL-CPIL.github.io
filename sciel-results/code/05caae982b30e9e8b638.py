import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 512
    max_size = 48
    
    # DP table: dp[k] is a bitset where bit s is 1 if sum s is achievable by k elements
    # We will maintain this for the current set.
    
    def build_dp(arr):
        """Build DP table for subset sums by cardinality."""
        m = len(arr)
        if m == 0:
            return [1]
        # Max sum is m * n
        max_sum = m * n
        # dp[k] is an integer representing the bitset
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                # Shift previous sums by val and OR into current
                # We only care about sums up to max_sum
                # Python ints are arbitrary precision, so this is fine
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

    def is_valid_full(arr):
        """
        Full validation for a set from scratch. Used for initial greedy or restarts.
        """
        if len(arr) < 2:
            return True
        dp = build_dp(arr)
        for i, x in enumerate(arr):
            # Create a temporary view or just use the logic above with x removed?
            # Easier: Check x against arr \ {x}
            # To avoid rebuilding DP for each x, we can use the global DP and "subtract" x?
            # No, DP doesn't support subtraction easily.
            # However, for full validation of a small set (<=48), we can just check each x against others.
            # But build_dp is O(N^2 * W). Doing it 48 times is O(48 * N^2 * W).
            # N=48, W=24576. 48 * 48 * 48 * 24576/64 ~ 3.4M ops. This is fast enough for a few checks.
            
            # Optimization: We only need to check if x is mean of subset of others.
            # We can build DP of others once per x? No, that's slow.
            
            # Alternative: Just use the incremental check logic in a loop?
            # Actually, if we build the set greedily using is_valid_with_dp, it is always valid.
            # We only need is_valid_full if we do a random restart or complex swap.
            
            # Let's just build DP of arr without x for each x.
            others = [arr[j] for j in range(len(arr)) if j != i]
            dp_others = build_dp(others)
            total_sum_others = sum(others)
            for k in range(2, len(others) + 1):
                target = k * x
                if target > total_sum_others:
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        return True

    # Greedy initialization
    def greedy_init():
        selected = []
        dp = [1] # dp for empty set
        # Order: Try numbers that are "far apart" first? Or random?
        # Let's try random shuffles with a bias towards primes or large gaps?
        # Simple: Random shuffle, add if valid.
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(selected) >= max_size:
                break
            if is_valid_with_dp(selected, dp, x):
                selected.append(x)
                # Update DP incrementally?
                # is_valid_with_dp used the old dp. We need to update dp for next iteration.
                # Let's just rebuild dp if it's fast enough, or update incrementally.
                # Incremental update:
                m = len(selected) - 1 # was m, now m+1
                # dp[k] |= (dp[k-1] << x)
                # We need to be careful with the order.
                # Since we are adding to the end, we can update dp in place if we go backwards.
                # But dp length increases.
                # Let's just rebuild for simplicity in the greedy phase, it's fast.
                dp = build_dp(selected)
        
        return selected, dp

    # Local Search
    def local_search(initial_set, initial_dp, time_limit):
        current_set = list(initial_set)
        current_dp = initial_dp
        best_set = list(current_set)
        best_size = len(current_set)
        
        # Precompute unused numbers
        used = set(current_set)
        unused = [x for x in range(1, n + 1) if x not in used]
        
        # Track "stuck" counter
        stuck_counter = 0
        
        while time.time() < time_limit:
            m = len(current_set)
            
            # 1. Try to Add
            if unused and m < max_size:
                # Randomly pick from unused
                # Bias: pick random index
                idx = random.randint(0, len(unused) - 1)
                x = unused[idx]
                
                if is_valid_with_dp(current_set, current_dp, x):
                    # Accept
                    current_set.append(x)
                    current_dp = build_dp(current_set)
                    used.add(x)
                    unused.remove(x)
                    stuck_counter = 0
                    
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    continue
            
            # 2. Try to Swap
            # Pick random element from current_set and random from unused
            if unused:
                i = random.randint(0, m - 1)
                x_remove = current_set[i]
                j = random.randint(0, len(unused) - 1)
                x_add = unused[j]
                
                # Check if (current_set - {x_remove}) + {x_add} is valid
                # We need to check validity of x_add against (current_set - {x_remove})
                # AND check if x_remove was the mean of any subset involving x_add?
                # Actually, we just need to check if the new set is valid.
                # New set = (current_set \ {x_remove}) U {x_add}
                # Let A' = current_set \ {x_remove}
                # Check 1: x_add not mean of subset of A'
                # Check 2: No a in A' is mean of subset of A' U {x_add} including x_add
                
                # We don't have DP for A'. We would need to build it.
                # Building DP for size m-1 is fast.
                A_prime = [x for idx, x in enumerate(current_set) if idx != i]
                dp_prime = build_dp(A_prime)
                
                if is_valid_with_dp(A_prime, dp_prime, x_add):
                    # Accept
                    current_set[i] = x_add
                    current_dp = build_dp(current_set)
                    used.remove(x_remove)
                    used.add(x_add)
                    unused.remove(x_add)
                    unused.append(x_remove)
                    stuck_counter = 0
                    continue
            
            # 3. Try to Remove and then Add (to escape local optima)
            # Remove a random element, then try to add a different one
            if m > 0:
                i = random.randint(0, m - 1)
                x_remove = current_set[i]
                A_prime = [x for idx, x in enumerate(current_set) if idx != i]
                dp_prime = build_dp(A_prime)
                
                # Try to add a random unused element
                if unused:
                    # Try a few random additions
                    for _ in range(10):
                        if not unused: break
                        j = random.randint(0, len(unused) - 1)
                        x_add = unused[j]
                        if is_valid_with_dp(A_prime, dp_prime, x_add):
                            # Accept the removal and addition
                            current_set = A_prime + [x_add]
                            current_dp = build_dp(current_set)
                            used.remove(x_remove)
                            used.add(x_add)
                            unused.remove(x_add)
                            unused.append(x_remove)
                            stuck_counter = 0
                            break
                    else:
                        # If we couldn't add anything, maybe just keep the removal?
                        # Or revert?
                        # If we keep the removal, size decreases.
                        # Let's just revert to avoid shrinking too much unless necessary.
                        # Actually, shrinking allows new configurations.
                        # Let's accept the removal with low probability or if stuck for long.
                        if stuck_counter > 50:
                            current_set = A_prime
                            current_dp = dp_prime
                            used.remove(x_remove)
                            unused.append(x_remove)
                            stuck_counter = 0
        
            stuck_counter += 1
            
            # If stuck for too long, do a small random restart of the local search state
            # by shuffling the order of unused or resetting stuck counter
            if stuck_counter > 100:
                random.shuffle(unused)
                stuck_counter = 0
                
        return best_set

    # Main Strategy: Multiple Greedy Inits + Local Search
    best_overall = []
    best_size = 0
    
    # Run several greedy inits with different random seeds
    for i in range(20):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(i * 100 + 42)
        initial_set, initial_dp = greedy_init()
        
        if len(initial_set) > best_size:
            best_size = len(initial_set)
            best_overall = list(initial_set)
            
        # Run local search on this initial set
        # Allocate remaining time / remaining trials
        time_left = deadline - time.time()
        # Heuristic: give more time to better initial sets
        weight = len(initial_set) / 40.0
        time_alloc = min(5.0, max(1.0, time_left * weight / 10.0))
        
        improved_set = local_search(initial_set, initial_dp, time.time() + time_alloc)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_overall = improved_set
            
        if best_size >= max_size:
            return best_overall

    # Final check: try to fill best_overall with any remaining valid numbers
    if len(best_overall) < max_size:
        used = set(best_overall)
        dp = build_dp(best_overall)
        for x in range(1, n + 1):
            if x not in used:
                if is_valid_with_dp(best_overall, dp, x):
                    best_overall.append(x)
                    used.add(x)
                    dp = build_dp(best_overall)
                    if len(best_overall) >= max_size:
                        break

    return best_overall
