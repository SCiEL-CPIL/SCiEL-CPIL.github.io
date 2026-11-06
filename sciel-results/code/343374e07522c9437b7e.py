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
        for i, x in enumerate(arr):
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

    # Greedy initialization with heuristics
    def greedy_init(strategy):
        """
        Greedy initialization with different strategies.
        strategy: 'random', 'edge', 'safe'
        """
        selected = []
        dp = [1] # dp for empty set
        candidates = list(range(1, n + 1))
        
        if strategy == 'random':
            random.shuffle(candidates)
        elif strategy == 'edge':
            # Prefer numbers near the edges (1, 2, ..., and 511, 512, ...)
            # Sort by distance from center (256.5)
            center = (1 + n) / 2.0
            candidates.sort(key=lambda x: -abs(x - center))
            # Add some randomness
            for i in range(len(candidates) - 1, 0, -1):
                j = random.randint(0, i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        elif strategy == 'safe':
            # Prefer numbers that are less likely to be means
            # Heuristic: prefer numbers with fewer divisors or prime numbers
            # For simplicity, prefer odd numbers (less likely to be mean of two integers)
            # and numbers with small number of factors
            def score(x):
                # Score: lower is better
                # Prefer odd
                if x % 2 == 0:
                    s = 10
                else:
                    s = 0
                # Prefer primes (harder to be a mean)
                if x > 1:
                    is_prime = True
                    for p in range(2, int(x**0.5) + 1):
                        if x % p == 0:
                            is_prime = False
                            break
                    if is_prime:
                        s -= 5
                return s
            
            # Shuffle first, then sort by score with randomness
            random.shuffle(candidates)
            candidates.sort(key=lambda x: (score(x), random.random()))
        
        for x in candidates:
            if len(selected) >= max_size:
                break
            if is_valid_with_dp(selected, dp, x):
                selected.append(x)
                dp = build_dp(selected)
        
        return selected, dp

    # Local Search with DP caching
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
        
        # Cache for DPs of subsets
        dp_cache = {}
        
        def get_dp_prime(removed_idx):
            """Get DP for current_set with element at removed_idx removed."""
            key = (tuple(current_set), removed_idx)
            if key in dp_cache:
                return dp_cache[key]
            
            A_prime = [x for idx, x in enumerate(current_set) if idx != removed_idx]
            dp_prime = build_dp(A_prime)
            dp_cache[key] = dp_prime
            # Limit cache size
            if len(dp_cache) > 2000:
                dp_cache.clear()
                dp_cache[key] = dp_prime
            return dp_prime
        
        while time.time() < time_limit:
            m = len(current_set)
            
            # 1. Try to Add
            if unused and m < max_size:
                idx = random.randint(0, len(unused) - 1)
                x = unused[idx]
                
                if is_valid_with_dp(current_set, current_dp, x):
                    current_set.append(x)
                    current_dp = build_dp(current_set)
                    used.add(x)
                    unused.remove(x)
                    stuck_counter = 0
                    dp_cache.clear()
                    
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    continue
            
            # 2. Try to Swap
            if unused:
                i = random.randint(0, m - 1)
                x_remove = current_set[i]
                j = random.randint(0, len(unused) - 1)
                x_add = unused[j]
                
                dp_prime = get_dp_prime(i)
                A_prime = current_set[:i] + current_set[i+1:]
                
                if is_valid_with_dp(A_prime, dp_prime, x_add):
                    current_set[i] = x_add
                    current_dp = build_dp(current_set)
                    used.remove(x_remove)
                    used.add(x_add)
                    unused.remove(x_add)
                    unused.append(x_remove)
                    stuck_counter = 0
                    dp_cache.clear()
                    continue
            
            # 3. Try to Remove and Add (escape local optima)
            if m > 0 and unused:
                i = random.randint(0, m - 1)
                x_remove = current_set[i]
                A_prime = current_set[:i] + current_set[i+1:]
                dp_prime = get_dp_prime(i)
                
                found = False
                for _ in range(20):
                    if not unused: break
                    j = random.randint(0, len(unused) - 1)
                    x_add = unused[j]
                    if is_valid_with_dp(A_prime, dp_prime, x_add):
                        current_set = A_prime + [x_add]
                        current_dp = build_dp(current_set)
                        used.remove(x_remove)
                        used.add(x_add)
                        unused.remove(x_add)
                        unused.append(x_remove)
                        stuck_counter = 0
                        dp_cache.clear()
                        found = True
                        break
                
                if not found and stuck_counter > 30:
                    # Accept removal to change structure
                    current_set = A_prime
                    current_dp = dp_prime
                    used.remove(x_remove)
                    unused.append(x_remove)
                    stuck_counter = 0
                    dp_cache.clear()
            
            stuck_counter += 1
            
            if stuck_counter > 100:
                random.shuffle(unused)
                stuck_counter = 0
                
        return best_set

    # Main Strategy: Multiple Greedy Inits + Local Search
    best_overall = []
    best_size = 0
    
    strategies = ['random', 'edge', 'safe']
    
    for trial in range(30):
        if time.time() > deadline - 15.0:
            break
        
        strategy = strategies[trial % len(strategies)]
        random.seed(trial * 137 + 42)
        
        initial_set, initial_dp = greedy_init(strategy)
        
        if len(initial_set) > best_size:
            best_size = len(initial_set)
            best_overall = list(initial_set)
        
        # Run local search
        time_left = deadline - time.time()
        # Allocate time based on initial quality
        weight = len(initial_set) / 40.0
        time_alloc = min(4.0, max(0.5, time_left * weight / 10.0))
        
        improved_set = local_search(initial_set, initial_dp, time.time() + time_alloc)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_overall = improved_set
            
        if best_size >= max_size:
            break

    # Final polish: try to add any remaining valid numbers
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
