import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    MAX_N = 512
    TARGET_SIZE = 48
    MAX_SUM = TARGET_SIZE * MAX_N  # 24576

    def try_greedy(order):
        """
        Attempts to build a valid set by adding elements in the given order.
        Returns the set found.
        """
        selected = []
        n = 0
        # dp[k] is a bitmask where bit s is set if sum s is achievable with exactly k elements from 'selected'
        # dp[0] = 1 (bit 0 set)
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # dp_excl[a] is a list of bitmasks for sums achievable with k elements from 'selected' excluding 'a'
        dp_excl = {}

        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= TARGET_SIZE:
                break

            # Check 1: Is x the mean of any subset of 'selected' of size k >= 2?
            valid = True
            for k in range(2, n + 1):
                target_sum = k * x
                if target_sum > MAX_SUM:
                    break
                if (dp[k] >> target_sum) & 1:
                    valid = False
                    break

            if not valid:
                continue

            # Check 2: For each a in selected, is a the mean of a subset of (selected \ {a}) U {x} of size >= 2?
            # A subset of (selected \ {a}) U {x} of size k >= 2 that includes x and sums to k*a
            # is equivalent to a subset of (selected \ {a}) of size k-1 >= 1 summing to k*a - x.
            # Let j = k-1. Then j >= 1 and sum = (j+1)*a - x.
            for a in selected:
                if not valid:
                    break
                # We need to check subsets of size j >= 1 from selected\{a}
                # Max j is n-1
                for j in range(1, n):
                    target_sum = (j + 1) * a - x
                    if target_sum < 0 or target_sum > MAX_SUM:
                        continue
                    if (dp_excl[a][j] >> target_sum) & 1:
                        valid = False
                        break

            if not valid:
                continue

            # x is valid. Add it to selected.
            selected.append(x)
            
            # Update dp
            new_dp = [0] * (n + 2)
            new_dp[0] = 1
            for k in range(1, n + 2):
                val = dp[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp[k - 1] << x)
                new_dp[k] = val
            dp = new_dp

            # Update dp_excl for existing elements
            new_dp_excl = {}
            for a in selected[:-1]:  # All except x
                old_dp_w = dp_excl[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_excl[a] = new_list

            # For new element x, dp_excl[x] is the state of dp BEFORE adding x
            # dp has length n+1 (indices 0..n). We need length n+2 for consistency in future updates?
            # Actually, dp_excl[x] will be used when n becomes n+1.
            # The list length should be at least n+1 (indices 0..n).
            # Let's just copy dp (which has length n+1) and pad if necessary, but usually we just use it as is.
            # In the next iteration, n will be n+1. The checks will go up to j=n.
            # dp_excl[x] needs to support indices up to n.
            # Current dp has length n+1. Indices 0..n.
            new_dp_excl[x] = dp[:]

            dp_excl = new_dp_excl
            n += 1

        return selected

    best_set = []
    best_len = 0

    # Strategy 1: Ascending
    res = try_greedy(range(1, MAX_N + 1))
    if len(res) > best_len:
        best_set = res
        best_len = len(res)

    # Strategy 2: Descending
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(MAX_N, 0, -1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 3: Randomized Restarts
    # We use multiple random seeds and shuffling strategies
    attempts = 0
    while time.time() - start_time < deadline - 2:
        attempts += 1
        seed = attempts * 104729 + 7919
        random.seed(seed)
        
        candidates = list(range(1, MAX_N + 1))
        
        # Different shuffling strategies
        if attempts % 3 == 0:
            random.shuffle(candidates)
        elif attempts % 3 == 1:
            # Sort by random hash
            candidates.sort(key=lambda x: hash((x, seed)))
        else:
            # Weighted shuffle: prefer numbers that are less likely to be means
            # Heuristic: numbers with fewer divisors or larger magnitude
            # Let's just do a random subset then shuffle to add variety
            subset_size = random.randint(300, 512)
            subset = random.sample(candidates, subset_size)
            random.shuffle(subset)
            candidates = subset

        res = try_greedy(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= TARGET_SIZE:
                break

    # Strategy 4: Local Improvement / Greedy Fill
    # If we have a good set, try to fill remaining slots with specific candidates
    if best_len < TARGET_SIZE and time.time() - start_time < deadline - 1:
        used = set(best_set)
        remaining = [x for x in range(1, MAX_N + 1) if x not in used]
        
        # Try adding remaining elements in a few orders
        for order in [remaining, list(reversed(remaining))]:
            # We need to rebuild the DP state from best_set to use try_greedy logic properly,
            # or write a helper that takes an initial set.
            # For simplicity and speed, we just run try_greedy on a combined list where best_set elements are prioritized?
            # No, try_greedy starts from empty.
            # Let's create a helper to extend a set.
            
            def extend_set(initial_set, candidates):
                selected = list(initial_set)
                n = len(selected)
                if n >= TARGET_SIZE:
                    return selected
                
                dp = [0] * (n + 1)
                dp[0] = 1
                for val in selected:
                    for k in range(n, 0, -1):
                        if k - 1 >= 0:
                            dp[k] |= (dp[k - 1] << val)
                
                dp_excl = {}
                for i, a in enumerate(selected):
                    dp_w = [0] * (n)
                    dp_w[0] = 1
                    for val in selected:
                        if val == a:
                            continue
                        for k in range(n - 1, 0, -1):
                            if k - 1 >= 0:
                                dp_w[k] |= (dp_w[k - 1] << val)
                    dp_excl[a] = dp_w + [0] * (n + 1 - n) # Pad to length n+1? No, length is n. Indices 0..n-1.
                    # Wait, dp_excl[a] needs to support j up to n-1 in the check loop.
                    # In try_greedy, when n is current size, j goes up to n-1.
                    # So dp_excl[a] needs length at least n.
                    # My dp_w construction above creates length n. Indices 0..n-1. This is correct.
                    
                for x in candidates:
                    if time.time() - start_time > deadline - 0.5:
                        break
                    if n >= TARGET_SIZE:
                        break
                    
                    valid = True
                    for k in range(2, n + 1):
                        target_sum = k * x
                        if target_sum > MAX_SUM:
                            break
                        if (dp[k] >> target_sum) & 1:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
                    for a in selected:
                        if not valid:
                            break
                        for j in range(1, n):
                            target_sum = (j + 1) * a - x
                            if target_sum < 0 or target_sum > MAX_SUM:
                                continue
                            if (dp_excl[a][j] >> target_sum) & 1:
                                valid = False
                                break
                                
                    if not valid:
                        continue
                        
                    selected.append(x)
                    
                    new_dp = [0] * (n + 2)
                    new_dp[0] = 1
                    for k in range(1, n + 2):
                        val = dp[k] if k <= n else 0
                        if k - 1 >= 0 and k - 1 <= n:
                            val |= (dp[k - 1] << x)
                        new_dp[k] = val
                    dp = new_dp
                    
                    new_dp_excl = {}
                    for a in selected[:-1]:
                        old_dp_w = dp_excl[a]
                        new_list = [0] * (n + 2)
                        new_list[0] = 1
                        for k in range(1, n + 2):
                            val = old_dp_w[k] if k <= n - 1 else 0
                            if k - 1 >= 0 and k - 1 <= n - 1:
                                val |= (old_dp_w[k - 1] << x)
                            new_list[k] = val
                        new_dp_excl[a] = new_list
                        
                    new_dp_excl[x] = dp[:]
                    dp_excl = new_dp_excl
                    n += 1
                    
                return selected

            improved = extend_set(best_set, order)
            if len(improved) > best_len:
                best_set = improved
                best_len = len(improved)
                if best_len >= TARGET_SIZE:
                    break

    return best_set
