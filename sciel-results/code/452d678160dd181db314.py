"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    def update_dp(dp, x):
        """Update DP table by adding element x. dp[k] is bitmask of sums of size k."""
        n = len(dp)
        new_dp = [0] * (n + 1)
        new_dp[0] = 1
        for k in range(1, n + 1):
            # Option 1: don't take x (from old dp[k])
            # Option 2: take x (from old dp[k-1] shifted by x)
            val = dp[k] if k < n else 0
            if k > 0:
                val |= (dp[k-1] << x)
            new_dp[k] = val
        return new_dp

    def greedy_incremental(order, max_size=48):
        """
        Greedy construction that maintains DP tables for S and S\{a} for each a in S.
        This allows O(N^2) check for adding a new element x.
        """
        selected = []
        n = 0
        dp_all = [1]  # dp_all[k] for current set S
        dp_without = {} # dp_without[a][k] for set S \ {a}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= max_size:
                break
            
            # Check 1: Is x the mean of any subset of S of size >= 2?
            valid = True
            for k in range(2, n + 1):
                target = k * x
                # Max sum of k elements from S is bounded by k * 512
                if target > 512 * k: 
                    continue
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: For each a in S, is a the mean of any subset of (S U {x}) \ {a} of size >= 2?
            # The subsets of (S U {x}) \ {a} are:
            # 1. Subsets of S \ {a} (already valid, so no violation)
            # 2. Subsets of (S \ {a}) U {x} that contain x.
            #    Let T be a subset of S \ {a} of size k-1.
            #    New subset is T U {x} of size k.
            #    Mean condition: a = (sum(T) + x) / k  => sum(T) = k*a - x.
            #    So we check if (k*a - x) is in dp_without[a][k-1].
            
            for a in selected:
                # k is the size of the subset of (S U {x}) \ {a}
                # Since it must contain x, and size >= 2, k >= 2.
                # Max k is n + 1.
                for k in range(2, n + 2):
                    target_sum = k * a - x
                    if target_sum < 0:
                        continue
                    # Max sum of k-1 elements from S\{a} is (k-1)*512
                    if target_sum > 512 * (k - 1):
                        continue
                    # Check if target_sum is achievable with k-1 elements from S\{a}
                    # dp_without[a] has size n (since S has n elements, S\{a} has n-1)
                    # So indices 0..n-1 are valid. k-1 must be <= n-1 => k <= n.
                    # Wait, if k = n+1, we need n elements from S\{a}, which has size n-1. Impossible.
                    # So k-1 <= n-1 => k <= n.
                    if k - 1 < len(dp_without[a]):
                        if (dp_without[a][k-1] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Accept x
            selected.append(x)
            n += 1
            
            # Update dp_all
            old_dp_all = dp_all
            dp_all = update_dp(dp_all, x)
            
            # Update dp_without
            # For existing a in selected[:-1] (all except x), S\{a} becomes (S_old\{a}) U {x}
            for a in selected[:-1]:
                dp_without[a] = update_dp(dp_without[a], x)
            
            # For the new x, S\{x} is S_old. So dp_without[x] is old_dp_all.
            dp_without[x] = old_dp_all
        
        return selected

    def validate_set(s):
        """Full bitset verification for safety."""
        n = len(s)
        if n < 3:
            return True
        
        dp_all = [1]
        for x in s:
            dp_all = update_dp(dp_all, x)
        
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            dp_others = [1]
            for x in others:
                dp_others = update_dp(dp_others, x)
            
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    return False
        
        return True

    best_set = []
    best_len = 0
    
    # Run multiple greedy attempts with different orderings
    for seed in range(100):
        if time.time() - start_time > deadline:
            break
        
        random.seed(seed)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # Try a few different orderings derived from the shuffle
        # 1. Random
        res = greedy_incremental(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
        
        # 2. Sorted (sometimes helps with structure)
        res_sorted = sorted(candidates)
        res = greedy_incremental(res_sorted)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            
        # 3. Reverse sorted
        res_rev = sorted(candidates, reverse=True)
        res = greedy_incremental(res_rev)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            
        if best_len >= 48:
            break

    # Final safety check
    if best_set and not validate_set(best_set):
        # If something went wrong, try to repair by removing elements
        # This should rarely happen if logic is correct
        valid_set = []
        for x in best_set:
            test = valid_set + [x]
            if validate_set(test):
                valid_set.append(x)
        best_set = valid_set

    return best_set

# EVOLVE-BLOCK-END
