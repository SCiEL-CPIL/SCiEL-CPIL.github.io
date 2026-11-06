import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    N = 512
    MAX_SIZE = 48
    
    # We will try multiple random restarts to find a larger set
    best_set = []
    
    for restart in range(20):  # Try up to 20 restarts
        if time.time() > deadline:
            break
            
        random.seed(restart)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected = []
        
        # DP state for 'selected'
        # dp[k] is a bitset of sums reachable with exactly k elements from 'selected'
        # Max sum: 48 * 512 = 24576
        # We use Python integers as bitsets.
        # Bit i represents sum i.
        # We need to handle sums up to ~25000.
        
        # Initialize DP
        # dp[0] = 1 (sum 0)
        # We'll maintain a list of bitsets.
        # To avoid reallocation, we can just use a list of ints.
        
        max_k = 0
        dp = [0] * (MAX_SIZE + 1)
        dp[0] = 1  # Sum 0 is reachable with 0 elements
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(selected) >= MAX_SIZE:
                break
                
            # 1. Check if x is a mean of a subset of 'selected'
            # x = (sum(T)) / |T| => sum(T) = |T| * x
            # T is subset of selected, |T| >= 2
            valid_x = True
            for k in range(2, len(selected) + 1):
                target_sum = k * x
                # Check if target_sum is in dp[k]
                if (dp[k] >> target_sum) & 1:
                    valid_x = False
                    break
            
            if not valid_x:
                continue
            
            # 2. Check if any y in selected becomes a mean of (selected U {x}) \ {y}
            # y = (sum(T) + x) / (|T| + 1)
            # sum(T) = (|T| + 1) * y - x
            # T is subset of selected \ {y}, |T| >= 1
            # Let K = |T| + 1. K ranges from 2 to len(selected)
            # sum(T) = K * y - x
            # T is subset of selected \ {y}
            
            valid_set = True
            for y in selected:
                # We need to check if K*y - x is a subset sum of selected \ {y}
                # for any K in [2, len(selected)]
                
                # To check subset sums of selected \ {y}, we would ideally have a DP without y.
                # We don't. We have to compute it or approximate.
                # Given the small size, let's compute the subset sums of selected \ {y} on the fly?
                # That's expensive.
                
                # Alternative:
                # If we just check condition 1, we might get an invalid set.
                # But maybe we can just try to add x, and then verify the whole set at the end?
                # No, we need to ensure validity to maximize size.
                
                # Let's use a heuristic:
                # If x is large, K*y - x might be negative or small.
                
                # For now, let's skip the check for y to speed up, and rely on the fact that
                # random greedy often works?
                # No, that's risky.
                
                # Let's do the check but optimize it.
                # We can precompute the subset sums of 'selected' once?
                # We already have dp for 'selected'.
                # Can we derive subset sums of selected \ {y} from dp?
                # No.
                
                # Let's just do the DP for selected \ {y} if the set is small enough.
                # Since we are in a loop, this is O(|S|^2 * DP).
                
                # Optimization: Only check y if x is "close" to y?
                # No.
                
                # Let's try a different strategy:
                # Build the set, and if it becomes invalid, remove the last element?
                # This is backtracking.
                
                # For this iteration, let's assume the check for y is expensive and skip it?
                # If we skip it, the set might be invalid.
                # The verifier will reject it. Score 0.
                
                # We MUST check.
                # Let's implement the check for y efficiently.
                # We can compute the subset sum bitset for `selected` excluding `y` by
                # iterating through `selected` and skipping `y`.
                
                # But doing this for every y in every step is too slow.
                
                # Alternative:
                # Only check y if x < 2*y?
                # If x is very large, K*y - x is likely negative for small K.
                # Max sum of T is sum(selected \ {y}).
                # If K*y - x > sum(selected \ {y}), then impossible.
                
                sum_others = sum(selected) - y
                found_invalid = False
                for K in range(2, len(selected) + 1):
                    target = K * y - x
                    if target < 0:
                        continue
                    if target > sum_others:
                        continue
                    # Check if target is a subset sum of selected \ {y}
                    # We need a DP for selected \ {y}
                    # Let's compute it once per y?
                    # This is still heavy.
                    
                    # Approximation:
                    # If we don't check, we risk invalidity.
                    # Let's try to check only for K=2?
                    # No, K can be larger.
                    
                    # Let's just do the full check for y using a fresh DP for selected \ {y}
                    # but only if the set is small (< 20).
                    # If set is large, we might skip and hope?
                    # No.
                    
                    # Let's use the fact that we are in a restart loop.
                    # We can afford to be slower if we find a better set.
                    
                    # Compute subset sums of selected \ {y}
                    dp_temp = [0] * (len(selected) + 1)
                    dp_temp[0] = 1
                    for z in selected:
                        if z == y:
                            continue
                        for k in range(len(selected) - 1, -1, -1):
                            if dp_temp[k]:
                                dp_temp[k+1] |= dp_temp[k] << z
                        # Early break if target found?
                        # Check if target is in any dp_temp[k]
                        # We need to know which k.
                        # The target sum is for a subset T of size K-1.
                        # So we check dp_temp[K-1].
                        
                    # Check if target is in dp_temp[K-1]
                    if (dp_temp[K-1] >> target) & 1:
                        found_invalid = True
                        break
                if found_invalid:
                    valid_set = False
                    break
            
            if valid_set:
                # Add x to selected
                selected.append(x)
                # Update DP
                for k in range(len(selected) - 1, -1, -1):
                    dp[k+1] |= dp[k] << x
                
                if len(selected) > len(best_set):
                    best_set = selected[:]
        
        # End restart
    
    return best_set
