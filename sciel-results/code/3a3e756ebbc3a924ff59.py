import random

def construct():
    max_n = 512
    max_size = 48
    
    # Precompute powers of 2 for bit shifts? No, just use integers.
    
    def is_valid_addition(S, x, dp_sums):
        """
        Check if x can be added to S.
        S is a list of integers.
        dp_sums is a list of bitmasks, where dp_sums[k] has bit i set if there is a subset of S of size k with sum i.
        Returns True if valid, and the updated dp_sums if valid.
        """
        # Check 1: x is not mean of any subset of S
        for k in range(2, len(S) + 1):
            # Mean is x => Sum = k * x
            target_sum = k * x
            if (dp_sums[k] >> target_sum) & 1:
                return False, None
        
        # Check 2: No a in S is mean of any subset of S U {x}
        # Subset of S U {x} that includes x has size j+1 (where j is size of subset of S)
        # Sum = sum(subset_of_S) + x
        # Mean = a => sum(subset_of_S) + x = (j+1) * a
        # sum(subset_of_S) = (j+1) * a - x
        for a in S:
            for j in range(1, len(S) + 1): # j is size of subset of S
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                # Check if there is a subset of S of size j with sum target_sum
                if (dp_sums[j] >> target_sum) & 1:
                    return False, None
            
            # Also check if a is mean of a subset of S U {x} that does NOT include x?
            # That's already covered by Check 1 for the new element? No.
            # Check 1 checks if x is mean of S.
            # Check 2 checks if a in S is mean of S U {x}.
            # If the subset does not include x, it's a subset of S.
            # If a is mean of a subset of S, it was already invalid in S.
            # So we only need to check subsets that include x.
            
        # If valid, compute new dp
        new_dp = [0] * (len(S) + 2)
        for k in range(len(S) + 1):
            new_dp[k] = dp_sums[k]
        # Add x to subsets
        for k in range(len(S), -1, -1):
            new_dp[k+1] |= (dp_sums[k] << x)
            
        return True, new_dp

    def compute_dp(S):
        dp = [0] * (len(S) + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S), 0, -1):
                if k <= len(S):
                    dp[k] |= (dp[k-1] << x)
        return dp

    # 1. Initial Greedy (Deterministic, Ascending)
    # This tends to produce a valid set quickly.
    S = []
    dp = [1] # dp[0] = 1
    
    for x in range(1, max_n + 1):
        if len(S) >= max_size:
            break
        # Check if x can be added
        valid, new_dp = is_valid_addition(S, x, dp)
        if valid:
            S.append(x)
            dp = new_dp
            
    # 2. Local Search to try to reach max_size
    # If we didn't reach max_size, try to improve.
    # We try to swap elements.
    # To make it fast, we limit the number of iterations.
    
    if len(S) < max_size:
        # Create a set of available numbers
        available = set(range(1, max_n + 1)) - set(S)
        
        # Try to add any available number
        # We'll do a few passes
        for _ in range(100): # Limit iterations
            improved = False
            # Try adding
            for x in list(available):
                valid, new_dp = is_valid_addition(S, x, dp)
                if valid:
                    S.append(x)
                    dp = new_dp
                    available.remove(x)
                    improved = True
                    if len(S) == max_size:
                        break
            if len(S) == max_size:
                break
                
            if not improved:
                # Try swapping: remove one, add two
                # This is expensive. Only do if we are close.
                if len(S) >= max_size - 5:
                    # Pick a random element to remove
                    # Or try all?
                    # Let's try a random few
                    for _ in range(10):
                        if len(S) == max_size: break
                        idx = random.randint(0, len(S)-1)
                        a = S[idx]
                        # Temporarily remove a
                        S_temp = S[:idx] + S[idx+1:]
                        dp_temp = compute_dp(S_temp)
                        
                        # Try to add 2 numbers from available
                        # Pick 2 random candidates
                        candidates = list(available)
                        random.shuffle(candidates)
                        # Try first few
                        for i in range(min(20, len(candidates))):
                            x1 = candidates[i]
                            valid1, dp1 = is_valid_addition(S_temp, x1, dp_temp)
                            if valid1:
                                # Try to add a second one
                                S_temp2 = S_temp + [x1]
                                for j in range(i+1, min(40, len(candidates))):
                                    x2 = candidates[j]
                                    valid2, dp2 = is_valid_addition(S_temp2, x2, dp1)
                                    if valid2:
                                        # Success!
                                        S = S_temp2 + [x2]
                                        dp = dp2
                                        available.remove(x1)
                                        available.remove(x2)
                                        available.add(a)
                                        improved = True
                                        break
                            if improved: break
                        if improved: break
                    if improved:
                        break
                        
    return S
