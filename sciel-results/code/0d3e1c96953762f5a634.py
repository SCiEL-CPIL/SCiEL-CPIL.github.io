import random

def construct():
    max_n = 512
    max_size = 48
    best_set = []
    
    # Helper to check if x can be added to S
    def can_add(S, x, dp):
        # Check 1: x is not a mean of any subset of S
        for k in range(2, len(S) + 1):
            if (dp[k] >> (k * x)) & 1:
                return False
        
        # Check 2: Adding x does not make any a in S a mean of a subset of S U {x}
        # This happens if there exists a in S, j >= 1, U subset of S\{a}, |U|=j, sum(U) + x = (j+1)a
        for a in S:
            # We need to check subsets of S \ {a}
            # Let's build a small DP for S \ {a} or just check directly
            # Since this is expensive, we only do it if Check 1 passed.
            # We can optimize by precomputing sums of S\{a}?
            # For now, let's do a direct check.
            S_minus_a = [y for y in S if y != a]
            # We need to check if there is a subset of S_minus_a of size j
            # such that sum = (j+1)a - x
            # Let's iterate j
            for j in range(1, len(S_minus_a) + 1):
                target = (j + 1) * a - x
                if target < 0:
                    continue
                # Check if there is a subset of S_minus_a of size j summing to target
                # Use DP for this specific check
                # dp2[k] bitmask of sums
                dp2 = [0] * (j + 1)
                dp2[0] = 1
                for y in S_minus_a:
                    for k in range(j, 0, -1):
                        dp2[k] |= (dp2[k-1] << y)
                if (dp2[j] >> target) & 1:
                    return False
        return True

    # Greedy with randomization
    for _ in range(50): # 50 runs
        candidates = list(range(1, max_n + 1))
        random.shuffle(candidates)
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if can_add(S, x, dp):
                S.append(x)
                # Update dp
                for k in range(len(S), 0, -1):
                    dp[k] |= (dp[k-1] << x)
        
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
