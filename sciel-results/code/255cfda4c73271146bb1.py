import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def dp_subset_sums(elements):
        n = len(elements)
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in elements:
            old_dp = dp[:]
            for k in range(n - 1, -1, -1):
                if old_dp[k]:
                    dp[k+1] |= (old_dp[k] << x)
        return dp

    def try_add(S, x):
        if x in S:
            return False
        
        # Check 1: Is x a mean of a subset of S?
        if S:
            dp_S = dp_subset_sums(S)
            for k in range(2, len(S) + 1):
                target = k * x
                if (dp_S[k] >> target) & 1:
                    return False
        
        # Check 2: Does adding x make any a in S a mean?
        for a in S:
            others = [y for y in S if y != a and y != x]
            if not others:
                continue
            dp_others = dp_subset_sums(others)
            for j in range(1, len(others) + 1):
                k = j + 1
                target = k * a - x
                if target < 0:
                    continue
                if (dp_others[j] >> target) & 1:
                    return False
        return True

    best_S = set()
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        S = set()
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
            if try_add(S, x):
                S.add(x)
        
        if len(S) > len(best_S):
            best_S = S.copy()
            if len(best_S) == 48:
                break
                
    return sorted(list(best_S))
