import time
import random

def construct():
    start = time.time()
    deadline = start + 50.0

    max_n = 512
    max_size = 48

    # Precompute subset-sum bitsets for each cardinality k
    # dp[k] is a bitset where bit s is 1 if some subset of size k has sum s
    # We'll build these incrementally as we add elements.
    
    def build_dp(S):
        """Build dp table for set S."""
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
        return dp

    def check_valid(x, S, dp):
        """
        Check if x can be added to S.
        Returns True if valid.
        """
        n_s = len(S)
        
        # Check 1: x is not the mean of any subset T of S with |T| >= 2
        # Need sum(T) = x * |T| for some |T| in [2, n_s]
        for k in range(2, n_s + 1):
            target = x * k
            if target > max_n * max_size:  # sanity
                continue
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: No y in S becomes mean of subset involving x
        # y = mean(T ∪ {x}) where T ⊆ S\{y}, |T| = j >= 1
        # sum(T) + x = y * (j+1) => sum(T) = y*(j+1) - x
        # Conservative: check against full S (rejects more, but ensures validity)
        for y in S:
            for j in range(1, n_s + 1):
                target = y * (j + 1) - x
                if target < 0:
                    continue
                if target > max_n * max_size:
                    continue
                if (dp[j] >> target) & 1:
                    return False
        
        return True

    def greedy_build(order, S_init=None):
        """Build set greedily using given order."""
        S = list(S_init) if S_init else []
        if not S:
            dp = [0] * (max_size + 1)
            dp[0] = 1
        else:
            dp = build_dp(S)
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
            if x in S:
                continue
            if check_valid(x, S, dp):
                S.append(x)
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
        
        return S

    # Strategy: Try multiple greedy orders, keep best, then try to fill gaps
    best_set = []
    best_size = 0

    # Descending order
    S1 = greedy_build(range(max_n, 0, -1))
    best_set = S1
    best_size = len(best_set)

    # Ascending order  
    if time.time() < deadline - 15.0:
        S2 = greedy_build(range(1, max_n + 1))
        if len(S2) > best_size:
            best_set = S2
            best_size = len(best_set)

    # Random orders
    random.seed(42)
    while time.time() < deadline - 15.0:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S_rand = greedy_build(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break

    # Try to fill gaps in best set
    if time.time() < deadline - 5.0 and best_size < max_size:
        S_filled = list(best_set)
        dp = build_dp(S_filled)
        changed = True
        while changed and len(S_filled) < max_size:
            changed = False
            for x in range(1, max_n + 1):
                if x in S_filled:
                    continue
                if time.time() > deadline - 5.0:
                    break
                if check_valid(x, S_filled, dp):
                    S_filled.append(x)
                    for k in range(len(S_filled) - 1, -1, -1):
                        dp[k + 1] |= (dp[k] << x)
                    changed = True
                    break
        if len(S_filled) > best_size:
            best_set = S_filled
            best_size = len(best_set)

    return best_set
