import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    N = 512
    MAX_CARD = 48
    MAX_SUM = N * MAX_CARD  # 512 * 48 = 24576
    
    # Precompute nothing; we'll use Python big-int bitsets for subset sums.
    # dp[k] is a bitmask where bit s is set iff some k-element subset of the
    # current set sums to s.
    
    def build_dp(S):
        """Build subset-sum DP for set S (list of ints)."""
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for val in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << val)
        return dp
    
    def check_valid_add(S, dp_S, x):
        """
        Check if x can be added to S (with DP dp_S for S).
        S is a list, dp_S is the DP array for S.
        Returns True if valid.
        """
        n = len(S)
        
        # Check 1: x is not the mean of any subset of S of size >= 2
        # Mean of subset T (|T|=k, k>=2) equals x iff sum(T) = k*x
        for k in range(2, n + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if (dp_S[k] >> target) & 1:
                return False
        
        # Check 2: For each a in S, a is not the mean of any subset of (S\{a}) ∪ {x}
        # that has size >= 2 and includes x (or not, but we need to check all subsets
        # of (S\{a}) ∪ {x} of size >= 2 that don't include a).
        # Actually, the condition is: no element of the final set is the mean of any
        # subset of at least two OTHER elements. So for the new set S' = S ∪ {x}:
        #   - x is not mean of any subset of S of size >= 2 (checked above)
        #   - For each a in S: a is not mean of any subset of (S' \ {a}) of size >= 2
        #     S' \ {a} = (S \ {a}) ∪ {x}
        #     Subsets of (S \ {a}) ∪ {x} of size >= 2 can:
        #       a) not include x: subset of S\{a} of size >= 2 -> already valid in S
        #       b) include x: subset is {x} ∪ T where T ⊆ S\{a}, |{x}∪T| >= 2 => |T| >= 1
        #         Sum = x + sum(T), size = 1 + |T|
        #         Mean = a => x + sum(T) = (1+|T|)*a => sum(T) = (1+|T|)*a - x
        #         Let j = |T|, j >= 1, j <= n-1
        #         sum(T) = (j+1)*a - x
        #         Need to check if there exists a j-element subset T of S\{a} with sum (j+1)*a - x
        
        for i, a in enumerate(S):
            # Build DP for S \ {a}
            R = S[:i] + S[i+1:]
            m = len(R)
            if m == 0:
                continue
            dp_R = build_dp(R)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0 or target > MAX_SUM:
                    continue
                if (dp_R[j] >> target) & 1:
                    return False
        
        return True
    
    # Greedy construction with multiple orderings
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending order (large numbers first, fewer mean conflicts)
    candidates = list(range(512, 0, -1))
    current = []
    S_set = set()
    
    for x in candidates:
        if len(current) >= MAX_CARD:
            break
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        
        # Quick check: is x valid to add?
        if not current:
            current.append(x)
            S_set.add(x)
        else:
            dp_S = build_dp(current)
            if check_valid_add(current, dp_S, x):
                current.append(x)
                S_set.add(x)
    
    if len(current) > best_size:
        best_size = len(current)
        best_set = list(current)
    
    if best_size >= MAX_CARD:
        return best_set
    
    # Strategy 2: Randomized shuffles
    for seed in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        
        current = []
        S_set = set()
        
        for x in candidates:
            if len(current) >= MAX_CARD:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            if not current:
                current.append(x)
                S_set.add(x)
            else:
                dp_S = build_dp(current)
                if check_valid_add(current, dp_S, x):
                    current.append(x)
                    S_set.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
            
        if best_size >= MAX_CARD:
            return best_set
    
    # Local search: try to add more elements to the best set found
    if best_set and best_size < MAX_CARD and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng = random.Random(int(time.time() * 1000) % (2**31))
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= MAX_CARD:
                    break
                dp_S = build_dp(S)
                if check_valid_add(S, dp_S, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= MAX_CARD:
                break
            
            # Try swapping one element out for another
            if not improved:
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    a = S[i]
                    S_temp = S[:i] + S[i+1:]
                    for x in available:
                        if not S_temp:
                            S = [x]
                            S_set = {x}
                            available.remove(x)
                            available.append(a)
                            improved = True
                            break
                        dp_temp = build_dp(S_temp)
                        if check_valid_add(S_temp, dp_temp, x):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(a)
                            improved = True
                            break
                    if improved:
                        break
                    if time.time() > deadline:
                        break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return best_set
