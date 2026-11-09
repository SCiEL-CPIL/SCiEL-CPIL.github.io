import random

def construct():
    n = 512
    max_size = 48
    
    def is_valid_set(selected, candidate):
        """
        Check if adding candidate to selected keeps the set valid.
        Uses bitset DP with Python integers for speed.
        """
        # 1. Check if candidate is the mean of any subset of selected (size >= 2)
        if len(selected) >= 2:
            others = selected
            m = len(others)
            # dp[j] is a bitset where bit s is 1 if sum s is achievable with j elements
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for j in range(m - 1, -1, -1):
                    if dp[j]:
                        dp[j + 1] |= (dp[j] << x)
            
            for k in range(2, m + 1):
                target = k * candidate
                if (dp[k] >> target) & 1:
                    return False
        
        # 2. Check if any existing element 'a' in selected becomes the mean of a subset
        #    of (selected \ {a}) U {candidate} that includes candidate.
        for a in selected:
            others = [x for x in selected if x != a]
            m = len(others)
            if m < 1:
                continue
                
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for j in range(m - 1, -1, -1):
                    if dp[j]:
                        dp[j + 1] |= (dp[j] << x)
            
            # We need subset T' of others such that:
            # c + sum(T') = (1 + |T'|) * a
            # sum(T') = (1 + k) * a - c  where k = |T'|
            # k ranges from 1 to m
            for k in range(1, m + 1):
                target_sum = (k + 1) * a - candidate
                if target_sum < 0:
                    continue
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    # Heuristic: Try multiple random orders to find a valid set of size 48
    # Since the check is fast, we can afford several attempts.
    
    best_set = []
    candidates = list(range(1, n + 1))
    
    # Try a few deterministic orders first
    orders = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
        list(range(n, 0, -2)) + list(range(n - 1, 0, -2)),
        list(range(n // 2 + 1, n + 1)) + list(range(1, n // 2 + 1))
    ]
    
    # Add a few random shuffles
    for _ in range(10):
        random.shuffle(candidates)
        orders.append(candidates[:])
    
    for order in orders:
        selected = []
        for x in order:
            if len(selected) >= max_size:
                break
            if is_valid_set(selected, x):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= max_size:
                break
    
    return best_set[:max_size]
