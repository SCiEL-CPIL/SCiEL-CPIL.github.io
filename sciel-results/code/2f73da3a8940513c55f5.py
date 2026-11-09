import time
import random

def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    def is_valid_set(S):
        n = len(S)
        if n < 3: return True
        
        # For each a in S, check if a is mean of subset of S \ {a}
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 2: continue
            
            # DP for others: dp[k] is bitset of reachable sums using k elements
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
                        
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum: break
                if (dp[k] >> target) & 1:
                    return False
        return True

    # Precompute a few candidate orderings
    # Strategy: 
    # 1. Random shuffles
    # 2. Sorted ascending
    # 3. Sorted descending
    # 4. Alternating small/large
    
    best_size = 0
    best_result = []
    
    # Try different ordering strategies
    orderings = []
    
    # Random shuffles
    for seed in range(30):
        random.seed(seed)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        orderings.append(candidates)
    
    # Sorted ascending
    orderings.append(list(range(1, 513)))
    # Sorted descending
    orderings.append(list(range(512, 0, -1)))
    # Alternating: odd then even
    odds = list(range(1, 513, 2))
    evens = list(range(2, 513, 2))
    orderings.append(odds + evens)
    orderings.append(evens + odds)
    
    for candidates in orderings:
        if time.time() > deadline: break
        
        current_set = []
        for x in candidates:
            if len(current_set) >= 48:
                break
            new_set = current_set + [x]
            if is_valid_set(new_set):
                current_set = new_set
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set
            if best_size == 48:
                break
                
    return best_result
