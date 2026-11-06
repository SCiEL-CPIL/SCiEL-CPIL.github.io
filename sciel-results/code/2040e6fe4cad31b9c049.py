import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 512
    max_size = 48
    candidates = list(range(1, n + 1))
    
    best_set = []
    best_size = 0
    
    max_sum = 48 * 512
    
    # Pre-allocate dp? No, it changes per restart.
    
    while time.time() < deadline:
        # Random permutation
        perm = candidates[:]
        random.shuffle(perm)
        
        S = []
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        
        current_max_sum = 0
        
        for x in perm:
            # Check validity
            valid = True
            # Optimization: if k*x > current_max_sum, break
            # Max k is len(S)
            for k in range(2, len(S) + 1):
                target_sum = k * x
                if target_sum > current_max_sum:
                    break
                if dp[target_sum] & (1 << k):
                    valid = False
                    break
            
            if valid:
                S.append(x)
                # Update DP
                # Iterate backwards
                # We can limit the range to current_max_sum
                for s in range(current_max_sum, -1, -1):
                    val = dp[s]
                    if val:
                        dp[s + x] |= val << 1
                
                current_max_sum += x
                if len(S) == max_size:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            if best_size == max_size:
                break

    return best_set
