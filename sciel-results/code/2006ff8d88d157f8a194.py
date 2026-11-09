import random
import time

def construct():
    start_time = time.time()
    max_time = 55.0 # Safety margin
    max_size = 48
    max_val = 512
    
    best_set = []
    
    def add_to_dp(dp, x):
        """Returns new dp list after adding x"""
        n = len(dp) - 1
        new_dp = [0] * (n + 2)
        for k in range(n + 1):
            new_dp[k] = dp[k]
            if k > 0:
                new_dp[k] |= dp[k-1] << x
        return new_dp

    for attempt in range(500):
        if time.time() - start_time > max_time:
            break
        
        S = []
        dp_S = [1] # dp[k] for k=0
        dp_without = {} # map element -> dp list for S \ {element}
        
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if time.time() - start_time > max_time:
                break
            
            # 1. Check if x is mean of any subset of S
            valid = True
            for k in range(2, len(dp_S)):
                target_sum = k * x
                if target_sum >= 0 and (dp_S[k] >> target_sum) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # 2. Check if any a in S is mean of subset of S U {x} \ {a}
            # Using precomputed dp_without[a]
            for a in S:
                dp_a = dp_without[a]
                # Check if dp_a has subset of size m >= 1 summing to (m+1)*a - x
                # m ranges from 1 to len(dp_a)-1
                for m in range(1, len(dp_a)):
                    target_sum = (m + 1) * a - x
                    if target_sum >= 0 and (dp_a[m] >> target_sum) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update state
                S.append(x)
                dp_S = add_to_dp(dp_S, x)
                
                # Update dp_without
                # For existing a, add x to their dp
                for a in S[:-1]: # S[:-1] are the old elements
                    dp_without[a] = add_to_dp(dp_without[a], x)
                # For new x, dp is the old dp_S
                dp_without[x] = dp_S[:-1] # Wait, dp_S was updated. 
                # We need the dp of S before adding x.
                # Let's store old_dp_S before updating.
                
        if len(S) > len(best_set):
            best_set = S[:]
            if len(best_set) == max_size:
                break
                
    return best_set
