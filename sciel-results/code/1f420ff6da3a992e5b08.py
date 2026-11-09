import numpy as np
import time
import random

def _check_mean_condition(S, a):
    """
    Check if 'a' is the arithmetic mean of any subset of S (excluding a itself if a is in S, 
    but here we pass S without a, or S with a and we handle it).
    Actually, let's pass the list of OTHER elements.
    Condition: sum(T) = k * a for some k >= 2.
    Equivalent: sum(t - a for t in T) = 0 for some subset T of size >= 2.
    """
    if len(S) < 2:
        return True
    
    # Calculate differences
    diffs = [t - a for t in S]
    
    # Range of sums
    # Max abs diff is 512. Max elements 48.
    # Range approx 25000.
    max_sum = sum(abs(d) for d in diffs)
    if max_sum == 0:
        return True # All diffs 0, but t!=a so this shouldn't happen unless S empty or duplicates (distinct required)
        
    offset = max_sum
    size = 2 * max_sum + 1
    
    # dp[i] = min number of elements to reach sum (i - offset)
    # Initialize with a large number
    INF = 1000
    dp = np.full(size, INF, dtype=np.int32)
    dp[offset] = 0
    
    for d in diffs:
        new_dp = dp.copy()
        if d > 0:
            # new_dp[d:] = min(new_dp[d:], dp[:-d] + 1)
            new_dp[d:] = np.minimum(new_dp[d:], dp[:-d] + 1)
        elif d < 0:
            k = -d
            # new_dp[:-k] = min(new_dp[:-k], dp[k:] + 1)
            new_dp[:-k] = np.minimum(new_dp[:-k], dp[k:] + 1)
        # if d == 0, no change needed (but t!=a so d!=0)
        dp = new_dp
        
        # If 0 is reachable with count > 0, it's invalid
        # But we need count >= 2.
        # Since no single d is 0, if count > 0, it is >= 2.
        if dp[offset] > 0:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit, 10s buffer
    
    best_set = []
    
    # Try different strategies
    strategies = []
    
    # 1. Descending
    strategies.append(list(range(512, 0, -1)))
    
    # 2. Ascending
    strategies.append(list(range(1, 513)))
    
    # 3. Random shuffles
    for _ in range(5):
        cands = list(range(1, 513))
        random.shuffle(cands)
        strategies.append(cands)
        
    for order in strategies:
        if time.time() > deadline:
            break
            
        S = []
        for x in order:
            if time.time() > deadline:
                break
                
            # Check if x can be added to S
            valid = True
            
            # 1. Check if x is mean of any subset of S
            if not _check_mean_condition(S, x):
                valid = False
            
            # 2. Check if any a in S is mean of any subset of (S \ {a}) U {x}
            if valid:
                for a in S:
                    # Subsets of S \ {a} are already safe.
                    # We only need to check subsets containing x.
                    # T U {x} where T subset of S \ {a}.
                    # sum(T) + x = k * a
                    # sum(t - a for t in T) = a - x
                    target = a - x
                    others = [t for t in S if t != a]
                    
                    if len(others) < 1:
                        continue # No T possible, so T U {x} has size 1. k=1.
                        # sum = x. x = a? No, distinct.
                        # So no subset of size 1 works.
                        # We need k >= 2?
                        # The condition is "subset of at least two OTHER elements".
                        # If T is empty, subset is {x}. Size 1. Not allowed.
                        # So we need T to be non-empty?
                        # If T is non-empty, size of T U {x} is >= 2.
                        
                    # We need to check if target is reachable by a NON-EMPTY subset of others?
                    # If T is empty, sum=0. target = a-x.
                    # If a-x == 0, then a=x, impossible.
                    # So we need a non-empty subset of others that sums to target.
                    
                    # Modify _check_mean_condition to check for specific target sum?
                    # Or just reuse it?
                    # _check_mean_condition checks if 0 is reachable.
                    # Here we want to know if `target` is reachable.
                    
                    # Let's write a helper for specific target
                    # But for now, let's approximate:
                    # If target is 0, we need non-empty subset summing to 0.
                    # If target != 0, we need subset summing to target.
                    
                    # To save time, let's just check if target is reachable.
                    # If it is reachable by empty set (target=0), we need to ensure it's reachable by non-empty.
                    
                    # Re-use logic:
                    diffs = [t - a for t in others]
                    max_sum = sum(abs(d) for d in diffs)
                    if max_sum == 0:
                        continue
                    offset = max_sum
                    size = 2 * max_sum + 1
                    INF = 1000
                    dp = np.full(size, INF, dtype=np.int32)
                    dp[offset] = 0
                    
                    found = False
                    for d in diffs:
                        new_dp = dp.copy()
                        if d > 0:
                            new_dp[d:] = np.minimum(new_dp[d:], dp[:-d] + 1)
                        elif d < 0:
                            k = -d
                            new_dp[:-k] = np.minimum(new_dp[:-k], dp[k:] + 1)
                        dp = new_dp
                        
                        # Check if target is reachable
                        target_idx = offset + target
                        if 0 <= target_idx < size:
                            if dp[target_idx] > 0: # Non-empty
                                found = True
                                break
                            elif target == 0 and dp[target_idx] == 0:
                                # Reachable by empty set. Need to check if also reachable by non-empty?
                                # We continue to see if a non-empty path appears.
                                pass
                                
                    if found:
                        valid = False
                        break
                        
            if valid:
                S.append(x)
                
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
