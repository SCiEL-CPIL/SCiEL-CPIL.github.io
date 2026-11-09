import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def dfs(rem_num, rem_den, start, k, path):
        # rem = rem_num / rem_den
        if time.time() > deadline:
            return None
        
        if k == 0:
            if rem_num == 0:
                return path
            return None
        
        if k == 1:
            # rem_num / rem_den = 1 / d => d = rem_den / rem_num
            if rem_num == 0:
                return None
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= start and d <= 10000000:
                    return path + [d]
            return None

        # Pruning: 
        # 1. Max possible sum with k terms starting from 'start'
        #    Sum_{i=0}^{k-1} 1/(start+i)
        #    If rem > this sum, impossible.
        #    Approximation: k / start is an overestimate, so if rem > k/start, it's definitely impossible?
        #    No, 1/start + 1/(start+1) + ... < k/start.
        #    So if rem > k/start, it is impossible.
        if rem_num * start > rem_den * k:
            return None
            
        # 2. Min possible sum with k terms, denominators <= 10^7
        #    The smallest sum is when denominators are as large as possible: 10^7, 10^7-1, ...
        #    Sum approx k / 10^7.
        #    If rem < k / 10^7, impossible.
        if rem_num * 10000000 < rem_den * k:
            return None

        # Lower bound for next denominator d:
        # 1/d <= rem  => d >= 1/rem = rem_den/rem_num
        # Also, sum of remaining k-1 terms must be able to cover the rest?
        # A tighter lower bound:
        # rem - 1/d <= sum_{j=0}^{k-2} 1/(d+1+j) <= (k-1)/(d+1)
        # This is complex. Let's stick to:
        # d >= ceil(k * rem_den / rem_num) ?
        # Sum of k terms each <= 1/d is k/d. So rem <= k/d => d <= k/rem.
        # Wait, the *largest* term is 1/d. The sum of k terms is at most k/d.
        # So if rem > k/d, it's impossible. Thus d <= k/rem.
        # This gives an UPPER bound on d? No.
        # If we pick d too small, 1/d is large.
        # We need rem = 1/d + rest.
        # rest = rem - 1/d.
        # rest must be representable by k-1 terms > d.
        # Max value of rest is sum_{i=1}^{k-1} 1/(d+i) < (k-1)/(d+1).
        # So rem - 1/d <= (k-1)/(d+1).
        # rem <= 1/d + (k-1)/(d+1).
        # This gives a lower bound on d?
        # If d is too large, 1/d is small, and (k-1)/(d+1) is small, so the sum is small.
        # If the max possible sum for a given d is less than rem, then d is too large.
        # So we need d such that 1/d + (k-1)/(d+1) >= rem.
        # This implies d cannot be too large.
        
        # Let's use the standard bounds:
        # d_min = max(start, ceil(rem_den / rem_num))
        # d_max = floor(k * rem_den / rem_num)
        
        d_min = max(start, math.ceil(rem_den / rem_num))
        d_max = math.floor(k * rem_den / rem_num)
        
        if d_min > d_max:
            return None
            
        # Iterate d
        for d in range(d_min, d_max + 1):
            # New remainder: rem - 1/d
            # (rem_num * d - rem_den) / (rem_den * d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num < 0:
                continue
                
            if new_num == 0:
                if k == 1: # Should have been caught above
                    return path + [d]
                else:
                    # Remaining k-1 terms must sum to 0. Impossible unless k-1=0 and rem=0.
                    # But we are in k>1 branch. If new_num==0, we need k-1 terms to sum to 0.
                    # Only possible if k-1==0. But k>1. So invalid.
                    continue

            # Simplify fraction
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            res = dfs(new_num, new_den, d + 1, k - 1, path + [d])
            if res is not None:
                return res
                
        return None

    # Try increasing k
    for k in range(1, 11):
        # Initial remainder is 1/1
        res = dfs(1, 1, 5, k, [])
        if res is not None:
            return res
            
    # Fallback if no solution found in time or k range (should not happen for k up to 10)
    # The seed program finds a solution, so we can return a known valid one if DFS fails
    # But DFS should find it.
    # To be safe, return the seed's logic result if time permits?
    # No, just return a hardcoded valid solution if DFS fails?
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 ... let's check sum.
    # 1/5 + 1/6 + 1/7 = (42+35+30)/210 = 107/210
    # 1 - 107/210 = 103/210
    # 1/42 = 5/210. 103/210 - 5/210 = 98/210 = 7/15
    # 1/15? No, 15 is not > 42? Wait, denominators must be distinct and increasing in our DFS.
    # The set {5, 6, 7, 42, 140} sum:
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140
    # LCD 420
    # 84 + 70 + 60 + 10 + 3 = 227/420 != 1.
    
    # Let's just trust DFS. If it times out, it returns None.
    # We must return *something*.
    # If DFS fails, we can try to run the seed logic quickly?
    # Or just raise an error? No, must return valid.
    # The seed program returns a valid list.
    # Let's copy the seed's return if DFS fails.
    
    # Seed logic:
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no solution found")
