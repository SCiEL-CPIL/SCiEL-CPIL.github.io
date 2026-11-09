"""Optimized program for finding the longest interval with no modular inverse pairs."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008  # values are 1..1008

    # Compute modular inverses in O(N)
    # inv[1] = 1
    # inv[i] = (MOD - MOD//i) * inv[MOD%i] % MOD
    inv = [0] * (MOD + 1)
    inv[1] = 1
    for i in range(2, MOD):
        inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD

    # For each x in 1..1008, determine the pair (a, b) = (min(x, inv[x]), max(x, inv[x]))
    # We need to find the longest interval [s, e] such that no pair (a, b) is fully contained in [s, e].
    # Condition for [s, e] to be valid: for all x, NOT (s <= a_x AND b_x <= e).
    # This is equivalent to: for all x with a_x >= s, we must have b_x > e.
    # So e < min{b_x | a_x >= s}.
    # Let M[s] = min{b_x | a_x >= s}. Then max valid e for start s is M[s] - 1.
    
    # We want to maximize length L(s) = (M[s] - 1) - s + 1 = M[s] - s, subject to M[s] - 1 >= s => M[s] >= s + 1.
    # Note: If no such x exists for a given s (i.e., all a_x < s), then the set is empty. 
    # However, a_x >= 1 always. For s > 1, there are always pairs with a_x >= s? 
    # a_x can be as large as 1008 (for x=1008, a=1008).
    # If s=1008, a_x >= 1008 only for x=1008 (a=1008, b=1008). M[1008]=1008. e = 1007. 1007 < 1008, invalid.
    # If s=1007, a_x >= 1007. Pairs: (1007, inv[1007]), (1008, 1008).
    # inv[1007] = inv[-2] = inv[1007]. 1007*1007 = 1014049. 1014049 % 1009.
    # 1009 * 1005 = 1014045. 1014049 - 1014045 = 4. So 1007*4 = 4028. 4028 % 1009 = 4028 - 3*1009 = 4028-3027=1001. Not 1.
    # Actually, let's just trust the array.
    
    # We compute M[s] for s from 1008 down to 1.
    # M[s] = min(M[s+1], min{b_x | a_x == s})
    
    # Initialize min_b_for_a[s] = infinity
    INF = 10**9
    min_b_for_a = [INF] * (MOD + 2) # indices 1..1008
    
    for x in range(1, MOD):
        ix = inv[x]
        a = min(x, ix)
        b = max(x, ix)
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b

    # Compute suffix minimums M[s]
    M = [INF] * (MOD + 2)
    current_min = INF
    for s in range(MOD, 0, -1):
        if min_b_for_a[s] < current_min:
            current_min = min_b_for_a[s]
        M[s] = current_min

    best_len = 0
    best_start = 1
    
    # Check all possible starts s in 1..1008
    for s in range(1, MOD):
        # Check time
        if time.time() - start_time > DEADLINE:
            break
            
        e_max = M[s] - 1
        if e_max >= s:
            length = e_max - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    if best_len > 0:
        return [best_start, best_len]
    else:
        # Fallback: return a valid single element if possible, else [1,1] is invalid.
        # We know 2 is valid (2*505=1010=1 mod 1009? No. 2*505=1010. 1010%1009=1. So 2 and 505 are a pair.)
        # If interval is [2,2], x=2, y=2. 2*2=4 != 1. Valid.
        # So [2,1] is always a valid fallback.
        return [2, 1]

# EVOLVE-BLOCK-END
