"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000
    WINDOW = 24
    MAX_VAL = N + WINDOW
    
    # Primes strictly below 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Step 1: Check for distinct count = 1
    # This requires s[n+i] = 1 for all i in 1..24
    # i.e., n+i is coprime to P = prod(PRIMES)
    
    # is_coprime[m] is True if m is coprime to P
    is_coprime = np.ones(MAX_VAL + 1, dtype=bool)
    indices = np.arange(MAX_VAL + 1)
    for p in PRIMES:
        is_coprime[indices % p == 0] = False
        
    # Sliding window sum of is_coprime
    # We want window of size 24 to be all True (sum == 24)
    # Window covers indices n+1 to n+24
    # Let's use cumulative sum
    prefix_sum = np.cumsum(is_coprime)
    # window_sum[i] corresponds to sum of is_coprime[i : i+24]
    # We need i such that i >= 1001 (since n >= 1000, n+1 >= 1001)
    # and i+24 <= MAX_VAL
    # prefix_sum[i+24-1] - prefix_sum[i-1] == 24
    # Let j = i-1. Sum of is_coprime[j+1 : j+25]
    # We can compute diffs
    # window_sums[k] = sum of is_coprime[k+1 : k+25] for k=0..MAX_VAL-24
    # Using prefix: P[x] = sum(is_coprime[0..x])
    # Sum(a..b) = P[b] - P[a-1]
    # Sum(i..i+23) = P[i+23] - P[i-1]
    
    # We want to find i in [1001, MAX_VAL-23] such that P[i+23] - P[i-1] == 24
    # Let's create an array of window sums for all valid i
    # i ranges from 1 to MAX_VAL-23
    # P indices range from 0 to MAX_VAL
    # We need P[i+23] and P[i-1]
    
    # Efficiently:
    # left_idx = i-1, right_idx = i+23
    # i = 1001 .. MAX_VAL-23
    # left = 1000 .. MAX_VAL-24
    # right = 1024 .. MAX_VAL
    
    left = np.arange(1000, MAX_VAL - 23) # i-1
    right = np.arange(1024, MAX_VAL + 1) # i+23
    
    window_sums = prefix_sum[right] - prefix_sum[left]
    
    # Find indices where window_sums == 24
    # The index in window_sums array corresponds to offset from start
    # k = 0 corresponds to i = 1001
    matches = np.where(window_sums == 24)[0]
    if len(matches) > 0:
        # First match is best
        k = matches[0]
        i = 1001 + k
        n = i - 1
        return n

    # Step 2: General minimization
    # Compute s[m]
    s = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        k = 1
        power = p
        while power <= MAX_VAL:
            next_power = power * p
            # mask: m % power == 0 and m % next_power != 0
            # This is equivalent to v_p(m) == k
            # We multiply s[m] by power
            mask = (indices % power == 0)
            if next_power <= MAX_VAL:
                mask &= (indices % next_power != 0)
            else:
                # If next_power > MAX_VAL, then all m divisible by power have v_p(m) >= k
                # But we want exactly k? No, we want the full power.
                # The logic "multiply by p for each factor" is additive in log space, or multiplicative in value.
                # My previous logic: s[m] *= p for each factor.
                # Here I am doing it by powers.
                # If v_p(m) = v, then s[m] should be multiplied by p^v.
                # The condition (m % p^k == 0) and (m % p^(k+1) != 0) identifies v=k.
                # So for each k, we identify numbers with v_p(m)=k and multiply by p^k.
                # This is correct.
                pass
            
            s[mask] *= power
            power = next_power
            k += 1
            
    # Now find min distinct count in sliding window
    # Use a dictionary for frequency
    freq = {}
    distinct = 0
    
    # Initialize window for n=1000
    # s[1001] to s[1024]
    for i in range(1001, 1025):
        v = int(s[i])
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
            distinct += 1
            
    best_n = 1000
    best_distinct = distinct
    
    # Slide window
    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
            
        # Remove s[n]
        v_out = int(s[n])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            distinct -= 1
            
        # Add s[n+24]
        v_in = int(s[n + 24])
        if v_in in freq:
            freq[v_in] += 1
        else:
            freq[v_in] = 1
            distinct += 1
            
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct == 1:
                return best_n
            # If we found 2, 3, etc., we keep searching for 1 or lower (impossible)
            # But we already checked for 1 globally. So min is >= 2.
            # We can break if best_distinct == 2? No, we want the absolute minimum.
            # But if 1 is impossible, 2 is the best.
            # We can add a check: if best_distinct == 2, we might still find another 2, but not better.
            # However, we should ensure we don't miss a 1 if the first check had a bug.
            # But the first check is robust.
            
    return best_n

# EVOLVE-BLOCK-END
