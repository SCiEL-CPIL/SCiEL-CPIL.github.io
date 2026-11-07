import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Calculate prime powers P_p
    P = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        P[p] = power

    # Partition primes
    # A: Small primes with P_p > p (i.e., p^2 <= 256)
    # Actually, let's just use the specific set identified: 2, 3, 5, 7, 11, 13
    # Primes > 13 have P_p = p?
    # 17^2 = 289 > 256. Yes.
    A = [2, 3, 5, 7, 11, 13]
    B = [p for p in primes if p > 13 and p <= 127] # Can potentially match with A
    C = [p for p in primes if p > 127] # Isolated

    # Precompute gains for edges (a, b)
    # gain[a][b] = a*b - P[a] - P[b]
    # If a*b > 256, gain is -infinity (cannot match)
    # We only care about positive gains for the matching logic, but we can just max over all valid.
    
    # Map primes in B to indices for quick lookup
    B_idx = {b: i for i, b in enumerate(B)}
    
    # Recursion to find max weight matching between A and B
    # State: index in A, set of used indices in B
    # Since |A| is small, we can just iterate.
    
    max_gain = 0
    best_match = [None] * len(A) # best_match[i] = index in B or -1
    
    def solve(i, used_B, current_gain):
        nonlocal max_gain, best_match
        if i == len(A):
            if current_gain > max_gain:
                max_gain = current_gain
                # Copy current assignment
                # We need to track the assignment path
                # To avoid copying lists in every call, we can backtrack or just store the best
                # Given small size, copying is fine.
                best_match = current_match.copy()
            return
        
        a = A[i]
        
        # Option 1: a is unmatched
        current_match[i] = -1
        solve(i + 1, used_B, current_gain)
        
        # Option 2: a is matched with some b in B
        for j, b in enumerate(B):
            if j in used_B:
                continue
            prod = a * b
            if prod > N:
                continue
            gain = prod - P[a] - P[b]
            # It's always better to match if gain > 0? 
            # Not necessarily, if matching a prevents a better match for another a'?
            # But since we are maximizing total gain, and gains are additive,
            # and unmatched gives 0 gain, we should only match if gain > 0?
            # Actually, if gain < 0, it's better not to match.
            # But what if matching a with b (gain -5) allows c to match with d (gain 100)?
            # No, matching a with b consumes b. If b is not useful for c, it doesn't help.
            # If b is useful for c, then matching a with b blocks c.
            # So we should only consider matches with gain > 0?
            # Wait, if gain is negative, it reduces the sum.
            # The baseline is sum of P_p.
            # If we don't match, we get P_a + P_b.
            # If we match, we get ab.
            # Gain = ab - P_a - P_b.
            # If Gain < 0, we lose sum. So we should NOT match.
            if gain <= 0:
                continue
            
            used_B.add(j)
            current_match[i] = j
            solve(i + 1, used_B, current_gain + gain)
            used_B.remove(j)
            current_match[i] = -1

    # Initialize current_match
    current_match = [-1] * len(A)
    solve(0, set(), 0)

    # Construct result
    result = [1]
    
    # Add unmatched A
    for i, a in enumerate(A):
        if current_match[i] == -1:
            result.append(P[a])
            
    # Add matched A-B pairs
    for i, a in enumerate(A):
        j = current_match[i]
        if j != -1:
            b = B[j]
            result.append(a * b)
            
    # Add unmatched B (those not used in match)
    used_B_indices = set(j for j in current_match if j != -1)
    for j, b in enumerate(B):
        if j not in used_B_indices:
            result.append(P[b])
            
    # Add C (isolated primes)
    for c in C:
        result.append(P[c])

    # Verify distinctness
    if len(set(result)) != len(result):
        # Fallback to baseline
        result = [1] + [P[p] for p in primes]
    else:
        # Verify coprimality
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            result = [1] + [P[p] for p in primes]

    result = [int(x) for x in result]
    return result
