import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin

    # 1. Sieve for Smallest Prime Factor (SPF) up to N
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Largest Prime Factor (LPF) for each number
    # P(1) = 1 by definition in task
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_p = 1
        while n > 1:
            p = spf[n]
            max_p = p
            while n % p == 0:
                n //= p
        lpf[i] = max_p

    # 3. Get unique primes up to N
    primes = set()
    for i in range(2, N + 1):
        if spf[i] == i:
            primes.add(i)
    primes = sorted(primes)
    
    # We want to find min M such that there exists a in S_M with N-a in S_M.
    # S_M = {x in [1, N] | P(x) <= M}
    
    # Efficient strategy:
    # Iterate M through sorted primes (and 1).
    # For each M, add all x with P(x) == M to S.
    # Then check if there exists x in S such that N-x is in S.
    # To speed up the check, we can maintain a count or just check the newly added elements?
    # No, a new element x might pair with an old element y.
    # So we need to check all x in S? That's O(|S|) per M.
    # Sum of |S| over M is roughly N * (avg M)? No.
    # |S| grows.
    # If optimal M is small, |S| is small.
    
    # Let's assume optimal M is small.
    # We will iterate M.
    
    # To make it faster, we can check only if N-x is in S.
    # Membership in S is O(1).
    
    # List of primes to consider, starting from 1
    candidates_for_M = [1] + primes
    
    # Use a boolean array for O(1) membership check, which is faster than a set
    in_S = [False] * (N + 1)
    
    # We will check only the newly added elements against the existing set.
    # If x is added, and N-x is already in S, we found a pair.
    # If N-x is not in S, it might be added later.
    # When N-x is added later, it will check if x is in S (which it is).
    # So we will find the pair when the *second* of the two numbers is added.
    
    for M in candidates_for_M:
        if time.time() > deadline:
            break
            
        # Add numbers with P(x) == M to S
        if M == 1:
            # Only x=1 has P(x)=1
            if not in_S[1]:
                in_S[1] = True
                b = N - 1
                if b >= 1 and b <= N and in_S[b]:
                    return (1, b)
        else:
            # Find all x with lpf[x] == M
            # We can iterate through all x, but that's O(N) per M.
            # Better to pre-bucket.
            # But we didn't pre-bucket. Let's just iterate.
            # Actually, for small M, there are few such x.
            # But we don't know which x have lpf[x] == M without checking.
            # Let's stick to the original approach of checking all x in S,
            # but using the boolean array for faster membership.
            # Wait, the original approach was to iterate through S.
            # If we use a boolean array, we still need to know which x are in S.
            # We can maintain a list of elements in S.
            
            # Let's restart with a list to track elements in S.
            # This is getting complicated. Let's just use the set approach but with the O(1) check optimization.
            
            pass
    
    # Let's re-implement with a list to track elements in S.
    S_list = []
    in_S = [False] * (N + 1)
    
    for M in candidates_for_M:
        if time.time() > deadline:
            break
            
        # Add numbers with P(x) == M to S
        if M == 1:
            x = 1
            if not in_S[x]:
                in_S[x] = True
                S_list.append(x)
                b = N - x
                if b >= 1 and b <= N and in_S[b]:
                    return (x, b)
        else:
            # Find all x with lpf[x] == M
            # We can pre-compute buckets to avoid scanning all N numbers for each M.
            # But for small M, the number of such x is small.
            # However, finding them requires scanning or pre-computation.
            # Let's pre-compute buckets.
            pass
    
    # Pre-compute buckets
    buckets = {p: [] for p in primes}
    buckets[1] = []
    for x in range(1, N + 1):
        p = lpf[x]
        if p in buckets:
            buckets[p].append(x)
    
    # Now iterate through M
    S_list = []
    in_S = [False] * (N + 1)
    
    for M in candidates_for_M:
        if time.time() > deadline:
            break
            
        # Add numbers with P(x) == M to S
        for x in buckets.get(M, []):
            if not in_S[x]:
                in_S[x] = True
                S_list.append(x)
                b = N - x
                if b >= 1 and b <= N and in_S[b]:
                    return (x, b)
                    
    # If not found, return a default pair
    return (1, N - 1)
