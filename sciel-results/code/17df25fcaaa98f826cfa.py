import time
import random

def verify(A):
    n = len(A)
    if n < 3:
        return True
    # To speed up, we can skip if n is small? No.
    # We check each a in A
    for i in range(n):
        a = A[i]
        # Build bitsets for A without a
        # bits[k] has bit s set if sum s is possible with k elements
        # Max sum is 48 * 512 = 24576
        # We only need to check up to k*n_max?
        # Actually, we just iterate k from 2 to n-1
        
        # Optimization: If a is very large, k*a might exceed max sum of others.
        # Max sum of others is sum(A) - a.
        # If k*a > sum(A) - a, it's impossible.
        max_sum_others = sum(A) - a
        max_k_possible = max_sum_others // a
        
        if max_k_possible < 2:
            continue # No subset of size >= 2 can sum to k*a
        
        bits = [0] * (n) # indices 0..n-1
        bits[0] = 1
        
        for j in range(n):
            if j == i: continue
            x = A[j]
            # Update bits backwards
            # We only need to update up to current count
            # But simple loop is fine
            for k in range(n - 1, 0, -1):
                if bits[k-1]:
                    bits[k] |= (bits[k-1] << x)
            # bits[0] remains 1
            
        # Check
        for k in range(2, min(max_k_possible, n - 1) + 1):
            target = k * a
            if (bits[k] >> target) & 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # 1. Greedy Ascending
    A1 = []
    for x in range(1, 513):
        if len(A1) >= 48: break
        if time.time() > deadline - 10: break
        # Check if adding x is valid
        # To save time, we can just try verify(A1+[x])
        # But verify is O(N^2).
        # We can do incremental check?
        # Just call verify.
        if verify(A1 + [x]):
            A1.append(x)
            
    if len(A1) == 48:
        return A1
        
    # 2. Greedy Descending
    A2 = []
    for x in range(512, 0, -1):
        if len(A2) >= 48: break
        if time.time() > deadline - 10: break
        if verify(A2 + [x]):
            A2.append(x)
            
    if len(A2) == 48:
        return A2
        
    # 3. Local Search
    # Pick the larger set
    A = A1 if len(A1) >= len(A2) else A2
    
    # Pad to 48
    while len(A) < 48:
        # Add random
        v = random.randint(1, 512)
        if v not in A:
            A.append(v)
            
    # Local search
    # Keep track of best valid set?
    # We just want any valid set.
    
    current_A = A[:]
    # If current is valid, return
    if verify(current_A):
        return current_A
        
    while time.time() < deadline:
        # Find a violation to guide swap?
        # For simplicity, random swap.
        # Or, find first a that is bad, swap it.
        bad_idx = -1
        for i in range(len(current_A)):
            a = current_A[i]
            # Quick check?
            # Just use verify logic to find if a is bad
            # Re-using verify logic is expensive.
            # Let's just do random swap for now.
            pass
            
        idx = random.randint(0, len(current_A) - 1)
        old_val = current_A[idx]
        new_val = random.randint(1, 512)
        while new_val in current_A:
            new_val = random.randint(1, 512)
            
        current_A[idx] = new_val
        
        if verify(current_A):
            return current_A
            
    return current_A
