import numpy as np
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds margin
    
    N = 10**7
    
    # 1. Compute sigma(n) for n=1..N
    # Using a sieve approach
    # sigma = np.zeros(N+1, dtype=np.int64)
    # for i in range(1, N+1):
    #     sigma[i::i] += i
    # This is O(N log N) but with large constant in numpy?
    # Actually, numpy slicing is fast.
    
    # Alternative: Linear sieve for sigma?
    # Linear sieve is O(N) but complex to implement in numpy efficiently without python loops for the inner part.
    # The slicing method is standard for this size in numpy.
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    # Initialize sigma[1] = 1? The loop handles it.
    # Loop i from 1 to N
    # This loop is the bottleneck. 10^7 iterations.
    # In numpy, `sigma[i::i] += i` creates a view and adds.
    # Total operations sum(N/i) = N log N.
    
    # To speed up, we can process primes separately?
    # No, the simple loop is usually fast enough for 10^7 in numpy if the machine is decent.
    # Let's estimate: 10^7 * 16 = 160MB memory access per iteration? No, it's sparse.
    # The number of elements touched in iteration i is N/i.
    # Total elements touched is N log N ~ 1.6 * 10^8.
    # 1.6 * 10^8 int64 additions. This is very fast in C/numpy.
    
    for i in range(1, N + 1):
        if time.time() > deadline - 10: # Check time periodically? 
            # This check adds overhead. 
            # Let's just run the sieve. It should take < 5s.
            pass
        sigma[i::i] += i
        
    if time.time() > deadline - 5:
        # Fallback if sieve is too slow (unlikely)
        return [11, 6]

    # 2. Group indices by sigma value
    # We want to avoid a python loop of 10^7 if possible, but it's likely the best way to build the dict.
    # We can use np.unique? No, we need the inverse indices.
    
    groups = {}
    # Iterate through 1 to N
    # To speed up, we can process in chunks or just let it run.
    # 10^7 dictionary inserts is ~0.5-1s in Python.
    
    # Optimization: Only keep groups with size >= 2?
    # We can't know size without counting.
    # Let's just build the dict.
    
    # Using a local variable for setdefault to speed up
    setdefault = groups.setdefault
    
    for i in range(1, N + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # We can filter groups: if max(vals)/min(vals) < 1.5 * (1 - epsilon) or > 1.5 * (1 + epsilon)
    # Actually, we just need to check if any pair can be close to 1.5.
    # If max(vals) < 1.5 * min(vals), then a/b < 1.5 for all pairs? No, a/b can be anything.
    # We want a/b approx 1.5. So a approx 1.5 b.
    # This implies a > b and b < a < 1.5 b roughly? Or b > a and a < 1.5 b?
    # If a/b = 1.5, a = 1.5 b.
    # So we need a and b in the same group such that a is close to 1.5 b.
    
    for s, vals in groups.items():
        if len(vals) < 2:
            continue
            
        # Sort the values
        vals.sort()
        
        # For each b in vals, find a in vals closest to 1.5 b
        # Since vals is sorted, we can use bisect
        import bisect
        
        for b in vals:
            target_a = 1.5 * b
            # Find position in vals
            pos = bisect.bisect_left(vals, target_a)
            
            # Check pos
            if pos < len(vals):
                a = vals[pos]
                if a != b: # a/b != 1
                    err = abs(a/b - 1.5)
                    if err < best_err:
                        if err == 0:
                            # Invalid, skip
                            pass
                        else:
                            best_err = err
                            best_a, best_b = a, b
            # Check pos - 1
            if pos - 1 >= 0:
                a = vals[pos - 1]
                if a != b:
                    err = abs(a/b - 1.5)
                    if err < best_err:
                        if err == 0:
                            pass
                        else:
                            best_err = err
                            best_a, best_b = a, b
                            
            # Also check if we should look for b/a approx 1.5?
            # The problem asks for |a/b - 1.5|.
            # If we swap a and b, we get |b/a - 1.5|.
            # The pair (a,b) and (b,a) are distinct solutions.
            # My loop iterates all b in vals. For each b, it finds a.
            # This covers all ordered pairs (a,b) where a,b in vals.
            
        if time.time() > deadline - 2:
            break

    return [int(best_a), int(best_b)]
