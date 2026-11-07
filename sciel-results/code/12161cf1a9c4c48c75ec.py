import time
import math
import numpy as np
from collections import defaultdict
import bisect

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma(n) for all n <= MAX_N using linear sieve
    # sigma is multiplicative.
    # We track:
    # lp[n]: largest prime power of the smallest prime factor of n
    # sp[n]: sigma(lp[n])
    # sigma[n]: sum of divisors of n
    
    n_max = MAX_N
    sigma = np.zeros(n_max + 1, dtype=np.int64)
    sigma[1] = 1
    
    lp = np.zeros(n_max + 1, dtype=np.int64)
    sp = np.zeros(n_max + 1, dtype=np.int64)
    sp[1] = 1
    lp[1] = 1
    
    is_composite = np.zeros(n_max + 1, dtype=bool)
    primes = []
    
    # Linear sieve
    for i in range(2, n_max + 1):
        if not is_composite[i]:
            primes.append(i)
            lp[i] = i
            sp[i] = 1 + i
            sigma[i] = 1 + i
            
        for p in primes:
            v = i * p
            if v > n_max:
                break
            is_composite[v] = True
            if i % p == 0:
                # p divides i
                lp[v] = lp[i] * p
                sp[v] = sp[i] + lp[v]
                sigma[v] = (sigma[i] // sp[i]) * sp[v]
                break
            else:
                # p does not divide i
                lp[v] = p
                sp[v] = 1 + p
                sigma[v] = sigma[i] * (1 + p)
                
    # 2. Group indices by sigma value
    # We use a dictionary mapping sigma value to a list of indices
    # To save memory and time, we can filter out sigma values that are too large or small?
    # No, we need all.
    
    # Using defaultdict(list) is standard, but converting to numpy arrays later helps.
    groups = defaultdict(list)
    
    # Iterating 10M times in Python is slow (~1-2s).
    # We can use numpy to group indices by value more efficiently if we sort.
    # However, sigma values can be large.
    # Let's stick to Python loop for grouping as it's simple and likely fast enough (< 3s).
    # sigma is a numpy array, accessing sigma[n] is fast.
    
    for n in range(1, n_max + 1):
        groups[sigma[n]].append(n)
        
    # 3. Search for best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Equivalent to minimizing |a - 1.5b| / b
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We will process each group.
    # For each group, we have a sorted list of numbers L.
    # For each b in L, we want a in L such that a is close to 1.5*b.
    # Since L is sorted, we can use bisect to find the position of 1.5*b.
    # The candidates are the element at that position and the one before it.
    
    # Optimization:
    # Many groups are small.
    # For large groups, we can vectorize.
    
    # Let's iterate through groups.
    # We can skip groups where the ratio of max/min is not close to 1.5?
    # Actually, we just need to check if there's any pair.
    
    # To speed up, we can convert small lists to numpy arrays?
    # Bisect works on lists. Numpy arrays support bisect too (via searchsorted).
    # Let's use bisect on lists for simplicity and correctness first, then optimize if needed.
    
    # Time check
    if time.time() - start_time > DEADLINE:
        return [int(best_a), int(best_b)]
        
    for sig_val, lst in groups.items():
        if time.time() - start_time > DEADLINE:
            break
            
        if len(lst) < 1:
            continue
            
        # If the group has only 1 element, a=b is the only option.
        # Error = |1 - 1.5| = 0.5.
        # We already have a better error (from 11,6 which is 1/3 ~ 0.33).
        # So we can skip groups of size 1 unless we haven't found anything better yet.
        # But 11,6 is a valid pair with sigma(11)=12, sigma(6)=12.
        # So best_error <= 1/3.
        # Groups of size 1 yield error 0.5, which is worse.
        # So we can skip groups of size 1.
        
        if len(lst) == 1:
            continue
            
        # For each b in lst, find closest a to 1.5*b
        # We can vectorize this for the whole list?
        # Let's try a loop for now, but with early termination if possible?
        # No, we need to check all b to find the global min.
        
        # Optimization: 
        # If the group is very large, use numpy.
        # If small, use python loop with bisect.
        
        if len(lst) < 100:
            # Python loop
            for b in lst:
                target = 1.5 * b
                pos = bisect.bisect_left(lst, target)
                
                candidates = []
                if pos < len(lst):
                    candidates.append(lst[pos])
                if pos > 0:
                    candidates.append(lst[pos-1])
                
                for a in candidates:
                    # Calculate error: |2a - 3b| / (2b)
                    num = abs(2*a - 3*b)
                    den = 2*b
                    err = num / den
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
        else:
            # Use numpy for larger groups
            # Convert to numpy array for vectorized operations
            arr = np.array(lst, dtype=np.int64)
            
            # For each b in arr, target is 1.5*b
            targets = arr * 1.5
            
            # Find indices where targets would be inserted
            # searchsorted returns the index of the first element >= target
            idxs = np.searchsorted(arr, targets)
            
            # We need to check idxs and idxs-1
            # Clamp indices to valid range [0, len(arr)-1]
            
            # Candidate 1: idxs
            idx1 = np.clip(idxs, 0, len(arr) - 1)
            a1 = arr[idx1]
            
            # Candidate 2: idxs - 1
            idx2 = np.clip(idxs - 1, 0, len(arr) - 1)
            a2 = arr[idx2]
            
            # Calculate errors for both candidates
            # err1 = |2*a1 - 3*b| / (2*b)
            # err2 = |2*a2 - 3*b| / (2*b)
            
            num1 = np.abs(2*a1 - 3*arr)
            den = 2*arr
            err1 = num1 / den
            
            num2 = np.abs(2*a2 - 3*arr)
            err2 = num2 / den
            
            # Combine and find min
            # We need to handle the case where a1 == a2 (if target is exactly in array, bisect_left gives index, idx-1 is previous)
            # If a1 == a2, err1 == err2.
            
            # Find min error
            min_err = np.minimum(err1, err2)
            global_min_idx = np.argmin(min_err)
            
            current_min_err = min_err[global_min_idx]
            
            if current_min_err < best_error:
                b_val = arr[global_min_idx]
                # Determine which a was used
                if err1[global_min_idx] <= err2[global_min_idx]:
                    a_val = a1[global_min_idx]
                else:
                    a_val = a2[global_min_idx]
                
                best_error = current_min_err
                best_a = int(a_val)
                best_b = int(b_val)
                
    # Fallback if no better found (should not happen as 11,6 exists)
    if best_a is None:
        return [11, 6]
        
    # Ensure types are Python int
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
