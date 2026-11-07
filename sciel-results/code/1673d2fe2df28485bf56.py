import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = 85.0 # 5s margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for sigma
    # sigma[n] = sum of divisors
    # We can do this by iterating i from 1 to N and adding i to sigma[j] for all multiples j of i.
    # This is O(N log N). For N=10^7, this is ~ 10^7 * 16 = 1.6 * 10^8 operations.
    # In NumPy, sigma[i::i] += i is fast.
    
    # Optimization: Only iterate i up to N? Yes.
    # But we can be smarter.
    # Standard sieve:
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 10:
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline - 5:
        return [11, 6]

    # 2. Group indices by sigma value
    # We only care about sigma values that appear more than once.
    # Using a dictionary of lists might be memory heavy if we store all indices.
    # But total number of indices is N.
    # Let's use a dictionary mapping sigma -> list of indices.
    
    # To avoid Python overhead of appending to lists in a loop of 10^7, 
    # we can use np.unique or similar? 
    # Or just loop in Python. 10^7 iterations in Python takes ~0.5-1s.
    
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    if time.time() - start_time > deadline - 2:
        return [11, 6]

    # 3. Search for best pair
    best_err = float('inf')
    best_a = 11
    best_b = 6
    best_num = 4 # |2a - 3b|
    best_den = 12 # 2b
    
    # Iterate through groups
    for s, indices in groups.items():
        if len(indices) < 2:
            continue
            
        # Sort indices
        indices.sort()
        
        # For each b in indices, find a in indices closest to 1.5 * b
        # We can iterate b and use bisect.
        # Since indices is sorted, we can do this efficiently.
        
        # Optimization: 
        # If the range of indices in this group is small, we can do a double loop?
        # No, group sizes can be large (e.g., sigma=2 has only 1, but sigma of primes+1 can have many).
        # Actually, sigma(n) = n+1 for primes. So sigma(p) = p+1.
        # If sigma(p1) = sigma(p2), then p1 = p2. So primes are unique in their sigma value (unless same prime).
        # So groups of primes are size 1.
        # Groups with size > 1 come from composite numbers or non-prime numbers sharing the same sigma.
        # E.g., sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12.
        # sigma(11)=12, sigma(12)=28.
        # sigma(13)=14, sigma(14)=24.
        # sigma(15)=24. So 14 and 15 have sigma 24.
        # a=15, b=14 -> 15/14 = 1.07. 14/15 = 0.93.
        # Target 1.5.
        
        # Let's iterate b from indices.
        # To find a close to 1.5b, we use bisect_left on the sorted list `indices`.
        
        for b in indices:
            target = 1.5 * b
            # Find insertion point
            pos = bisect_left(indices, int(target))
            
            # Check neighbors around pos
            # pos can be 0 to len(indices)
            # Candidates: indices[pos-1], indices[pos]
            
            for idx in [pos - 1, pos]:
                if 0 <= idx < len(indices):
                    a = indices[idx]
                    # Check if a == b? 
                    # If a == b, ratio is 1, error 0.5. 
                    # But we want ratio 1.5.
                    # If a == b, |a/b - 1.5| = 0.5.
                    # We should avoid a == b if possible, but it's a valid pair if no better exists.
                    # However, the problem implies distinct a, b? "Approximate a ratio". Usually distinct.
                    # If a=b, error is 0.5.
                    # Let's just compute error.
                    
                    # Error = |a/b - 1.5| = |2a - 3b| / (2b)
                    # We want to minimize this.
                    # num = abs(2*a - 3*b)
                    # den = 2*b
                    
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    
                    # Compare num/den with best_num/best_den
                    # num * best_den < best_num * den
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a
                        best_b = b
                        # Update best_err for potential early break?
                        # If best_num == 0, error is 0. But a/b = 1.5 => 2a=3b.
                        # If 2a=3b, then a/b = 1.5 exactly.
                        # Then score is inf? "A raw value of exactly 0 cannot be scored and counts as invalid."
                        # So we must avoid a/b = 1.5 exactly?
                        # "Minimize |a/b - 3/2|".
                        # If |a/b - 3/2| = 0, it is invalid.
                        # So we must find a pair where a/b != 1.5.
                        # If we find num=0, we must skip it or find the next best?
                        # The task says "raw value of exactly 0 ... counts as invalid".
                        # So we should NOT return a pair with 2a = 3b.
                        # We need the smallest non-zero error.
                        
                        if num == 0:
                            # This pair is invalid. 
                            # We should not update best to this.
                            # But we might have a previous best.
                            # We should continue searching for a non-zero error.
                            # However, if this is the only pair, we are stuck?
                            # No, there are many pairs.
                            # So we just don't update best if num == 0.
                            pass
                        else:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b

    # Final check: if best_num is 0 (should not happen if we skipped), or if we didn't find any?
    # We initialized with 11, 6. num=4, den=12.
    
    return [int(best_a), int(best_b)]
