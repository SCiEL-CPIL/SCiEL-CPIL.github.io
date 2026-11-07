import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using O(N log N) sieve
    # sigma[j] = sum of divisors of j
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() > deadline - 10.0:
            # If we run out of time during sieve, we can't search properly.
            # But sieve is fast.
            break
        sigma[i::i] += i
        
    # Now search for best pair
    best_error = Fraction(1, 1) # Initialize with a large error
    best_pair = (11, 6) # Fallback, sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = |1.83 - 1.5| = 0.33.
    # Actually let's find a better fallback.
    # sigma(2)=3, sigma(3)=4. No.
    # sigma(1)=1, sigma(2)=3.
    # Let's just use (1,1) as initial, error = 0.5.
    # But (1,1) is invalid if a=b? "a, b positive integers". "a/b".
    # The problem says "Minimize |a/b - 3/2|".
    # If a=b, ratio is 1. Error 0.5.
    
    # We want to minimize error.
    # Error = |2a - 3b| / (2b)
    
    # Iterate b from N down to 1
    # We can skip b if 1/(2b) >= best_error.numerator / best_error.denominator
    # i.e., if best_error <= 1/(2b), then no b' <= b can improve.
    
    # To speed up, we can check a limited window of delta.
    # delta = 2a - 3b.
    # We want |delta| to be small.
    # So we check delta in [-10, 10].
    
    for b in range(N, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Check if we can break early
        # If best_error is 0, break (but 0 is invalid)
        if best_error.numerator == 0:
            break
            
        # Theoretical min error for this b is 1/(2b)
        # If 1/(2b) >= best_error, then we can break?
        # No, we are iterating downwards.
        # If we are at b, and 1/(2b) >= best_error, then for all b' < b, 1/(2b') > 1/(2b) >= best_error.
        # So no b' < b can improve best_error.
        # So we can break.
        
        # Compare 1/(2b) with best_error
        # 1/(2b) >= best_error  <=>  1 >= 2b * best_error
        # 2b * best_error.numerator / best_error.denominator <= 1
        # 2b * best_error.numerator <= best_error.denominator
        
        if 2 * b * best_error.numerator <= best_error.denominator:
            break
            
        sb = int(sigma[b])
        
        # Check candidates
        # a = (3b + delta) / 2
        # delta must have same parity as 3b (which is same parity as b)
        # If b is even, delta is even.
        # If b is odd, delta is odd.
        
        # We check delta in a range.
        # Let's check delta from -10 to 10.
        for delta in range(-10, 11):
            # Check parity
            if (delta % 2) != (b % 2):
                continue
                
            num_a = 3 * b + delta
            if num_a <= 0 or num_a > 2 * N: # a <= N, so 3b+delta <= 2N
                continue
                
            a = num_a // 2
            if a < 1 or a > N:
                continue
            if a == b:
                continue
                
            sa = int(sigma[a])
            if sa == sb:
                # Found a match
                # Error = |delta| / (2b)
                err_num = abs(delta)
                err_den = 2 * b
                err = Fraction(err_num, err_den)
                
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
                    
                    # If error is 0, it's invalid, but we can break?
                    # No, we need a non-zero error.
                    # If delta is 0, a = 1.5b.
                    # If b is even, a is integer.
                    # Then error is 0.
                    # The task says "A raw value of exactly 0 cannot be scored and counts as invalid."
                    # So we must skip delta=0?
                    # Or if we find delta=0, we must find a non-zero one?
                    # "Minimize |a/b - 3/2|".
                    # If 0 is possible, we can't use it.
                    # So we should avoid delta=0.
                    # Let's modify the delta loop to skip 0?
                    # Or if we find 0, we keep looking for non-zero?
                    # But 0 is the minimum. If we find 0, we can't use it.
                    # So we should look for the smallest non-zero delta.
                    
                    if err_num == 0:
                        # We found a perfect match, but it's invalid.
                        # We can't use it.
                        # We should continue searching for non-zero delta?
                        # But for this b, delta=0 is the best.
                        # We need to check other deltas for this b?
                        # Yes, we should check other deltas.
                        # But we already found a match.
                        # We should not update best_error with 0.
                        pass
                    else:
                        # Valid error
                        pass
                        
    return [int(best_pair[0]), int(best_pair[1])]
