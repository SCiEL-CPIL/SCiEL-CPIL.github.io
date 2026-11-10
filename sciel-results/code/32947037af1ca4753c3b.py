import math
import time
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 10000000
    
    # SPF sieve
    spf = array('I', range(N + 1))
    # 0 and 1 are not prime
    spf[0] = 0
    spf[1] = 1
    
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i: # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def sigma(n):
        if n <= 1:
            return n
        res = 1
        while n > 1:
            p = spf[n]
            count = 0
            term_sum = 1
            term_pow = 1
            while n % p == 0:
                n //= p
                count += 1
                term_pow *= p
                term_sum += term_pow
            res *= term_sum
        return res

    best_a, best_b = 1, 1
    best_err = float('inf')
    
    # Iterate k
    # 2a - 3b = k  => a = (3b+k)/2
    # 2a - 3b = -k => a = (3b-k)/2
    
    max_k = 10000 # Check up to k=10000
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
            
        # Case 1: 2a - 3b = k
        # a = (3b + k) / 2
        # Constraints: a <= N, b <= N, a >= 1, b >= 1
        # 3b + k <= 2N => b <= (2N - k) / 3
        # 3b + k >= 2 => b >= (2 - k) / 3 (always true for k small, b>=1)
        # Parity: 3b + k even => b % 2 == k % 2
        
        max_b1 = (2 * N - k) // 3
        if max_b1 < 1:
            continue
        # Adjust for parity
        if max_b1 % 2 != k % 2:
            max_b1 -= 1
            
        if max_b1 >= 1:
            b = max_b1
            a = (3 * b + k) // 2
            if a <= N and a >= 1:
                if sigma(a) == sigma(b):
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        # Since k is increasing, and we want to minimize error, 
                        # and for this k we found the max b (min err for this k),
                        # any subsequent k will have larger numerator.
                        # However, b might be smaller for larger k?
                        # b_max ~ 2N/3. It decreases slowly.
                        # err ~ k / (4N).
                        # So smaller k is always better, provided a solution exists.
                        # We can break if we find a solution for k=1?
                        # Not necessarily, because k=1 might not have a solution.
                        # But if k=1 has a solution, it is the global minimum possible error?
                        # Yes, because k=1 is the smallest positive integer.
                        # And b is maximized.
                        # So if we find a solution for k=1, it is the best possible.
                        # We can break immediately.
                        # Wait, is it possible that a larger k allows a MUCH larger b?
                        # No, b is bounded by 2N/3.
                        # So k=1 with max b is the best possible error.
                        # So we can break on first found solution?
                        # Only if we are sure no other k=1 solution with larger b exists?
                        # We picked max_b1, so it is the largest b for k=1.
                        # So yes, if we find a match for k=1, it is optimal.
                        break
                        
        # Case 2: 2a - 3b = -k
        # a = (3b - k) / 2
        # Constraints: a <= N, b <= N
        # 3b - k <= 2N => b <= (2N + k) / 3
        # 3b - k >= 2 => b >= (k + 2) / 3
        # Parity: 3b - k even => b % 2 == k % 2
        
        max_b2 = (2 * N + k) // 3
        if max_b2 > N:
            max_b2 = N
        if max_b2 < 1:
            continue
            
        # Adjust for parity
        if max_b2 % 2 != k % 2:
            max_b2 -= 1
            
        # Also need b >= (k+2)//3
        min_b2 = (k + 2) // 3
        if max_b2 >= min_b2:
            b = max_b2
            a = (3 * b - k) // 2
            if a <= N and a >= 1:
                if sigma(a) == sigma(b):
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
                        break

    return [int(best_a), int(best_b)]
