import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Precompute primes up to 256 (max sum = 128+128)
    # Using a simple sieve
    sieve = [True] * 257
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if sieve[i]:
            for j in range(i * i, 257, i):
                sieve[j] = False
    
    # Set of primes for O(1) lookup
    primes = {i for i in range(2, 257) if sieve[i]}

    def is_prime(n):
        return n in primes

    def is_pairwise_coprime(values):
        vals = sorted(set(values))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) > 1:
                    return False
        return True

    # Strategy: 
    # Iterate over candidate A sets.
    # For each A, use backtracking to find B.
    # Prune aggressively if sums are not prime or duplicate.
    
    # We want 9 distinct primes.
    # If we find 9 distinct primes, they are automatically pairwise coprime.
    
    best_result = None
    best_count = 0

    # Generate candidate A sets. 
    # Small numbers are more likely to produce small primes.
    # We'll iterate a1, a2, a3 in increasing order.
    
    # Optimization: 
    # If a1+b1, a2+b1, a3+b1 are primes, then b1 must be such that these are primes.
    # This is a strong constraint.
    
    # We'll try a limited set of A structures first, then broaden.
    
    # Phase 1: Focused search on small A and B
    # A = [a1, a2, a3] with a1 < a2 < a3
    # B = [b1, b2, b3] with b1 < b2 < b3
    
    # To speed up, we can pre-compute valid b's for a given a?
    # No, b depends on all a's.
    
    # Let's just do the backtracking efficiently.
    
    for a1 in range(1, 20):
        if time.time() > deadline: break
        for a2 in range(a1 + 1, 30):
            if time.time() > deadline: break
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline: break
                
                A = [a1, a2, a3]
                
                # Backtrack for B
                found = False
                for b1 in range(1, 50):
                    if time.time() > deadline: break
                    
                    s11 = a1 + b1
                    s21 = a2 + b1
                    s31 = a3 + b1
                    if not (is_prime(s11) and is_prime(s21) and is_prime(s31)):
                        continue
                    
                    seen = {s11, s21, s31}
                    
                    for b2 in range(b1 + 1, 50):
                        if time.time() > deadline: break
                        
                        s12 = a1 + b2
                        s22 = a2 + b2
                        s32 = a3 + b2
                        if not (is_prime(s12) and is_prime(s22) and is_prime(s32)):
                            continue
                        
                        # Check distinctness against seen
                        if len(seen | {s12, s22, s32}) < 6:
                            continue
                        
                        seen2 = seen | {s12, s22, s32}
                        
                        for b3 in range(b2 + 1, 50):
                            if time.time() > deadline: break
                            
                            s13 = a1 + b3
                            s23 = a2 + b3
                            s33 = a3 + b3
                            if not (is_prime(s13) and is_prime(s23) and is_prime(s33)):
                                continue
                            
                            # Check distinctness against seen2
                            if len(seen2 | {s13, s23, s33}) < 9:
                                continue
                            
                            # If we reach here, we have 9 distinct primes.
                            # They are automatically pairwise coprime.
                            return [A, [b1, b2, b3]]

    # Phase 2: Broader search if Phase 1 fails
    # Expand ranges
    
    for a1 in range(1, 40):
        if time.time() > deadline: break
        for a2 in range(a1 + 1, 60):
            if time.time() > deadline: break
            for a3 in range(a2 + 1, 80):
                if time.time() > deadline: break
                
                A = [a1, a2, a3]
                
                for b1 in range(1, 60):
                    if time.time() > deadline: break
                    
                    s11 = a1 + b1
                    s21 = a2 + b1
                    s31 = a3 + b1
                    if not (is_prime(s11) and is_prime(s21) and is_prime(s31)):
                        continue
                    
                    seen = {s11, s21, s31}
                    
                    for b2 in range(b1 + 1, 60):
                        if time.time() > deadline: break
                        
                        s12 = a1 + b2
                        s22 = a2 + b2
                        s32 = a3 + b2
                        if not (is_prime(s12) and is_prime(s22) and is_prime(s32)):
                            continue
                        
                        if len(seen | {s12, s22, s32}) < 6:
                            continue
                        
                        seen2 = seen | {s12, s22, s32}
                        
                        for b3 in range(b2 + 1, 60):
                            if time.time() > deadline: break
                            
                            s13 = a1 + b3
                            s23 = a2 + b3
                            s33 = a3 + b3
                            if not (is_prime(s13) and is_prime(s23) and is_prime(s33)):
                                continue
                            
                            if len(seen2 | {s13, s23, s33}) < 9:
                                continue
                            
                            return [A, [b1, b2, b3]]

    # Fallback: Return a known valid solution with 5 sums if no 9 found
    # A=[1,3,5], B=[4,6,8] -> sums {5,7,9,11,13}
    # Check if this is valid:
    # 5,7,9,11,13. 
    # gcd(5,7)=1, gcd(5,9)=1, gcd(5,11)=1, gcd(5,13)=1
    # gcd(7,9)=1, gcd(7,11)=1, gcd(7,13)=1
    # gcd(9,11)=1, gcd(9,13)=1
    # gcd(11,13)=1
    # All pairwise coprime. Distinct count: 5.
    return [[1, 3, 5], [4, 6, 8]]
