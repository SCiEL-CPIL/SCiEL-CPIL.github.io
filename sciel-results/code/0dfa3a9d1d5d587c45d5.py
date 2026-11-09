import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # We search for A and B such that all 9 sums are distinct primes.
    # This guarantees pairwise coprime and max cardinality 9.
    
    # Parameters: a1, b1, da2, da3, db2, db3
    # a = [a1, a1+da2, a1+da3]
    # b = [b1, b1+db2, b1+db3]
    
    # Constraints:
    # 1 <= a_i <= 128
    # 1 <= b_j <= 128
    # a_i distinct
    # b_j distinct
    # Sums s_ij = a_i + b_j are distinct primes
    
    # To keep values small, we assume a1, b1 are small.
    # da2, da3, db2, db3 are positive integers (to ensure distinctness easily).
    
    found = False
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Search space
    # a1, b1 in [1, 20]
    # diffs in [1, 30]
    
    for a1 in range(1, 20):
        if time.time() > deadline:
            break
        for b1 in range(1, 20):
            if time.time() > deadline:
                break
            s11 = a1 + b1
            if not is_prime(s11):
                continue
                
            # We need 8 more distinct primes.
            # Let's just iterate diffs and check.
            
            for da2 in range(1, 30):
                if time.time() > deadline:
                    break
                a2 = a1 + da2
                if a2 > 128:
                    break
                for da3 in range(da2 + 1, 30):
                    if time.time() > deadline:
                        break
                    a3 = a1 + da3
                    if a3 > 128:
                        break
                    
                    # Check A distinctness (guaranteed by da2 < da3)
                    
                    for db2 in range(1, 30):
                        if time.time() > deadline:
                            break
                        b2 = b1 + db2
                        if b2 > 128:
                            break
                        for db3 in range(db2 + 1, 30):
                            if time.time() > deadline:
                                break
                            b3 = b1 + db3
                            if b3 > 128:
                                break
                            
                            # Check B distinctness (guaranteed by db2 < db3)
                            
                            # Calculate all 9 sums
                            sums = []
                            valid = True
                            for a in [a1, a2, a3]:
                                for b in [b1, b2, b3]:
                                    s = a + b
                                    if s in sums:
                                        valid = False
                                        break
                                    if not is_prime(s):
                                        valid = False
                                        break
                                    sums.append(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                # Found optimal solution
                                return [[a1, a2, a3], [b1, b2, b3]]
            
    # If no prime solution found quickly, fall back to broader search for coprime sums
    # But prime solution is very likely to exist.
    
    # Fallback: Use the known good solution from the prompt's current program if it was valid
    # The current program found a solution with score 1.8, which implies 9 sums.
    # Let's just return the hardcoded solution if we didn't find one?
    # No, we should try to find one.
    
    # Let's try a slightly wider search for non-prime but coprime sums if prime search fails
    # But 9 distinct primes is the target.
    
    # If we reach here, we failed to find a prime solution in the small box.
    # Let's return a known valid solution with 9 sums if we can construct one,
    # or fall back to the seed solution.
    
    # Actually, let's just return the hardcoded solution that is known to work?
    # The prompt says "Rewrite the program".
    # If the search is fast, it will find it.
    
    # Let's add a second phase with larger diffs if needed.
    
    for a1 in range(1, 50):
        if time.time() > deadline:
            break
        for b1 in range(1, 50):
            if time.time() > deadline:
                break
            s11 = a1 + b1
            if not is_prime(s11):
                continue
            for da2 in range(1, 50):
                if time.time() > deadline:
                    break
                a2 = a1 + da2
                if a2 > 128: break
                for da3 in range(da2 + 1, 50):
                    if time.time() > deadline:
                        break
                    a3 = a1 + da3
                    if a3 > 128: break
                    for db2 in range(1, 50):
                        if time.time() > deadline:
                            break
                        b2 = b1 + db2
                        if b2 > 128: break
                        for db3 in range(db2 + 1, 50):
                            if time.time() > deadline:
                                break
                            b3 = b1 + db3
                            if b3 > 128: break
                            
                            sums = []
                            valid = True
                            for a in [a1, a2, a3]:
                                for b in [b1, b2, b3]:
                                    s = a + b
                                    if s in sums or not is_prime(s):
                                        valid = False
                                        break
                                    sums.append(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                return [[a1, a2, a3], [b1, b2, b3]]
                                
    # If all else fails, return the seed solution
    return [best_A, best_B]
