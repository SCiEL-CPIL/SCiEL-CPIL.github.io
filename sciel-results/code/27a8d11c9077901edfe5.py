import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin within 90s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    # Precompute primes up to 256
    primes_set = set(x for x in range(2, 257) if is_prime(x))

    def check_pairwise_coprime(values):
        distinct_vals = list(set(values))
        for i in range(len(distinct_vals)):
            for j in range(i + 1, len(distinct_vals)):
                if math.gcd(distinct_vals[i], distinct_vals[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        """
        Returns (cardinality, is_valid).
        If valid and cardinality is 9, we can stop.
        """
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        if len(distinct_sums) < 9:
            # We want max cardinality. If < 9, it's not optimal, but might be valid.
            # However, the task asks to maximize cardinality. 9 is the max possible.
            # If we find a valid 9, we return it.
            # If we find a valid <9, we keep it as a backup, but prefer 9.
            if check_pairwise_coprime(distinct_sums):
                return len(distinct_sums), True
            return len(distinct_sums), False
        
        # If 9 distinct sums
        if check_pairwise_coprime(distinct_sums):
            return 9, True
        return 9, False

    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Initial valid check for fallback
    s, v = evaluate(best_A, best_B)
    if v:
        best_score = s

    # Strategy 1: Systematic search for all-prime sumsets
    # If all 9 sums are distinct primes, they are pairwise coprime.
    # We iterate A with small values to maximize the range of valid B.
    
    # Limit A search to small values to allow B to be in [1, 128]
    # a1 < a2 < a3
    max_a_val = 60 # If A is too large, B must be small, reducing options
    
    found_perfect = False
    
    for a1 in range(1, max_a_val):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, max_a_val):
            if time.time() > deadline - 5:
                break
            for a3 in range(a2 + 1, max_a_val):
                if time.time() > deadline - 5:
                    break
                
                A = [a1, a2, a3]
                
                # Find all b in [1, 128] such that a1+b, a2+b, a3+b are all prime
                valid_bs = []
                for b in range(1, 129):
                    if (a1 + b) in primes_set and (a2 + b) in primes_set and (a3 + b) in primes_set:
                        valid_bs.append(b)
                
                if len(valid_bs) < 3:
                    continue
                
                # We need to pick 3 distinct b's from valid_bs
                # such that the 9 sums are distinct.
                # Sums are a_i + b_j.
                # If b_j are distinct, are sums always distinct?
                # a1+b1, a1+b2 ...
                # If b1 != b2, a1+b1 != a1+b2.
                # But a1+b2 could equal a2+b1?
                # a1+b2 = a2+b1 => b2-b1 = a2-a1.
                # So we must avoid b's that differ by differences of a's.
                
                # Try combinations of 3 from valid_bs
                # To save time, if we have many valid_bs, we can just pick the first 3 
                # and check. If they fail distinctness, try others.
                # However, distinctness failure is rare if gaps are large.
                
                # Optimization: Just pick the first 3 valid_bs and check.
                # If distinctness fails, we might miss a solution, but it's a good heuristic.
                # To be safer, try a few combinations if the first fails.
                
                b_combinations_to_try = []
                # Add the first 3
                if len(valid_bs) >= 3:
                    b_combinations_to_try.append((valid_bs[0], valid_bs[1], valid_bs[2]))
                # Add last 3
                if len(valid_bs) >= 3:
                    b_combinations_to_try.append((valid_bs[-3], valid_bs[-2], valid_bs[-1]))
                # Add middle 3
                if len(valid_bs) >= 5:
                    b_combinations_to_try.append((valid_bs[1], valid_bs[2], valid_bs[3]))
                
                # Also try random samples if list is long
                if len(valid_bs) > 5:
                    for _ in range(5):
                        if len(valid_bs) >= 3:
                            sample = random.sample(valid_bs, 3)
                            b_combinations_to_try.append(tuple(sample))

                for b_tuple in b_combinations_to_try:
                    B = list(b_tuple)
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    distinct_sums = set(sums)
                    
                    if len(distinct_sums) == 9:
                        # Since they are all prime (by construction of valid_bs), they are pairwise coprime.
                        return A, B
                    
                    # If not distinct, check if it's a valid pairwise coprime set (though unlikely if not distinct primes)
                    # Actually, if they are not distinct, cardinality < 9, so it's not the max.
                    # We only care about max cardinality 9.
                    
    # Strategy 2: If no all-prime solution found, do a broader search for pairwise coprime sums
    # This handles cases where composite sums are used but still coprime.
    
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 2 and attempts < 50000:
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        
        if len(distinct_sums) == 9:
            if check_pairwise_coprime(distinct_sums):
                return A, B
        elif len(distinct_sums) > best_score:
            if check_pairwise_coprime(distinct_sums):
                best_score = len(distinct_sums)
                best_A = A
                best_B = B
        
        attempts += 1
        
        # Periodic check for time
        if attempts % 1000 == 0 and time.time() > deadline - 5:
            break

    # Fallback to best found
    return best_A, best_B

if __name__ == "__main__":
    A, B = construct()
    print(f"A: {A}, B: {B}")
    sums = [a+b for a in A for b in B]
    print(f"Sums: {sorted(set(sums))}")
    print(f"Cardinality: {len(set(sums))}")
