import random
import math
import time

def construct():
    # Deadline guard: 60s limit, use 50s for safety
    deadline = time.time() + 50.0
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # Fallback from seed

    def check_pairwise_coprime(nums):
        """Check if all distinct numbers in the list are pairwise coprime."""
        distinct_nums = list(set(nums))
        n = len(distinct_nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct_nums[i], distinct_nums[j]) != 1:
                    return False
        return True

    def get_distinct_sum_count(A, B):
        """Calculate the number of distinct sums in A+B."""
        return len({a + b for a in A for b in B})

    # Precompute primes up to 256 (max sum 128+128)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, val in enumerate(is_prime) if val]

    primes = sieve(256)
    prime_set = set(primes)

    # Strategy: Randomized search
    # We want A and B to be such that a_i + b_j are likely primes.
    # Primes > 2 are odd. So if we pick A and B with mixed parity, sums will be mixed.
    # However, if we pick A all odd and B all even, sums are odd.
    # If we pick A all even and B all odd, sums are odd.
    # If we pick A mixed and B mixed, sums can be even (2 is the only even prime).
    # To maximize distinct prime sums, it's often easier to target odd primes.
    # Let's try generating A and B from numbers that are not too large to keep sums small and prime-rich.
    
    # Candidate pool for A and B: numbers in [1, 128]
    # To make sums prime, we can bias towards numbers that, when added, yield primes.
    # A simple heuristic: pick random numbers, check if all 9 sums are primes.
    
    # To speed up, we can restrict the search space or use a smarter generation.
    # Let's try generating A and B such that their elements are drawn from a range that yields many primes.
    # Range [1, 128] is fixed.
    
    # Randomization with restarts
    for _ in range(1000000):  # Upper bound on iterations, will break on time
        if time.time() > deadline:
            break
            
        # Generate random sets A and B
        # To increase chance of prime sums, we can sample from numbers that are "prime-friendly".
        # But random sampling in [1, 128] is fast enough for a few thousand tries.
        
        # Ensure distinctness
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        # Calculate sums
        sums = [a + b for a in A for b in B]
        
        # Check if all 9 sums are distinct
        if len(set(sums)) != 9:
            continue
            
        # Check if all distinct sums are pairwise coprime
        # Faster check: if all are primes, they are pairwise coprime.
        # Let's first check if they are all primes.
        if all(s in prime_set for s in sums):
            # If all are primes, they are distinct (checked above) and pairwise coprime.
            # This achieves the maximum cardinality of 9.
            return A, B
            
        # If not all primes, check pairwise coprimality for the distinct sums
        # This is more expensive, so only do it if we don't find a prime solution quickly
        # But since we want max cardinality 9, and prime solution is sufficient, 
        # we can skip non-prime 9-sum sets unless we are desperate.
        # However, a 9-sum set with composite numbers could also be valid if pairwise coprime.
        # But finding a prime solution is a strong target.
        
        # Let's also check for valid 9-sum sets that are not all primes, just in case.
        # But prime solution is easier to verify.
        
        # If we don't find a prime solution, we might find a composite one.
        # Let's check coprimality for distinct sums.
        distinct_sums = list(set(sums))
        if check_pairwise_coprime(distinct_sums):
            # We found a valid set with 9 distinct pairwise coprime sums.
            return A, B

    # If time runs out or no perfect solution found, return the best found or fallback
    # The fallback [[1, 7, 13], [4, 10, 16]] has sums:
    # 1+4=5, 1+10=11, 1+16=17
    # 7+4=11, 7+10=17, 7+16=23
    # 13+4=17, 13+10=23, 13+16=29
    # Distinct sums: {5, 11, 17, 23, 29}. Count = 5.
    # Are they pairwise coprime?
    # 5, 11, 17, 23, 29 are all primes. Yes.
    # So fallback is valid with score corresponding to 5.
    
    return best_result
