import time
import math
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 85  # Safety margin before 90s kill

    # Precompute primality for all possible sums (2..256)
    def sieve(max_n):
        is_p = [True] * (max_n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(max_n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, max_n+1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)

    def is_coprime_set(nums):
        """Check if all pairs in nums are coprime."""
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def score_pair(A, B):
        """Count distinct pairwise-coprime values in A+B."""
        if len(A) != 3 or len(B) != 3:
            return 0
        if len(set(A)) != 3 or len(set(B)) != 3:
            return 0
        for x in A:
            if x < 1 or x > 128:
                return 0
        for x in B:
            if x < 1 or x > 128:
                return 0
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        # Greedy: pick sums in sorted order, each must be coprime to all previously chosen
        count = 0
        chosen = []
        for s in sorted(sums):
            ok = True
            for t in chosen:
                if math.gcd(s, t) != 1:
                    ok = False
                    break
            if ok:
                chosen.append(s)
                count += 1
        return count

    def all_nine_distinct_coprime(A, B):
        """Check if all 9 sums are distinct and pairwise coprime."""
        sums = []
        for a in A:
            for b in B:
                s = a + b
                if s in sums:
                    return False
                sums.append(s)
        return is_coprime_set(sums)

    # Strategy: Find A, B such that all 9 sums are distinct primes.
    # This implies A and B must have opposite parity (one all-odd, one all-even)
    # to ensure sums are odd (and thus potentially prime, except 2).
    
    # Let A be odd numbers, B be even numbers.
    odd_candidates = [i for i in range(1, 129) if i % 2 != 0]
    even_candidates = [i for i in range(1, 129) if i % 2 == 0]
    
    # Precompute prime status for sums of odd + even
    # sum = a + b, where a is odd, b is even -> sum is odd
    # We need sum <= 128+128=256
    
    found_solution = None
    
    # Iterate over all combinations of 3 odd numbers for A
    # There are C(64, 3) = 41664 combinations. This is small enough to brute force.
    for A in combinations(odd_candidates, 3):
        if time.time() > DEADLINE - 5:
            break
            
        a1, a2, a3 = A
        
        # Find all even b in [1, 128] such that a1+b, a2+b, a3+b are all prime
        valid_bs = []
        for b in even_candidates:
            if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                valid_bs.append(b)
        
        if len(valid_bs) < 3:
            continue
        
        # We have at least 3 valid even numbers.
        # We need to pick 3 distinct b's such that all 9 sums are distinct.
        # Since a's are distinct and b's are distinct, and we are adding,
        # collisions can happen if a_i + b_j == a_k + b_l.
        # Let's just check combinations of 3 from valid_bs.
        
        # If valid_bs is large, we might have many combos.
        # But usually, the set of valid b's is small.
        # Let's try all combinations of 3 from valid_bs if len(valid_bs) is small,
        # otherwise sample.
        
        if len(valid_bs) <= 10:
            for B in combinations(valid_bs, 3):
                B = list(B)
                if all_nine_distinct_coprime(A, B):
                    found_solution = (list(A), B)
                    break
            if found_solution:
                break
        else:
            # Sample some
            import random
            rng = random.Random(42)
            for _ in range(100):
                B = list(rng.sample(valid_bs, 3))
                if all_nine_distinct_coprime(A, B):
                    found_solution = (list(A), B)
                    break
            if found_solution:
                break
    
    if found_solution:
        return found_solution
    
    # Fallback: If the strict prime search didn't find a solution (unlikely),
    # use the previous hill-climbing or random search approach.
    
    rng = random.Random(42)
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]
    
    # Phase 2: Random search with scoring
    # Score is the number of distinct pairwise coprime sums.
    
    def score_pair(A, B):
        """Count distinct pairwise-coprime values in A+B."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        # Greedy: pick sums in sorted order, each must be coprime to all previously chosen
        count = 0
        chosen = []
        for s in sorted(sums):
            ok = True
            for t in chosen:
                if math.gcd(s, t) != 1:
                    ok = False
                    break
            if ok:
                chosen.append(s)
                count += 1
        return count

    attempts = 0
    while time.time() < DEADLINE - 2 and best_score < 9:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = score_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        
        # Simple hill climbing step
        if s < 9:
            improved = True
            while improved and time.time() < DEADLINE - 2:
                improved = False
                # Try changing A
                for i in range(3):
                    old = A[i]
                    for v in range(1, 129):
                        if v == old or v in A:
                            continue
                        A2 = A[:]
                        A2[i] = v
                        s2 = score_pair(A2, B)
                        if s2 > s:
                            s = s2
                            A = A2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Try changing B
                for j in range(3):
                    old = B[j]
                    for v in range(1, 129):
                        if v == old or v in B:
                            continue
                        B2 = B[:]
                        B2[j] = v
                        s2 = score_pair(A, B2)
                        if s2 > s:
                            s = s2
                            B = B2
                            improved = True
                            break
                    if improved:
                        break
                
                if s > best_score:
                    best_score = s
                    best_A, best_B = A[:], B[:]
                if s >= 9:
                    break
        
        attempts += 1
        if attempts % 1000 == 0 and best_score < 9:
             # Random restart if stuck
             A = sorted(rng.sample(range(1, 129), 3))
             B = sorted(rng.sample(range(1, 129), 3))
             s = score_pair(A, B)
             if s > best_score:
                 best_score = s
                 best_A, best_B = A[:], B[:]

    return [best_A, best_B]
