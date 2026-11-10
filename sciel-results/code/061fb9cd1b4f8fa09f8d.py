"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations, permutations
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s to leave safety margin
    
    # Precompute primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(256)
    
    def is_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        if len(distinct) < 2:
            return True
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Return (valid, cardinality) for a pair."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if len(distinct) < 2:
            return (False, 0)
        if not is_pairwise_coprime(sums):
            return (False, 0)
        return (True, len(distinct))
    
    # Start with known good solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # |{5,11,17,23,29}| = 5
    
    # Try to find better solutions by search
    # Strategy: enumerate all possible A, B from [1..128]
    # 128 choose 3 = 349,280 possibilities for each, too many to brute force both
    # Instead, use structured search
    
    # Key insight: for 9 distinct pairwise coprime sums, each sum must be 
    # "nearly prime" in the sense that no prime divides two different sums.
    
    # Let's try: make A and B such that sums are all prime or products of unique primes
    # Best approach: try A = {1, p, q} where p,q are primes, B = {r, s, t}
    
    # First, let's do a smarter search:
    # For each A (sorted triple from 1..128), find best B
    # But that's still 349k * 349k which is too much.
    
    # Alternative: random search + local improvement
    
    import random
    random.seed(42)
    
    # Candidate pools: use numbers that tend to produce coprime sums
    # If we use odd numbers for A and even for B (or vice versa), sums have mixed parity
    # Actually, let's just do structured random search
    
    # Precompute: for a given A, what B values produce good sums?
    # Let's try a more targeted approach
    
    # Observation: if A = {1, a, b} and B = {c, d, e}, sums are:
    # 1+c, 1+d, 1+e, a+c, a+d, a+e, b+c, b+d, b+e
    # For these to be pairwise coprime, we need careful selection.
    
    # Let's try: A contains 1, and B contains values such that 1+b is prime
    # Then a+1 and a+b should also be coprime to all others
    
    # Try systematic search with A = {1, x, y}
    # and B = {u, v, w} where 1+u, 1+v, 1+w are primes
    
    prime_set = set(primes)
    
    # B candidates: values b in [1,128] such that 1+b is prime
    # i.e., b in [1,127] such that b+1 is prime
    # b+1 prime means b is in {p-1 : p prime, 2<=p<=128}
    B_candidates = [p - 1 for p in primes if p <= 128]  # b such that b+1 is prime
    B_candidates = [b for b in B_candidates if 1 <= b <= 128]
    
    # Now search over A = {1, x, y} and B = {u, v, w} from B_candidates
    # This reduces the space significantly
    
    # Further: for the sums to be pairwise coprime, x+u, x+v, x+w, y+u, y+v, y+w
    # must all be coprime to 1+u, 1+v, 1+w and to each other.
    
    # Let's do a two-level search:
    # Level 1: pick B triple from B_candidates (or broader)
    # Level 2: pick A triple and check
    
    # Actually, let's just do random sampling with time budget
    
    found_better = False
    
    # Try structured: A = {1, a, b} where a, b are odd (so a+even, b+even have structure)
    # Or A = {1, a, b} where a, b are primes
    
    # Let's try: for many random (A, B) pairs, check
    # And also try local search from the seed
    
    # Local search: modify one element at a time
    def local_search(A, B, deadline):
        nonlocal best_A, best_B, best_score
        current_A = list(A)
        current_B = list(B)
        current_score = score_pair(current_A, current_B)[1]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in current_A and new_val != current_A[i]:
                        continue
                    trial_A = list(current_A)
                    trial_A[i] = new_val
                    if len(set(trial_A)) < 3:
                        continue
                    valid, sc = score_pair(trial_A, current_B)
                    if valid and sc > current_score:
                        current_A = trial_A
                        current_score = sc
                        improved = True
                        break
                    elif valid and sc == current_score:
                        # Keep for diversity, but don't accept as improvement
                        pass
                if improved:
                    break
            for i in range(3):
                if time.time() >= deadline:
                    break
                for new_val in range(1, 129):
                    if new_val in current_B and new_val != current_B[i]:
                        continue
                    trial_B = list(current_B)
                    trial_B[i] = new_val
                    if len(set(trial_B)) < 3:
                        continue
                    valid, sc = score_pair(current_A, trial_B)
                    if valid and sc > current_score:
                        current_B = trial_B
                        current_score = sc
                        improved = True
                        break
                    elif valid and sc == current_score:
                        pass
                if improved:
                    break
        
        if current_score > best_score:
            best_A = list(current_A)
            best_B = list(current_B)
            best_score = current_score
            found_better = True
        return current_score
    
    # Start local search from seed
    local_search([1, 7, 13], [4, 10, 16], deadline)
    
    # Try more random starts
    if time.time() < deadline:
        for attempt in range(200000):
            if time.time() >= deadline:
                break
            # Random A, B
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
            valid, sc = score_pair(A, B)
            if valid and sc > best_score:
                best_A = list(A)
                best_B = list(B)
                best_score = sc
                found_better = True
                # Do local search from here
                local_search(best_A, best_B, deadline)
    
    # Also try targeted: A = {1, p, q} for primes p, q
    if time.time() < deadline:
        small_primes = [p for p in primes if p <= 128]
        for a in small_primes:
            if time.time() >= deadline:
                break
            for b in small_primes:
                if b <= a:
                    continue
                if time.time() >= deadline:
                    break
                A = [1, a, b]
                # Try a few B values
                for _ in range(50):
                    B = sorted(random.sample(range(1, 129), 3))
                    valid, sc = score_pair(A, B)
                    if valid and sc > best_score:
                        best_A = list(A)
                        best_B = list(B)
                        best_score = sc
                        found_better = True
                        local_search(best_A, best_B, deadline)
    
    # Final: try to get 9 distinct sums
    # Theoretical max is 9. Let's see if we can find it.
    # For 9 pairwise coprime sums, each must be a power of a distinct prime (or 1, but sums >= 2)
    # Actually they just need to be pairwise coprime, not necessarily prime powers.
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
