"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _check_pairwise_coprime(values):
    """Check if all distinct values are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (cardinality, is_valid) for the sumset A+B."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if not _check_pairwise_coprime(distinct):
        return (0, False)
    return (len(distinct), True)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Initialize with a known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # known valid score
    
    # Verify the seed
    score, valid = _evaluate(best_A, best_B)
    if valid:
        best_score = score
    else:
        best_score = 0
        best_A = [1, 2, 3]
        best_B = [4, 5, 6]
    
    # If we already found the maximum possible score (9), we can return early
    if best_score >= 9:
        return [best_A, best_B]
    
    rng = random.Random(42)
    range_ = range(1, 129)
    
    # Precompute primes and prime powers for structured search
    # A sum is "good" if it's a prime or a power of a prime (to help pairwise coprime)
    # Actually, just primes are safer for pairwise coprime if we want many.
    # Let's generate primes up to 256 (max sum 128+128)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(256)
    primes = [i for i in range(2, 257) if is_prime[i]]
    
    # Strategy 1: Parity restriction
    # If A is all odd and B is all even (or vice versa), all sums are odd.
    # This avoids the factor 2 collision.
    odd_numbers = [i for i in range(1, 129) if i % 2 != 0]
    even_numbers = [i for i in range(1, 129) if i % 2 == 0]
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Mix of strategies:
        # 1. Random search with parity mismatch (higher density of valid candidates)
        # 2. Random search general
        # 3. Local search
        
        if iteration % 10 < 6:
            # Parity mismatched random search
            # A from odd, B from even OR A from even, B from odd
            if rng.random() < 0.5:
                A = list(rng.sample(odd_numbers, 3))
                B = list(rng.sample(even_numbers, 3))
            else:
                A = list(rng.sample(even_numbers, 3))
                B = list(rng.sample(odd_numbers, 3))
        else:
            # General random search
            A = list(rng.sample(range_, 3))
            B = list(rng.sample(range_, 3))
        
        score, valid = _evaluate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return [best_A, best_B]
        
        # Local search: perturb best solution
        if iteration % 5 == 0 and best_score < 9:
            # Try perturbing A
            A2 = best_A[:]
            idx = rng.randint(0, 2)
            # Try a few random new values
            for _ in range(5):
                A2[idx] = rng.randint(1, 128)
                if len(set(A2)) == 3:
                    score2, valid2 = _evaluate(A2, best_B)
                    if valid2 and score2 > best_score:
                        best_score = score2
                        best_A = A2[:]
                        if best_score >= 9:
                            return [best_A, best_B]
                        break # Found improvement, stop this idx
            
            # Try perturbing B
            B2 = best_B[:]
            idx = rng.randint(0, 2)
            for _ in range(5):
                B2[idx] = rng.randint(1, 128)
                if len(set(B2)) == 3:
                    score3, valid3 = _evaluate(best_A, B2)
                    if valid3 and score3 > best_score:
                        best_score = score3
                        best_B = B2[:]
                        if best_score >= 9:
                            return [best_A, best_B]
                        break
        
        # Occasional structured search: try to pick sums that are primes
        # This is harder to implement directly on A,B, but we can bias random choices
        # towards numbers that sum to primes.
        # For now, the parity bias is the main structural improvement.
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
