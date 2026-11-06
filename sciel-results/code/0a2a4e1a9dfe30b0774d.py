"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all pairwise coprime pairs from 1..128
    # A sum value s is "good" if it's 1 or a prime power p^k
    # Two sums s1, s2 are compatible if gcd(s1, s2) == 1
    
    def is_prime_power_or_one(n):
        if n == 1:
            return True
        # Check if n is a prime power
        # Try dividing by primes up to sqrt(n)
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
                return False
        # n itself is prime
        return True
    
    # Build set of "atomic" sum values: 1 and prime powers up to 256
    atomic_sums = set()
    for n in range(1, 257):
        if is_prime_power_or_one(n):
            atomic_sums.add(n)
    
    # For each pair of atomic sums, check if they're coprime
    # We want to find A (3 distinct ints in [1,128]) and B (3 distinct ints in [1,128])
    # such that all distinct values in {a+b for a in A, b in B} are pairwise coprime
    
    # Strategy: enumerate candidate A sets and B sets, but use smart pruning
    # Key insight: if we pick A and B carefully, we can control which sums appear
    
    # Better approach: directly search over (A, B) pairs with backtracking
    # Use the fact that we want exactly 9 distinct sums that are pairwise coprime
    
    # Precompute: for each a in [1,128], b in [1,128], s=a+b
    # Group by s
    
    # Let's try a different structural approach:
    # Pick 9 pairwise coprime target sums first, then find A, B that produce them
    
    # Generate all sets of 9 pairwise coprime numbers in [2, 256]
    # (sums of two numbers in [1,128] range from 2 to 256)
    
    # Actually, let's do direct search with good heuristics
    # Random restarts with local improvement
    
    best_score = 0
    best_A = None
    best_B = None
    
    def evaluate(A, B):
        """Return number of distinct pairwise-coprime sums, or -1 if not all pairwise coprime"""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        # Check pairwise coprime
        slist = list(sums)
        for i in range(len(slist)):
            for j in range(i+1, len(slist)):
                if math.gcd(slist[i], slist[j]) != 1:
                    return -1
        return len(slist)
    
    # Candidate values: focus on numbers that lead to prime-power sums
    # If a+b is a prime power, it's more likely to be coprime to other prime powers
    
    # Generate candidate A and B values
    # A value a is "good" if for many b, a+b is a prime power
    
    all_vals = list(range(1, 129))
    
    # Score each value by how many prime-power sums it produces
    val_score = {}
    for v in all_vals:
        count = 0
        for w in all_vals:
            if (v + w) in atomic_sums:
                count += 1
        val_score[v] = count
    
    # Sort values by score
    sorted_vals = sorted(all_vals, key=lambda x: val_score[x], reverse=True)
    
    # Use top values as primary candidates
    top_candidates = sorted_vals[:40]
    
    # Random search with restarts
    random.seed(42)
    
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        
        # Pick random A and B from candidates (with some randomness)
        if iterations % 10 == 0:
            # Occasionally include a random value from full range
            A = random.sample(top_candidates, 3)
            B = random.sample(top_candidates, 3)
        else:
            # Mix of top candidates and some randomness
            A = random.sample(sorted_vals[:60], 3)
            B = random.sample(sorted_vals[:60], 3)
        
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                break
        
        # Local search: try swapping one element
        if score > 0:  # valid configuration
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Try modifying A
                idx = random.randint(0, 2)
                old_val = A[idx]
                new_val = random.choice(sorted_vals[:80])
                if new_val in A and new_val != old_val:
                    continue
                A[idx] = new_val
                new_score = evaluate(A, B)
                if new_score > score:
                    score = new_score
                    if score > best_score:
                        best_score = score
                        best_A = A[:]
                        best_B = B[:]
                        if best_score >= 9:
                            break
                else:
                    A[idx] = old_val
            
            if best_score >= 9:
                break
            
            # Try modifying B
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, 2)
                old_val = B[idx]
                new_val = random.choice(sorted_vals[:80])
                if new_val in B and new_val != old_val:
                    continue
                B[idx] = new_val
                new_score = evaluate(A, B)
                if new_score > score:
                    score = new_score
                    if score > best_score:
                        best_score = score
                        best_A = A[:]
                        best_B = B[:]
                        if best_score >= 9:
                            break
                else:
                    B[idx] = old_val
            
            if best_score >= 9:
                break
    
    if best_A is not None and best_B is not None:
        return [best_A, best_B]
    else:
        # Fallback: return seed
        return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
