"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _sieve_primes(n):
    """Return all primes up to n using sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

def _compute_covered_length(classes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered.
    classes: list of [p, r] where r is the residue mod p.
    Returns L (capped at max_L).
    """
    if not classes:
        return 0
    # For each number n, check if any class covers it
    # Use a boolean array for efficiency
    covered = [False] * (max_L + 1)
    for p, r in classes:
        # n ≡ r (mod p) means n = r + k*p for k >= 0
        # First n >= 1: if r == 0, start at p; else start at r
        start = r if r >= 1 else p
        if start <= max_L:
            for n in range(start, max_L + 1, p):
                covered[n] = True
    # Find the largest L such that all 1..L are covered
    L = 0
    for i in range(1, max_L + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _sum_reciprals(classes):
    """Compute exact sum of 1/p for all classes."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 140s out of 150s limit
    
    primes = _sieve_primes(1000)
    # We need sum of 1/p <= 1. Let's figure out which primes we can use.
    # Greedy: start with small primes (largest reciprocal) and add while sum <= 1
    # But we want to maximize coverage, so we need to be strategic.
    
    # Strategy: Use backtracking/CSP to assign residues to a subset of primes
    # such that the covered length is maximized, subject to sum(1/p) <= 1.
    
    # First, determine candidate primes. We'll consider all primes up to 1000.
    # The constraint is sum(1/p) <= 1.
    
    # Key insight: To cover 1..L, we need for each n in 1..L, at least one prime p
    # in our set with residue r such that n ≡ r (mod p).
    
    # This is equivalent to: for each n, the set of primes NOT covering n must be
    # such that no single prime covers it.
    
    # Let's use a different approach: 
    # 1. Select a subset of primes with sum(1/p) <= 1
    # 2. For that subset, find residues that maximize coverage
    # 
    # For step 2, given a set of primes, we can use a greedy/local search approach:
    # Start with random residues, then iteratively improve by changing one residue
    # at a time to maximize coverage.
    
    # But first, which primes to use? 
    # The seed uses primes from 13 upward. Let's think about this differently.
    # 
    # Actually, the problem is: choose primes and residues to maximize L.
    # 
    # Let's use simulated annealing with incremental updates.
    
    # Precompute: for each prime p and residue r, which numbers in 1..MAX_L does it cover?
    # That's too much memory. Instead, we'll compute coverage on the fly.
    
    # Let's use a count array: count[n] = number of classes covering n.
    # When we change a class from (p_old, r_old) to (p_new, r_new), we:
    #   - Decrement count[n] for all n ≡ r_old (mod p_old)
    #   - Increment count[n] for all n ≡ r_new (mod p_new)
    #   - The covered length is the largest L such that count[n] >= 1 for all n in 1..L
    
    MAX_L = 100000
    
    # We'll work with a fixed set of primes. Which set?
    # Let's try: all primes up to some limit, but we need sum(1/p) <= 1.
    # 
    # The sum of reciprocals of primes diverges, so we can't use all primes.
    # Let's compute cumulative sum and find how many primes we can use.
    
    cum_sum = Fraction(0, 1)
    usable_primes = []
    for p in primes:
        new_sum = cum_sum + Fraction(1, p)
        if new_sum <= 1:
            cum_sum = new_sum
            usable_primes.append(p)
        else:
            break
    
    # usable_primes contains primes we can use (in order of increasing p)
    # But maybe we should skip some small primes and use larger ones?
    # Small primes cover more numbers per unit reciprocal cost? No, actually
    # a prime p covers 1/p fraction of numbers, so the "efficiency" is the same.
    # But small primes have fewer residue options... wait, no, p residues.
    # 
    # Actually, the key is: with prime p, you can cover any 1/p fraction of integers.
    # The question is how to coordinate them to cover a prefix 1..L.
    
    # Let's try a different approach: use all primes up to some bound, 
    # and use local search to optimize residues.
    
    # For the prime selection: let's try different subsets.
    # Start with the greedy set (smallest primes), then try swapping.
    
    # Actually, let's just use a fixed set and optimize residues.
    # The seed uses primes from 13 upward. Let's see how many we can use.
    
    # Let's compute: sum of 1/p for primes from 13 to 997
    s = Fraction(0, 1)
    count = 0
    for p in primes:
        if p >= 13:
            s += Fraction(1, p)
            count += 1
    # This sum is about 0.92 as noted in the seed.
    
    # So we can use all primes from 13 to 997. That's 168 primes.
    # With 168 primes, we have a lot of degrees of freedom.
    
    # Let's use simulated annealing to optimize the residues.
    
    # State: list of residues for each prime in usable_primes
    # Objective: maximize covered length
    
    # For efficiency, we'll use a count array and a prefix "uncovered" tracking.
    
    # Let's define the prime set: all primes from 13 to 997
    prime_set = [p for p in primes if 13 <= p <= 997]
    n_primes = len(prime_set)
    
    # Verify sum
    total_recip = sum(Fraction(1, p) for p in prime_set)
    assert total_recip <= 1, f"Sum of reciprocals {total_recip} > 1"
    
    # Initialize residues randomly
    residues = [random.randint(0, p-1) for p in prime_set]
    
    # Build count array
    count_arr = [0] * (MAX_L + 1)
    for i, p in enumerate(prime_set):
        r = residues[i]
        start = r if r >= 1 else p
        if start <= MAX_L:
            for n in range(start, MAX_L + 1, p):
                count_arr[n] += 1
    
    # Compute current covered length
    def compute_L():
        L = 0
        for i in range(1, MAX_L + 1):
            if count_arr[i] > 0:
                L = i
            else:
                break
        return L
    
    best_L = compute_L()
    best_residues = residues[:]
    
    # Simulated annealing
    # Temperature schedule: start high, decrease exponentially
    T_start = 100.0
    T_end = 0.01
    n_iterations = 50000
    
    # We'll do iterations, each changing one residue
    current_L = best_L
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Temperature
        frac = it / n_iterations
        T = T_start * (T_end / T_start) ** frac
        
        # Choose a random prime index
        idx = random.randint(0, n_primes - 1)
        p = prime_set[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Update count array: remove old residue, add new residue
        # Remove old
        start_old = old_r if old_r >= 1 else p
        if start_old <= MAX_L:
            for n in range(start_old, MAX_L + 1, p):
                count_arr[n] -= 1
        
        # Add new
        start_new = new_r if new_r >= 1 else p
        if start_new <= MAX_L:
            for n in range(start_new, MAX_L + 1, p):
                count_arr[n] += 1
        
        # Compute new L
        new_L = compute_L()
        
        # Acceptance
        delta = new_L - current_L
        if delta >= 0 or random.random() < __import__('math').exp(delta / T):
            # Accept
            residues[idx] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
        else:
            # Reject: revert
            # Remove new
            if start_new <= MAX_L:
                for n in range(start_new, MAX_L + 1, p):
                    count_arr[n] -= 1
            # Add old
            if start_old <= MAX_L:
                for n in range(start_old, MAX_L + 1, p):
                    count_arr[n] += 1
    
    # Return best solution
    return [[prime_set[i], best_residues[i]] for i in range(n_primes)]
# EVOLVE-BLOCK-END
