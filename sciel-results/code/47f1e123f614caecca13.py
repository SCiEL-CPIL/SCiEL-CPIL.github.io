"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s kill
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # We'll use a greedy approach with coverage tracking
    # State: set of covered integers, current budget used (as Fraction)
    
    # Target: cover as much of [1, L] as possible
    # Strategy: greedy by coverage/budget ratio, with gap-filling
    
    # Use a large enough range to track coverage
    MAX_RANGE = 10000
    
    covered = [False] * (MAX_RANGE + 1)
    classes = []
    budget_used = Fraction(0)
    budget_limit = Fraction(1)
    
    # Precompute for each prime p, the coverage pattern for each residue r
    # coverage[p][r] = set of positions in [1, MAX_RANGE] that are ≡ r (mod p)
    # But this is too memory intensive. Instead, compute on the fly.
    
    def get_new_coverage(p, r, covered_arr):
        """Count how many new integers in [1, MAX_RANGE] are covered by (p, r)"""
        count = 0
        # Integers ≡ r (mod p) in [1, MAX_RANGE]
        # First such integer: r if r >= 1, else r + p, etc.
        # Actually, we want n ≡ r (mod p), 1 <= n <= MAX_RANGE
        # n = r + k*p for k >= 0, and n >= 1
        if r == 0:
            start = p  # smallest positive n ≡ 0 (mod p)
        else:
            start = r
        if start > MAX_RANGE:
            return 0
        for n in range(start, MAX_RANGE + 1, p):
            if not covered_arr[n]:
                count += 1
        return count
    
    def apply_class(p, r, covered_arr):
        """Mark all integers ≡ r (mod p) in [1, MAX_RANGE] as covered"""
        if r == 0:
            start = p
        else:
            start = r
        for n in range(start, MAX_RANGE + 1, p):
            covered_arr[n] = True
    
    # Greedy: repeatedly pick (p, r) that maximizes new_coverage / (1/p)
    # = new_coverage * p, subject to budget constraint
    
    used_primes = set()
    
    # Sort primes by value for deterministic tie-breaking
    prime_list = sorted(primes)
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        best_score = -1
        best_p = None
        best_r = None
        best_new_cov = 0
        
        for p in prime_list:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost > budget_limit:
                continue
            
            # Try all residues for this prime
            # For efficiency, only try residues that might help
            # But we need to check all r in [0, p-1]
            # Optimization: for small p, check all; for large p, sample
            if p <= 50:
                residues_to_try = range(p)
            else:
                # For large p, we can't check all residues efficiently
                # Use a heuristic: try residues that align with uncovered gaps
                # Sample some residues
                residues_to_try = list(range(p))
            
            for r in residues_to_try:
                new_cov = get_new_coverage(p, r, covered)
                if new_cov == 0:
                    continue
                # Score: new coverage per unit budget = new_cov * p
                score = new_cov * p
                if score > best_score or (score == best_score and p < best_p):
                    best_score = score
                    best_p = p
                    best_r = r
                    best_new_cov = new_cov
        
        if best_p is None:
            break
        
        # Apply the best class
        apply_class(best_p, best_r, covered)
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        budget_used += Fraction(1, best_p)
        
        # Check if we've run out of budget or time
        if time.time() - start_time > DEADLINE:
            break
    
    # Now compute the actual coverage length L
    # Find the largest L such that all of 1..L are covered
    L = 0
    for i in range(1, MAX_RANGE + 1):
        if covered[i]:
            L = i
        else:
            break
    
    # If we didn't reach MAX_RANGE, L is correct
    # But we should verify: L is the first gap
    
    return classes

# EVOLVE-BLOCK-END
