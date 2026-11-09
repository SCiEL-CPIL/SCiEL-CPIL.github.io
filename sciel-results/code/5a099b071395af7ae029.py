"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import math

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Generate primes up to 1000
    primes = _primes_upto(1000)
    
    # We want to cover 1..L. The seed covers up to ~999 using primes from 13 upward.
    # The sum of reciprocals of primes from 13 to 997 is about 0.92.
    # We need sum(1/p) <= 1.
    
    # Strategy: Use a greedy/CSP approach to maximize L.
    # The key insight: we can choose which primes to use and their residues.
    # To maximize L, we want to cover as many consecutive integers starting from 1.
    
    # Let's think about this differently.
    # For each integer n in 1..L, we need some prime p and residue r such that n ≡ r (mod p).
    # Equivalently, p divides (n - r).
    
    # A natural approach: for each prime p, assign it to cover a specific set of integers.
    # If p covers integer n, then r = n mod p.
    # One prime p can cover at most one residue class, so it covers {r, r+p, r+2p, ...}.
    
    # To cover 1..L, we need for each n in 1..L, at least one prime p in our set with r_p = n mod p.
    
    # Greedy approach: 
    # Sort primes by size. Small primes are "expensive" (1/p is large) but cover frequently.
    # Large primes are "cheap" but cover rarely.
    
    # Actually, the seed uses primes from 13 upward, assigning prime[i] to cover integer i+1.
    # This means prime[i] has residue (i+1) mod prime[i].
    # Since i+1 < prime[i] for small i (prime[0]=13, i+1=1, so residue=1), this works.
    # As i grows, eventually i+1 > prime[i], but that's fine since residue is just mod p.
    
    # The sum of 1/p for primes from 13 to 997 is about 0.92, which is < 1.
    # So we have some budget left. Can we add more primes or adjust residues to extend L?
    
    # Let's compute the current L of the seed:
    # primes from 13: 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, ...
    # Number of primes from 13 to 997: let's count.
    # Primes <= 1000: 168. Primes < 13: 2,3,5,7,11 -> 5 primes.
    # So 168 - 5 = 163 primes from 13 to 997.
    # The seed covers integers 1 through 163? No wait, it covers integer i+1 for i=0..162.
    # So it covers 1..163? But the score is 1.0 which corresponds to L=999?
    
    # Wait, let me re-read. "L >= 999 would realize the Erdős-Ruzsa covering statement at x = 1000"
    # The score is the largest L such that every integer 1..L lies in some class.
    # The seed returns score 1.0, which means L is some value. Let me check what L the seed achieves.
    
    # Actually, the seed uses primes from 13 upward. The i-th prime covers integer i+1.
    # But a prime p with residue r covers ALL integers ≡ r (mod p), not just i+1.
    # So prime 13 with residue 1 covers 1, 14, 27, 40, ...
    # Prime 17 with residue 2 covers 2, 19, 36, ...
    # Prime 19 with residue 3 covers 3, 22, 41, ...
    # etc.
    
    # So the coverage is much more than just 1..163. The classes overlap and cover many integers.
    # The question is: what's the largest L such that 1..L are ALL covered?
    
    # Let me think about what's NOT covered. An integer n is NOT covered if for every prime p in our set,
    # n mod p != r_p.
    
    # This is complex. Let me just implement a solver that tries to maximize L.
    
    # Approach: 
    # 1. Start with a good initial solution (the seed).
    # 2. Try to extend L by adding primes or adjusting residues.
    # 3. Use backtracking with the budget constraint.
    
    # First, let's evaluate the seed's L.
    seed_classes = _seed_classes(primes)
    seed_L = _compute_L(seed_classes)
    
    best_L = seed_L
    best_classes = seed_classes
    
    # Now try to improve.
    # The budget is sum(1/p) <= 1.
    # Seed budget: sum(1/p for p in primes[5:]) where primes[5:] are primes >= 13.
    
    # Let's try a different strategy: use ALL primes up to 1000, but choose residues optimally.
    # This is a hard problem. Let's try a greedy approach:
    
    # For L_target, try to find residues for a subset of primes such that 1..L_target are covered
    # and sum(1/p) <= 1.
    
    # Greedy: for each n from 1 to L_target, if n is not covered, add a prime that covers n.
    # Choose the prime that gives the best coverage per cost (1/p).
    
    # But this is expensive for large L. Let's try a different approach.
    
    # Key insight: if we use prime p with residue r, it covers one integer in every block of p consecutive integers.
    # So it covers about L/p integers in 1..L.
    # The "density" of coverage is 1/p.
    # To cover all of 1..L, we need the union of these arithmetic progressions to be dense enough.
    
    # By the Chinese Remainder Theorem and inclusion-exclusion, covering is non-trivial.
    
    # Let's try: use small primes with carefully chosen residues to cover 1..L.
    # Small primes have high 1/p cost but high coverage density.
    
    # Strategy: binary search on L, and for each L, try to find a valid assignment.
    # But finding a valid assignment is NP-hard in general.
    
    # Let's try a simpler improvement: start from the seed, and try to add primes 2, 3, 5, 7, 11
    # with appropriate residues to fill gaps, if budget allows.
    
    # Seed uses primes from 13. Budget used: sum(1/p for p in primes >= 13, p <= 997).
    # Let's compute this.
    
    seed_primes = [p for p in primes if p >= 13]
    seed_budget = sum(Fraction(1, p) for p in seed_primes)
    remaining_budget = Fraction(1) - seed_budget
    
    if remaining_budget <= 0:
        return best_classes
    
    # Try adding small primes 2, 3, 5, 7, 11 one by one.
    # For each, try all residues and see if it extends L.
    
    small_primes = [2, 3, 5, 7, 11]
    
    current_classes = list(best_classes)
    current_budget = seed_budget
    
    for p in small_primes:
        cost = Fraction(1, p)
        if current_budget + cost > Fraction(1):
            continue
        
        # Try all residues for this prime
        best_residue = None
        best_new_L = _compute_L(current_classes)
        
        for r in range(p):
            test_classes = current_classes + [[p, r]]
            test_L = _compute_L(test_classes)
            if test_L > best_new_L:
                best_new_L = test_L
                best_residue = r
        
        if best_residue is not None:
            current_classes.append([p, best_residue])
            current_budget += cost
            if best_new_L > best_L:
                best_L = best_new_L
                best_classes = list(current_classes)
        
        if time.time() > deadline - 5:
            break
    
    # Try to optimize residues of existing primes to extend L.
    # For each prime in the set, try all residues and keep the best.
    
    if time.time() < deadline - 10:
        # Iterate a few times
        for iteration in range(5):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, (p, r) in enumerate(best_classes):
                best_r = r
                best_L_i = best_L
                for r2 in range(p):
                    if r2 == r:
                        continue
                    test = list(best_classes)
                    test[i] = [p, r2]
                    test_L = _compute_L(test)
                    if test_L > best_L_i:
                        best_L_i = test_L
                        best_r = r2
                if best_r != r:
                    best_classes[i] = [p, best_r]
                    best_L = best_L_i
                    improved = True
            if not improved:
                break
    
    return best_classes


def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _seed_classes(primes):
    seed_primes = [p for p in primes if p >= 13]
    return [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]


def _compute_L(classes):
    """Compute the largest L such that every integer 1..L is covered by some class."""
    if not classes:
        return 0
    
    # For efficiency, use a bitset or set of covered integers.
    # L is capped at 10^5, but we're mostly interested in L up to ~1000.
    max_L = 10000  # reasonable upper bound for search
    
    # Build a boolean array for coverage
    covered = [False] * (max_L + 1)
    
    for p, r in classes:
        # Cover integers n where n ≡ r (mod p), i.e., n = r, r+p, r+2p, ...
        # But n must be >= 1.
        # First integer >= 1 with n ≡ r (mod p):
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p * ((1 - first + p - 1) // p)
        
        for n in range(first, max_L + 1, p):
            covered[n] = True
    
    # Find the largest L such that covered[1..L] are all True
    L = 0
    for n in range(1, max_L + 1):
        if covered[n]:
            L = n
        else:
            break
    
    return L

# EVOLVE-BLOCK-END
