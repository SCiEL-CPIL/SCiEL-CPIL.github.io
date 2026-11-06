"""Rewritten program for covering 1..L by prime residue classes with reciprocal budget 1."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_L(covered, cap=100000):
    """Compute the largest L such that all of 1..L are covered."""
    # covered is a set of integers
    L = 0
    for i in range(1, cap + 1):
        if i in covered:
            L = i
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    # Sort primes: we want to use larger primes first for coverage efficiency
    # but smaller primes cover more residues per unit budget... 
    # Actually, for covering a prefix, we want to be strategic.
    
    # Target: cover as much of [1, L] as possible.
    # Budget: sum(1/p) <= 1
    # 
    # Strategy: Greedy with lookahead.
    # At each step, pick (p, r) that maximizes the number of NEW integers covered in [1, current_max + buffer]
    # while keeping the remaining budget feasible.
    
    # Precompute: for each prime p, the residues r in [0, p-1]
    # Coverage of residue r mod p covers {r, r+p, r+2p, ...}
    # For covering [1, L], residue r covers numbers congruent to r mod p.
    
    # Let's use a different approach:
    # We want to cover [1, L]. The "gap" is the first uncovered number.
    # Greedy: at each step, find the (p, r) that covers the most new numbers in the prefix.
    
    # For efficiency, track covered as a boolean array up to some limit.
    MAX_L = 10000  # we'll track coverage up to this
    covered = [False] * (MAX_L + 1)
    
    budget = Fraction(1, 1)
    used_classes = []
    
    # For each prime, precompute which numbers in [1, MAX_L] each residue covers
    # residue r mod p covers: r, r+p, r+2p, ... (but 0 mod p covers p, 2p, 3p, ...)
    # Actually residue r means numbers n where n % p == r. If r=0, that's p, 2p, etc.
    # But we need to cover 1..L, so residue 0 covers p, 2p, ... (not 0)
    
    # Let's think differently. For a given prime p and residue r (0 <= r < p),
    # the numbers covered in [1, MAX_L] are: all n in [1, MAX_L] with n % p == r.
    # If r == 0: p, 2p, 3p, ...
    # If r > 0: r, r+p, r+2p, ... (but r must be >= 1 for it to be in [1, MAX_L])
    
    # Greedy selection:
    # At each iteration, for each unused prime p and each residue r, compute how many NEW
    # numbers in [1, MAX_L] would be covered. Pick the one with max new coverage.
    
    # To speed up, we can limit the search to primes that still have budget available.
    
    # Precompute coverage sets for each (p, r)
    coverage_map = {}  # (p, r) -> list of numbers covered
    for p in primes:
        for r in range(p):
            nums = []
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, MAX_L + 1, p):
                nums.append(n)
            coverage_map[(p, r)] = nums
    
    # Sort primes by size for ordering
    # Use a priority approach: try to cover the gap (first uncovered number)
    
    while time.time() < deadline:
        # Find the first uncovered number
        first_uncovered = None
        for i in range(1, MAX_L + 1):
            if not covered[i]:
                first_uncovered = i
                break
        
        if first_uncovered is None:
            break  # All covered up to MAX_L
        
        # Find best (p, r) that covers first_uncovered and maximizes new coverage
        best_score = -1
        best_pr = None
        best_new = 0
        
        # For efficiency, only consider primes where first_uncovered % p gives a valid residue
        # and where we have budget
        for p in primes:
            if budget < Fraction(1, p):
                continue
            r = first_uncovered % p
            # Check if this class is already used
            if any(c[0] == p and c[1] == r for c in used_classes):
                continue
            # Count new coverage
            nums = coverage_map[(p, r)]
            new_count = 0
            for n in nums:
                if not covered[n]:
                    new_count += 1
            if new_count > best_score:
                best_score = new_count
                best_pr = (p, r)
                best_new = new_count
        
        if best_pr is None:
            # Try any (p, r) that gives good coverage even if not covering first_uncovered
            for p in primes:
                if budget < Fraction(1, p):
                    continue
                for r in range(p):
                    if any(c[0] == p and c[1] == r for c in used_classes):
                        continue
                    nums = coverage_map[(p, r)]
                    new_count = 0
                    for n in nums:
                        if not covered[n]:
                            new_count += 1
                    if new_count > best_score:
                        best_score = new_count
                        best_pr = (p, r)
                        best_new = new_count
        
        if best_pr is None or best_score <= 0:
            break
        
        p, r = best_pr
        # Apply
        for n in coverage_map[(p, r)]:
            covered[n] = True
        budget -= Fraction(1, p)
        used_classes.append([p, r])
    
    # Compute L
    L = _compute_L(set(i for i in range(1, MAX_L + 1) if covered[i]))
    
    return used_classes

# EVOLVE-BLOCK-END
