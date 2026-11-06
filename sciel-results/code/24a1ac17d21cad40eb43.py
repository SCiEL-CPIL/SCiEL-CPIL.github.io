"""Narrow admissible tuple: greedy optimization of removed residue classes."""
import math
import random
from collections import defaultdict

N = 5000

def _primes(limit):
    """Return list of primes up to limit (inclusive)."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    n = N
    primes = _primes(n)
    
    # Initialize random residues
    residues = {p: random.randint(0, p - 1) for p in primes}
    
    # cover_count[x] is the number of primes p such that x % p == residues[p]
    cover_count = [0] * (n + 1)
    
    # Initialize cover counts
    for p in primes:
        r = residues[p]
        for x in range(r, n + 1, p):
            cover_count[x] += 1
            
    # Calculate initial score (number of elements with cover_count 0)
    current_score = sum(1 for c in cover_count if c == 0)
    
    # Local search: try to improve score by changing residue for each prime
    # We do multiple passes until no improvement is found
    improved = True
    while improved:
        improved = False
        for p in primes:
            r_old = residues[p]
            best_r = r_old
            best_score = current_score
            
            # Try all other residues
            for r_new in range(p):
                if r_new == r_old:
                    continue
                    
                # Calculate delta in score
                # We need to know how many x change from 0 to 1 or 1 to 0
                # Elements affected are those x where x % p == r_old or x % p == r_new
                
                # To do this efficiently without full re-calculation:
                # 1. Identify x in class r_old. For these, cover_count decreases by 1.
                #    If cover_count[x] becomes 0, score increases by 1.
                # 2. Identify x in class r_new. For these, cover_count increases by 1.
                #    If cover_count[x] was 0, score decreases by 1.
                
                # Note: A number x cannot be in both r_old and r_new classes mod p if r_old != r_new.
                
                delta = 0
                # Process r_old class (removing cover)
                for x in range(r_old, n + 1, p):
                    if cover_count[x] == 1:
                        # Becomes 0, so +1 to score
                        delta += 1
                    # If cover_count[x] > 1, it stays > 0, no change to score
                    # If cover_count[x] == 0, impossible since it's in the class
                
                # Process r_new class (adding cover)
                for x in range(r_new, n + 1, p):
                    if cover_count[x] == 0:
                        # Becomes 1, so -1 to score
                        delta -= 1
                    # If cover_count[x] > 0, it stays > 0, no change to score
                
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_r = r_new
            
            if best_r != r_old:
                # Apply the change
                residues[p] = best_r
                for x in range(r_old, n + 1, p):
                    cover_count[x] -= 1
                for x in range(best_r, n + 1, p):
                    cover_count[x] += 1
                current_score = best_score
                improved = True
    
    # Construct the final tuple
    result = [x for x in range(n + 1) if cover_count[x] == 0]
    
    return result
