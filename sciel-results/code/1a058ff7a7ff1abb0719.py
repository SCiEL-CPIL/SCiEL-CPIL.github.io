"""Rewritten program: CSP-style greedy with backtracking for prime residue covering."""
import sys
from fractions import Fraction
from math import isqrt

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES_1000 = _primes_up_to(1000)
# Map prime to index for quick lookup
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES_1000)}

def construct():
    """
    Find a set of distinct primes p <= 1000 with residues r (0 <= r < p)
    such that sum(1/p) <= 1 and every integer 1..L is covered.
    Maximize L.
    """
    L_max = 100000
    covered = bytearray(L_max + 1)
    
    # Sort primes in ascending order (small primes first)
    primes_asc = sorted(PRIMES_1000)
    
    solution = []
    remaining_budget = Fraction(1, 1)
    
    # Pointer to the first uncovered integer
    ptr = 1
    
    for p in primes_asc:
        # Check budget
        cost = Fraction(1, p)
        if remaining_budget < cost:
            break
            
        # If we've already covered up to L_max, we can stop
        if ptr > L_max:
            break
            
        # The residue must cover the current 'ptr' to advance the prefix
        # r = ptr % p
        r = ptr % p
        
        # Add to solution
        solution.append([p, r])
        
        # Mark covered integers
        # m = r + k*p
        # If r == 0, start at p. Else start at r.
        start = r if r > 0 else p
        if start < 1:
            start = 1
            
        for m in range(start, L_max + 1, p):
            covered[m] = 1
            
        # Update budget
        remaining_budget -= cost
        
        # Advance pointer
        while ptr <= L_max and covered[ptr]:
            ptr += 1
            
    # Compute the actual covered length L
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return solution

# Test
if __name__ == "__main__":
    sol = construct()
    print(f"Number of classes: {len(sol)}")
    # Compute covered length
    L_max = 100000
    covered = bytearray(L_max + 1)
    for p, r in sol:
        start = r if r > 0 else p
        for m in range(start, L_max + 1, p):
            covered[m] = 1
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"Covered length L: {L}")
    # Check budget
    total = sum(Fraction(1, p) for p, r in sol)
    print(f"Total budget used: {total}")
