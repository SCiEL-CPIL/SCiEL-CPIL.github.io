"""Rewritten program: Two-phase Greedy (Max Coverage for 1000, then Extension)."""
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

def construct():
    """
    Find a set of distinct primes p <= 1000 with residues r (0 <= r < p)
    such that sum(1/p) <= 1 and every integer 1..L is covered.
    Maximize L.
    """
    L_target = 1000
    L_limit = 10000 # For extension phase
    
    # Phase 1: Greedy Max Coverage for 1..L_target
    # is_uncovered for 1..L_target
    is_uncovered = bytearray([1]) * (L_target + 1)
    uncovered_list = list(range(1, L_target + 1))
    
    remaining_budget = Fraction(1, 1)
    used_primes = set()
    solution = []
    
    while True:
        best_gain = 0
        best_p = -1
        best_r = -1
        
        if not uncovered_list:
            break
            
        for p in PRIMES_1000:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if cost > remaining_budget:
                continue
            
            # Count coverage for each residue
            counts = [0] * p
            for u in uncovered_list:
                r = u % p
                counts[r] += 1
            
            max_r = 0
            max_count = 0
            for r in range(p):
                if counts[r] > max_count:
                    max_count = counts[r]
                    max_r = r
            
            if max_count > best_gain:
                best_gain = max_count
                best_p = p
                best_r = max_r
        
        if best_p == -1:
            break
        
        used_primes.add(best_p)
        solution.append([best_p, best_r])
        remaining_budget -= Fraction(1, best_p)
        
        # Update uncovered
        new_uncovered = []
        for u in uncovered_list:
            if u % best_p != best_r:
                new_uncovered.append(u)
            else:
                is_uncovered[u] = 0
        uncovered_list = new_uncovered
        
        if not uncovered_list:
            break

    # Phase 2: Extension
    # Check current coverage up to L_limit
    # Re-initialize is_uncovered for the larger range?
    # We can just compute coverage for 1..L_limit based on current solution.
    
    is_uncovered_large = bytearray([1]) * (L_limit + 1)
    for p, r in solution:
        start = r if r > 0 else p
        for m in range(start, L_limit + 1, p):
            is_uncovered_large[m] = 0
            
    # Find current L
    current_L = 0
    for i in range(1, L_limit + 1):
        if is_uncovered_large[i] == 0:
            current_L = i
        else:
            break
            
    # Extend
    # We need to cover current_L + 1, current_L + 2, etc.
    # We will iterate n from current_L + 1 to L_limit.
    # If n is uncovered, we must cover it.
    # We choose the largest unused prime p (cheapest) that fits in budget.
    # Why largest? Because it has the smallest 1/p, preserving budget for future extensions.
    # Also, large primes cover sparse sets, so they are less likely to interfere?
    # Actually, for extension, we just need to cover the specific hole.
    
    # Sort unused primes in descending order to pick the largest (cheapest) first
    unused_primes = [p for p in PRIMES_1000 if p not in used_primes]
    unused_primes.sort(reverse=True)
    
    # We need to be careful: picking a prime for n might cover other future numbers, which is good.
    # But we must ensure we don't "waste" a prime on a number that could be covered by a smaller, more "flexible" prime?
    # No, once a prime is used, its residue is fixed.
    # If we pick a large prime for n, it covers n, n+p, n+2p...
    # If we pick a small prime, it covers n, n+p, n+2p... (denser).
    # Denser coverage is better for future holes.
    # So maybe we should pick the *smallest* unused prime that fits in budget?
    # But small primes are expensive.
    # Let's stick with "Largest unused prime that fits" to conserve budget, 
    # but if we have lots of budget, maybe use smaller ones?
    # Given the tight budget (1.0), conserving budget is key.
    
    for n in range(current_L + 1, L_limit + 1):
        if is_uncovered_large[n] == 0:
            continue
            
        # Find a prime to cover n
        selected_p = -1
        selected_r = -1
        
        # Try largest primes first (cheapest)
        for p in unused_primes:
            cost = Fraction(1, p)
            if cost <= remaining_budget:
                # Check if this prime is still unused (it is, by list construction)
                # Calculate r
                r = n % p
                # We can use this prime
                selected_p = p
                selected_r = r
                break
        
        if selected_p == -1:
            # No prime can cover n within budget
            break
        
        # Apply selection
        used_primes.add(selected_p)
        solution.append([selected_p, selected_r])
        remaining_budget -= Fraction(1, selected_p)
        
        # Remove from unused_primes list? 
        # It's a list, removal is O(N). N=168. Fine.
        unused_primes.remove(selected_p)
        
        # Update coverage
        start = selected_r if selected_r > 0 else selected_p
        for m in range(start, L_limit + 1, selected_p):
            is_uncovered_large[m] = 0

    # Calculate final L
    final_L = 0
    for i in range(1, L_limit + 1):
        if is_uncovered_large[i] == 0:
            final_L = i
        else:
            break
            
    return solution

# Test
if __name__ == "__main__":
    sol = construct()
    print(f"Number of classes: {len(sol)}")
    L_limit = 10000
    is_uncovered_large = bytearray([1]) * (L_limit + 1)
    for p, r in sol:
        start = r if r > 0 else p
        for m in range(start, L_limit + 1, p):
            is_uncovered_large[m] = 0
    L = 0
    for i in range(1, L_limit + 1):
        if is_uncovered_large[i] == 0:
            L = i
        else:
            break
    print(f"Covered length L: {L}")
    total = sum(Fraction(1, p) for p, r in sol)
    print(f"Total budget used: {total}")
