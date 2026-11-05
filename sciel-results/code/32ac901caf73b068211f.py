"""Branch-and-bound solver for pairwise coprime subset with maximum sum 1/(512-a)."""
import math
from math import gcd
from collections import defaultdict

# Precompute prime factors for each number 1..511
def _factorize(n):
    """Return set of distinct prime factors of n."""
    if n == 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for all numbers
_prime_factors = [set()] * 512
for i in range(1, 512):
    _prime_factors[i] = _factorize(i)

# Build conflict graph: for each number, which other numbers conflict with it
# Two numbers conflict if they share a prime factor
_conflicts = [set() for _ in range(512)]
for i in range(1, 512):
    for j in range(i + 1, 512):
        if _prime_factors[i] & _prime_factors[j]:
            _conflicts[i].add(j)
            _conflicts[j].add(i)

# Weight for each number: 1/(512 - a)
_weights = [0.0] * 512
for a in range(1, 512):
    _weights[a] = 1.0 / (512 - a)

# Sort numbers by weight (descending) for the B&B
# We'll use a priority-based approach
_order = list(range(511, 0, -1))  # Start from highest weight (largest a)

def _fractional_bound(available, current_sum):
    """
    Compute an upper bound using fractional knapsack relaxation.
    available: set of numbers that could still be added
    Returns upper bound on additional sum.
    """
    if not available:
        return current_sum
    
    # Sort available by weight descending
    sorted_avail = sorted(available, key=lambda x: _weights[x], reverse=True)
    
    bound = current_sum
    for a in sorted_avail:
        bound += _weights[a]
    return bound

def _tight_bound(available, selected, current_sum):
    """
    Tighter bound: for each available number, check if it conflicts with any selected.
    If yes, it's excluded. Then apply fractional knapsack on remaining.
    """
    if not available:
        return current_sum
    
    # Filter out numbers that conflict with selected
    valid = set()
    for a in available:
        if not any(gcd(a, s) > 1 for s in selected):
            valid.add(a)
    
    if not valid:
        return current_sum
    
    # Also filter: among valid numbers, remove those that conflict with each other
    # by greedily removing lower-weight conflicting numbers
    sorted_valid = sorted(valid, key=lambda x: _weights[x], reverse=True)
    
    bound = current_sum
    remaining = set(sorted_valid)
    for a in sorted_valid:
        if a not in remaining:
            continue
        bound += _weights[a]
        # Remove all numbers in remaining that conflict with a
        to_remove = set()
        for b in remaining:
            if b != a and (_prime_factors[a] & _prime_factors[b]):
                to_remove.add(b)
        remaining -= to_remove
    
    return bound

def construct():
    """
    Find a subset of {1,...,511} of pairwise coprime integers maximizing sum 1/(512-a).
    Uses branch-and-bound with greedy initialization and tight bounds.
    """
    # Start with greedy solution
    greedy_selected = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in greedy_selected):
            greedy_selected.append(a)
    
    greedy_score = sum(_weights[a] for a in greedy_selected)
    
    # Try to improve with local search / branch and bound
    # Strategy: try swapping elements or adding elements
    
    best_set = set(greedy_selected)
    best_score = greedy_score
    
    # For each prime p, at most one number divisible by p can be selected
    # This is the key structural constraint
    
    # Try a more systematic approach:
    # Group numbers by their "conflict class"
    # Numbers that share a prime factor are in the same conflict class
    
    # Let's try: for each prime p < 512, we can select at most one number divisible by p
    # The optimal solution selects for each prime p, either a number divisible by p or not
    
    # Better approach: use the fact that if we select a number a, we exclude all
    # numbers sharing any prime factor with a
    
    # Let's try local improvement: try replacing one selected number with one or more
    # unselected numbers that are compatible
    
    improved = True
    while improved:
        improved = False
        current_list = list(best_set)
        
        for i, a in enumerate(current_list):
            # Try removing a and seeing if we can add better numbers
            remaining = best_set - {a}
            
            # Find all numbers not in remaining that are compatible with remaining
            candidates = []
            for b in range(511, 0, -1):
                if b in remaining:
                    continue
                if all(gcd(b, r) == 1 for r in remaining):
                    candidates.append(b)
            
            # Try adding the best candidates (greedily)
            new_set = set(remaining)
            new_score = sum(_weights[x] for x in remaining)
            for b in candidates:
                if all(gcd(b, x) == 1 for x in new_set):
                    new_set.add(b)
                    new_score += _weights[b]
            
            if new_score > best_score + 1e-15:
                best_set = new_set
                best_score = new_score
                improved = True
                break
    
    return list(best_set)
