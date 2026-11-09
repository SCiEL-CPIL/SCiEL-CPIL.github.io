"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def construct():
    """
    Find a subset of {1..256} such that every pair is coprime (gcd=1),
    maximizing the sum.
    
    Approach:
    - Build a conflict graph where edges connect non-coprime pairs.
    - The problem is Maximum Weight Independent Set (MWIS) on this graph.
    - Use branch-and-bound with a tight upper bound.
    
    Key insight: Since we need pairwise coprime integers, each prime can appear
    in at most one selected number (except 1, which is coprime with everything).
    Actually, more precisely: if two numbers share a prime factor, they conflict.
    
    A simpler and effective approach:
    - Number 1 is always beneficial to include (adds 1, conflicts with nothing).
    - For numbers >= 2, if we select a number n, we cannot select any other number
      that shares a prime factor with n.
    
    We can model this as: assign each number to its set of prime factors.
    Two numbers conflict if their prime factor sets intersect.
    
    For an upper bound in branch-and-bound:
    - Sort candidates by value descending.
    - Greedily pick the largest available number, remove all that conflict.
    - This gives a feasible solution (lower bound).
    - For upper bound: sum of all remaining candidate values (loose but valid).
    
    Better upper bound: Since each prime p can be "used" by at most one selected
    number (the one containing p in its factorization), we can think of it as:
    - Each number n has a "cost" in terms of primes it uses.
    - But the same prime appearing in multiple candidate numbers means we can only
      pick one of those.
    
    Actually, let's use a cleaner formulation:
    - Let S be the set of candidate numbers.
    - We want max sum(S') s.t. for all a,b in S', gcd(a,b)=1.
    
    Branch-and-bound:
    - Order candidates by value descending.
    - At each step, decide to include or exclude the current candidate.
    - If we include it, remove all candidates that share a prime factor with it.
    - Upper bound: current_sum + sum of all remaining candidates' values.
    - To make the bound tighter, we can use a greedy packing heuristic.
    
    Given the small size (256 numbers), and the fact that most large numbers are
    composite and share factors with many others, the search space should be manageable.
    
    Let's implement:
    """
    N = 256
    
    # Compute prime factors for each number 2..N
    def prime_factors(n):
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
    
    # Precompute prime factor sets
    pf = {1: set()}
    for i in range(2, N + 1):
        pf[i] = prime_factors(i)
    
    # Candidates: all numbers 1..N
    # We'll always include 1 since it's coprime with everything and adds value.
    # For numbers 2..N, we need pairwise coprime.
    
    # Let's do branch-and-bound on numbers 2..N, sorted by value descending.
    candidates = list(range(2, N + 1))
    candidates.sort(reverse=True)
    
    # For efficient conflict checking, precompute for each candidate which
    # other candidates it conflicts with (shares at least one prime factor).
    # conflict_mask[i] = bitmask of candidates that conflict with candidates[i]
    
    n_cand = len(candidates)
    # Use bitmasks for efficiency
    conflict = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if pf[candidates[i]] & pf[candidates[j]]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    best_sum = 0
    best_set = []
    
    def branch_and_bound(idx, current_sum, current_mask, current_set):
        nonlocal best_sum, best_set
        
        # Pruning: if current_sum + sum of all remaining <= best_sum, prune
        # Compute upper bound: current_sum + sum of values of all candidates not in current_mask
        # and not yet considered (idx onwards)
        
        # Quick upper bound: current_sum + sum of all candidates from idx to end
        # that are not in current_mask
        ub = current_sum
        for k in range(idx, n_cand):
            if not (current_mask & (1 << k)):
                ub += candidates[k]
        
        if ub <= best_sum:
            return
        
        # If we've considered all candidates
        if idx == n_cand:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
        
        # Option 1: Skip candidates[idx]
        branch_and_bound(idx + 1, current_sum, current_mask, current_set)
        
        # Option 2: Include candidates[idx] if it doesn't conflict with current_mask
        if not (current_mask & (1 << idx)):
            # Include it: remove it and all its conflicts from consideration
            new_mask = current_mask | (1 << idx) | conflict[idx]
            new_set = current_set + [candidates[idx]]
            branch_and_bound(idx + 1, current_sum + candidates[idx], new_mask, new_set)
    
    branch_and_bound(0, 0, 0, [])
    
    # Include 1 in the result
    result = [1] + best_set
    result.sort()
    
    # Ensure types are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
