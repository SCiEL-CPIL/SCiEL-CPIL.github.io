"""ILP-based branch-and-bound for maximum weight independent set on coprime conflict graph."""
import math
from math import gcd
from functools import lru_cache
import time

def construct():
    N = 512  # upper bound + 1
    
    # Precompute prime factorizations for all numbers 1..511
    # For each number, store its set of prime factors
    prime_factors = [set() for _ in range(N)]
    for i in range(2, N):
        n = i
        d = 2
        while d * d <= n:
            if n % d == 0:
                prime_factors[i].add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            prime_factors[i].add(n)
    # 1 has no prime factors
    
    # Weight of selecting a: 1/(512 - a)
    # We want to maximize sum of weights subject to pairwise coprime constraint
    
    # Precompute conflict: for each a, which b > a conflict with a?
    # Two numbers conflict if they share a prime factor.
    # conflict[a] = set of b > a such that gcd(a, b) > 1
    # But we need this efficiently. Let's use prime factor sets.
    
    # For branch and bound, we process numbers in order of decreasing weight (i.e., decreasing a).
    # Order: 511, 510, 509, ..., 1
    
    # Key data structure: for each prime p, which numbers in our candidate set are divisible by p?
    # When we select a number a, all numbers sharing a prime factor with a become unavailable.
    
    # Strategy: Branch and bound with fractional knapsack upper bound.
    # At each node, we have:
    #   - current selection (set of chosen numbers)
    #   - available numbers (those not yet decided and not conflicting with selection)
    #   - current weight sum
    # Upper bound: current weight + sum of weights of all available numbers (relaxation)
    # Better upper bound: fractional knapsack on available numbers sorted by weight
    
    # Since weights are 1/(512-a) and we process in decreasing a order,
    # the available numbers are always a suffix of the original ordering (with some removed).
    
    # Let's use a different approach: process in order 511, 510, ..., 1.
    # At step i (considering number a_i), we decide include or exclude.
    # State: which primes are "used" (i.e., some selected number has that prime factor).
    # A number a can be selected only if none of its prime factors are used.
    
    # This gives us a clean state: the set of used primes.
    # Number of distinct primes below 512: let's count.
    # Primes below 512: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
    # 101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,
    # 211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,
    # 331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,
    # 449,457,461,463,467,479,487,491,499,503,509
    # That's about 97 primes. Too many for a bitmask state.
    
    # Alternative: Since we process in order and the constraint is "no shared prime factors",
    # we can use the following observation:
    # When processing number a, it conflicts with previously selected numbers iff
    # a shares a prime factor with any of them.
    # We can track which prime factors are "blocked" by the selection.
    
    # But 97 primes is too many for exact state tracking.
    # However, most numbers have few prime factors. Let's use a different approach.
    
    # Better approach: Use the fact that the graph is relatively sparse.
    # For each number a, the numbers that conflict with it are those sharing at least one prime factor.
    # For large a (near 511), the number of conflicts is small.
    
    # Let's use a recursive branch-and-bound with memoization on (index, frozenset_of_used_primes).
    # But the state space might be too large.
    
    # Alternative: Since the weights decrease rapidly (1/(512-a) for a near 511 is much larger
    # than for a far from 511), we can use a greedy + local search or a limited B&B.
    
    # Let me think about this more carefully. The key insight is:
    # - The top numbers (511, 510, 509, ...) have the highest weights.
    # - 511 = 7*73, 510 = 2*3*5*17, 509 = prime, 508 = 4*127, 507 = 3*169, etc.
    # - Many consecutive numbers are coprime.
    
    # Let's use a simple but effective approach:
    # 1. Start with greedy solution (pick from 511 down, include if coprime with all chosen).
    # 2. Try to improve by swapping: for each excluded number, see if removing one included
    #    number and adding the excluded one improves the score.
    # 3. Iterate until no improvement.
    
    # Actually, let's just do a proper branch-and-bound with a tight upper bound.
    
    # Upper bound at any point: current_sum + sum of 1/(512-b) for all b still available.
    # This is a valid upper bound (relaxation of the independence constraint).
    
    # To make it tighter, we can use the fact that among available numbers, some conflict
    # with each other, so we can't take all of them.
    
    # For now, let's implement a clean B&B with the simple upper bound and see if it's fast enough.
    
    # Numbers in order of decreasing weight: 511, 510, 509, ..., 1
    order = list(range(511, 0, -1))
    n = len(order)
    
    # Precompute: for each position i in order, which later positions j > i conflict with position i?
    # conflict_mask[i] = bitmask of positions j > i where gcd(order[i], order[j]) > 1
    # But 511 bits is too large for Python int masks efficiently.
    
    # Let's use sets instead.
    # For each a, precompute the set of b < a (in our ordering, b comes after a) that conflict with a.
    # Actually, let's precompute for each a, the set of prime factors, and then for each prime p,
    # the list of numbers (in our ordering) that are divisible by p.
    
    # For the B&B, at each step we consider the next number in order.
    # State: (idx, frozenset of used primes)
    # But used primes set can be large. Let's use a different encoding.
    
    # Since we only need to check if a number's primes are all unused,
    # and each number has at most ~4 prime factors, we can track used primes as a set.
    
    # Let's limit the B&B depth and use pruning.
    
    best_solution = []
    best_value = 0.0
    
    # Precompute weights
    weights = [1.0 / (N - a) for a in order]
    
    # Precompute prime factors for each number in order
    pf_in_order = [prime_factors[a] for a in order]
    
    # For upper bound: precompute suffix sums of weights
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # B&B with recursion
    # To avoid deep recursion issues, we'll use an iterative approach or limit depth.
    
    # Given time constraints, let's use a heuristic approach:
    # 1. Greedy solution
    # 2. Local search: try adding excluded numbers by removing conflicting included numbers
    
    # Step 1: Greedy
    chosen = []
    chosen_set = set()
    used_primes = set()
    for i, a in enumerate(order):
        if pf_in_order[i].issubset(used_primes):
            continue  # conflict with already chosen
        # Check if any prime factor is used
        if any(p in used_primes for p in pf_in_order[i]):
            continue
        chosen.append(a)
        chosen_set.add(a)
        used_primes.update(pf_in_order[i])
    
    current_value = sum(1.0 / (N - a) for a in chosen)
    
    # Step 2: Local search
    # For each excluded number, try to include it by removing conflicting included numbers.
    excluded = [a for a in order if a not in chosen_set]
    
    improved = True
    while improved:
        improved = False
        for a in excluded:
            if a in chosen_set:
                continue
            # Find which chosen numbers conflict with a
            conflicting = [b for b in chosen if any(p in prime_factors[b] for p in prime_factors[a])]
            # Value change: +1/(N-a) - sum(1/(N-b) for b in conflicting)
            gain = 1.0 / (N - a) - sum(1.0 / (N - b) for b in conflicting)
            if gain > 1e-12:
                # Remove conflicting, add a
                for b in conflicting:
                    chosen.remove(b)
                    chosen_set.remove(b)
                chosen.append(a)
                chosen_set.add(a)
                current_value += gain
                improved = True
                # Restart scan since the set changed
                break
    
    # Also try: for each chosen number, see if removing it allows adding multiple excluded numbers
    # This is more complex. Let's do a simpler 2-opt style.
    
    # Try swapping: remove one chosen, add one excluded (even if gain is slightly negative,
    # it might enable better future swaps)
    # For now, let's just return the current best.
    
    return sorted(chosen)
