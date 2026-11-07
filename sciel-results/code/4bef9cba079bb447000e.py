"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import sys
from functools import lru_cache

# Precompute primes up to 256 using sieve
_MAX_N = 256

def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

_PRIMES = _sieve(_MAX_N)
_NUM_PRIMES = len(_PRIMES)

# Precompute prime factors for each number 1..256
# prime_factors[i] = list of distinct prime factors of i
_prime_factors = [[] for _ in range(_MAX_N + 1)]
for i in range(2, _MAX_N + 1):
    x = i
    factors = []
    for p in _PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            factors.append(p)
            while x % p == 0:
                x //= p
    if x > 1:
        factors.append(x)
    _prime_factors[i] = factors

# Map each prime to its index
_prime_index = {p: i for i, p in enumerate(_PRIMES)}

# For each number, compute a bitmask of its prime factors
# We use Python integers as bitmasks
_number_mask = [0] * (_MAX_N + 1)
for i in range(2, _MAX_N + 1):
    mask = 0
    for p in _prime_factors[i]:
        mask |= (1 << _prime_index[p])
    _number_mask[i] = mask

# For each number, store its value and mask
# We want to select a subset of numbers from 1..256 such that:
# - All selected numbers are distinct
# - Every pair is coprime (gcd = 1)
# - Maximize the sum

# Strategy: This is a weighted independent set problem on a conflict graph,
# where two numbers conflict if they share a prime factor.
# Since the numbers are small (up to 256) and we have ~54 primes,
# we can use a branch-and-bound or greedy approach with backtracking.

# Better approach: Since we need pairwise coprime, we can think of it as:
# Each prime can be "used" by at most one selected number.
# This is like an assignment problem: assign primes to numbers, but each number
# can take multiple primes, and each prime can go to at most one number.

# Actually, this is a weighted set packing problem: each number is a set of primes,
# and we want to maximize the sum of weights (the number values) such that no two sets overlap.

# Given the small size, we can use a greedy approach with backtracking, or we can
# use a more clever observation.

# Observation: For any prime p, we can include at most one multiple of p.
# To maximize the sum, for each prime p, we should consider including the largest
# multiple of p that doesn't conflict with other choices.

# However, numbers can have multiple prime factors, so this is complex.

# Let's use a different approach: since the universe is small (256 numbers),
# we can use a recursive search with pruning.

# But 256 numbers is too many for brute force. Let's think differently.

# Key insight: If we include a composite number, we "use" all its prime factors.
# It's often better to include the primes themselves rather than composites,
# because a prime p gives us value p and uses only prime p, while a composite
# like p*q gives us value p*q but uses both primes p and q.

# Wait, let's compare: if we take prime p and prime q separately, we get p+q.
# If we take p*q, we get p*q. For p,q >= 2, p*q >= p+q (with equality only for 2,2 but they're not distinct).
# Actually 2*3=6 > 2+3=5. So composites can be better!

# But we can only take one multiple of p and one multiple of q. If we take p*q, we use both p and q.
# So the question is: is it better to take p and q separately (sum p+q) or p*q (value p*q)?
# Since p*q >= p+q for p,q >= 2, taking the composite is better if we can.

# But we can't always take all composites because of conflicts.

# Let's think about this as a maximum weight independent set in a graph where
# vertices are numbers 1..256 and edges connect numbers that share a prime factor.

# Given the small size, let's try a greedy approach with local search, or a
# more systematic approach.

# Actually, let's reconsider. The number of primes up to 256 is 54.
# Each number uses a subset of these primes. We want to select numbers such that
# no prime is used more than once.

# This is equivalent to: for each prime, at most one selected number is divisible by it.

# We can model this as a constraint satisfaction / optimization problem.

# Given the small search space, let's use a recursive backtracking with memoization
# or a greedy approach.

# Let's try a greedy approach: sort numbers by value descending, and greedily
# add a number if it doesn't conflict with already-selected numbers.

def construct():
    """Return a list of pairwise coprime integers from 1..256 maximizing their sum."""
    # Greedy approach: try numbers in descending order of value
    selected = []
    used_primes = set()
    
    for num in range(_MAX_N, 0, -1):
        if num == 1:
            # 1 is coprime with everything, always include it
            selected.append(1)
            continue
        # Check if num conflicts with any already selected number
        conflict = False
        for p in _prime_factors[num]:
            if p in used_primes:
                conflict = True
                break
        if not conflict:
            selected.append(num)
            used_primes.update(_prime_factors[num])
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in selected]
    return result

# Let's test and potentially improve with a better algorithm
# The greedy approach may not be optimal. Let's try a more systematic approach.

# Better approach: Since the problem is small, we can use a branch and bound
# or a dynamic programming approach over primes.

# Actually, let's think about this more carefully. The key constraint is that
# each prime can be used by at most one number. This is a set packing problem.

# For a better solution, let's use a recursive search with pruning, but only
# over the "candidate" numbers. We can prune by noting that if a number's
# prime factors are all already used, we can't select it.

# Let's try a different greedy: for each prime, decide which multiple to take.
# But this is complex due to interactions.

# Let's stick with the greedy for now, but try to improve it by considering
# that we might want to skip a large number to take two slightly smaller ones.

# Actually, let's just return the greedy result for now, as it's a reasonable
# starting point.
# EVOLVE-BLOCK-END
