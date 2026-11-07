import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_MAX_N = 256

# Precompute primes
def _sieve(n):
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

_prime_index = {p: i for i, p in enumerate(_PRIMES)}

def construct():
    """
    Solves the maximum sum of pairwise-coprime integers from 1..256 using Integer Linear Programming.
    Uses variable reduction: for each set of prime factors, only the largest number is kept.
    """
    # 1. Reduce variables
    # Group numbers by their prime factor tuple
    factor_groups = {}
    for i in range(2, _MAX_N + 1):
        key = tuple(_prime_factors[i])
        if key not in factor_groups:
            factor_groups[key] = i
        else:
            if i > factor_groups[key]:
                factor_groups[key] = i
        
    # The reduced set of candidates
    # Note: 1 has no prime factors, key is empty tuple
    # We handle 1 separately or include it in reduction?
    # 1 is coprime to everything. It doesn't conflict.
    # It's always optimal to include 1.
    # We can just add 1 to the final result and not include it in the ILP.
    
    candidates = list(factor_groups.values())
    # Ensure 1 is not in candidates (it wasn't in the loop 2..256)
    # So candidates are from 2..256
    
    n_vars = len(candidates)
    
    if n_vars == 0:
        return [1]
        
    # Objective: maximize sum(i * x_i)
    # milp minimizes, so we use c = -i
    c = -np.array(candidates, dtype=float)
    
    # Constraints: For each prime p, sum(x_i for i divisible by p) <= 1
    # A is _NUM_PRIMES x n_vars
    A = np.zeros((_NUM_PRIMES, n_vars), dtype=float)
    
    for idx, num in enumerate(candidates):
        for p in _prime_factors[num]:
            A[_prime_index[p], idx] = 1.0
            
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all variables are integer (0 or 1)
    integrality = np.ones(n_vars)
    
    # Linear Constraint: A x <= 1
    constraint = LinearConstraint(A, ub=np.ones(_NUM_PRIMES))
    
    # Set time limit to ensure we don't hang
    options = {'time_limit': 55}
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds, options=options)
    
    if res.status != 0: # 0 is optimal
        # Fallback to greedy if ILP fails
        # Use a better greedy: sort by value, pick if no conflict
        # This is the same as before but on the reduced set?
        # No, greedy should be on the full set to be safe?
        # Actually, if ILP fails, we can just use the greedy on the full set as before.
        selected = []
        used_primes = set()
        for num in range(_MAX_N, 0, -1):
            if num == 1:
                selected.append(1)
                continue
            conflict = False
            for p in _prime_factors[num]:
                if p in used_primes:
                    conflict = True
                    break
            if not conflict:
                selected.append(num)
                used_primes.update(_prime_factors[num])
        return [int(x) for x in selected]
        
    x = res.x
    result = []
    for i in range(n_vars):
        if x[i] > 0.5:
            result.append(candidates[i])
            
    # Ensure 1 is included
    result.append(1)
        
    # Convert to list of ints
    return [int(v) for v in result]
