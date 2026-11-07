import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import math

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

def _local_search(initial_set):
    """
    Improves a set of pairwise-coprime integers using local search.
    Tries to replace subsets of the current set with a single number or add numbers.
    """
    current_set = set(initial_set)
    current_sum = sum(current_set)
    
    # Precompute conflicts for speed
    # conflict[n] is a set of numbers in current_set that conflict with n
    # We'll just check on the fly since N is small.
    
    improved = True
    while improved:
        improved = False
        # Try adding a single number n and removing all conflicts
        for n in range(_MAX_N, 0, -1):
            if n in current_set:
                continue
            
            # Check if n conflicts with any in current_set
            conflicts = []
            for m in current_set:
                if m == 1:
                    continue
                if math.gcd(n, m) > 1:
                    conflicts.append(m)
            
            # Calculate new sum
            new_sum = current_sum + n - sum(conflicts)
            
            if new_sum > current_sum:
                # Apply change
                for m in conflicts:
                    current_set.remove(m)
                current_set.add(n)
                current_sum = new_sum
                improved = True
                # Restart loop to allow cascading improvements
                break
        
        # Also try removing a number if it doesn't help? No, we always want to maximize sum.
        # But maybe removing a number allows adding two others?
        # The above loop handles adding 1 and removing k.
        # It does not handle removing 1 and adding 1 (swap) if the swap is not an "add and remove conflicts".
        # Actually, "add n and remove conflicts" covers swaps if the conflicts are exactly the ones being replaced.
        # If we want to swap A with B, and A and B don't conflict with others, 
        # then adding B will conflict with A. So it removes A and adds B.
        # This works.
        
        # Try removing a number that is "bad"? 
        # If we have a number n, and we can replace it with a larger number m that conflicts with n?
        # That is covered by adding m.
        
        # What if we need to remove a number to free up a prime for a better combination?
        # The "add n" step effectively does this.
        
        # One more check: try to remove a number if it allows adding multiple numbers?
        # This is a 2-opt move.
        # Given the small size, we can try simple 2-opt.
        for n in list(current_set):
            if n == 1:
                continue
            # Try removing n
            temp_set = current_set - {n}
            temp_sum = current_sum - n
            
            # Try adding a number m to temp_set
            for m in range(_MAX_N, 0, -1):
                if m in temp_set:
                    continue
                # Check conflicts with temp_set
                conflicts = []
                for k in temp_set:
                    if k == 1:
                        continue
                    if math.gcd(m, k) > 1:
                        conflicts.append(k)
                
                new_sum = temp_sum + m - sum(conflicts)
                
                if new_sum > current_sum:
                    # Apply: remove n and conflicts, add m
                    current_set = temp_set - set(conflicts)
                    current_set.add(m)
                    current_sum = new_sum
                    improved = True
                    break
            if improved:
                break
        if improved:
            break
            
    # Ensure 1 is in the set
    if 1 not in current_set:
        current_set.add(1)
        current_sum += 1
        
    return list(current_set)

def construct():
    """
    Solves the maximum sum of pairwise-coprime integers from 1..256 using Integer Linear Programming,
    with a local search fallback and polishing.
    """
    n_vars = _MAX_N
    c = -np.arange(1, _MAX_N + 1, dtype=float)
    A = np.zeros((_NUM_PRIMES, n_vars), dtype=float)
    
    for i in range(1, _MAX_N + 1):
        for p in _prime_factors[i]:
            A[_prime_index[p], i - 1] = 1.0
            
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars)
    constraint = LinearConstraint(A, ub=np.ones(_NUM_PRIMES))
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        result = []
        for i in range(n_vars):
            if x[i] > 0.5:
                result.append(i + 1)
        if 1 not in result:
            result.append(1)
        # Polish with local search to ensure optimality or fix numerical issues
        result = _local_search(result)
        return [int(v) for v in result]
    else:
        # Fallback: Start with primes and local search
        initial = [p for p in _PRIMES]
        result = _local_search(initial)
        return [int(v) for v in result]
