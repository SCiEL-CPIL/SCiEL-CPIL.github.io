import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_start = time.time()
_DEADLINE = 85

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    max_val = 256
    primes = [p for p in range(2, max_val + 1) if _is_prime(p)]
    
    # Split primes into small (<=128) and large (>128)
    small_primes = [p for p in primes if p <= 128]
    large_primes = [p for p in primes if p > 128]
    
    # Large primes are always included as they don't conflict with anything else
    # and are the only multiples of themselves in range.
    result_set = set(large_primes)
    
    # U: Numbers in 1..256 that are NOT large primes.
    # These are composed of small primes.
    U = [x for x in range(1, max_val + 1) if x not in large_primes]
    
    # Setup ILP
    # Variables: z_i for x_i in U
    n_vars = len(U)
    c = np.array([-x for x in U])  # Minimize -sum
    
    # Constraints: For each small prime p, sum(z_i for x_i divisible by p) <= 1
    # A matrix: rows are small_primes, cols are U
    A = np.zeros((len(small_primes), n_vars))
    for i, p in enumerate(small_primes):
        for j, x in enumerate(U):
            if x % p == 0:
                A[i, j] = 1
                
    constraints = LinearConstraint(A, ub=np.ones(len(small_primes)))
    bounds = Bounds(lb=0, ub=1)
    
    ilp_success = False
    try:
        res = milp(c=c, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
        if res.success:
            # Extract solution
            for j, x in enumerate(U):
                if res.x[j] > 0.5:
                    result_set.add(int(x))
            ilp_success = True
        else:
            # Fallback to greedy if ILP fails
            raise Exception("ILP failed")
    except Exception:
        # Fallback: Greedy + Local Search
        # Initialize with largest prime powers
        current_set = set()
        # Add 1
        current_set.add(1)
        
        # Greedy: sort U by value desc
        for x in sorted(U, reverse=True):
            # Check if x is coprime to all in current_set
            ok = True
            for y in current_set:
                if gcd(x, y) != 1:
                    ok = False
                    break
            if ok:
                current_set.add(x)
        
        result_set = current_set

    # Post-optimization local search on the ILP or fallback result
    if time.time() < _DEADLINE - 5:
        # Local search: try to improve by swaps and additions
        improved = True
        while improved and time.time() < _DEADLINE - 3:
            improved = False
            current_list = list(result_set)
            
            # Try adding any number that's coprime to all
            for x in U:
                if x in result_set:
                    continue
                ok = True
                for y in result_set:
                    if gcd(x, y) != 1:
                        ok = False
                        break
                if ok:
                    result_set.add(x)
                    improved = True
            
            # Try swapping: remove y, add x where x > y
            for y in list(result_set):
                if y == 1:
                    continue
                # Try to find x > y such that x is coprime to current_set - {y}
                rest = result_set - {y}
                best_x = None
                best_val = y
                for x in U:
                    if x in rest or x == y:
                        continue
                    if x <= best_val:
                        continue
                    ok = True
                    for z in rest:
                        if gcd(x, z) != 1:
                            ok = False
                            break
                    if ok:
                        best_x = x
                        best_val = x
                if best_x is not None:
                    result_set.remove(y)
                    result_set.add(best_x)
                    improved = True
            
            # Try double swaps: remove y1, y2 and add x1, x2
            if not improved and time.time() < _DEADLINE - 3:
                current_list = list(result_set)
                for i, y1 in enumerate(current_list):
                    if y1 == 1:
                        continue
                    for j, y2 in enumerate(current_list):
                        if j <= i or y2 == 1:
                            continue
                        rest = result_set - {y1, y2}
                        removed_sum = y1 + y2
                        # Find best pair (x1, x2) not in rest, not y1/y2
                        # such that x1,x2 coprime to rest and to each other
                        best_pair = None
                        best_pair_sum = removed_sum
                        for x1 in U:
                            if x1 in rest or x1 == y1 or x1 == y2:
                                continue
                            # Check x1 coprime to rest
                            ok1 = True
                            for z in rest:
                                if gcd(x1, z) != 1:
                                    ok1 = False
                                    break
                            if not ok1:
                                continue
                            for x2 in U:
                                if x2 in rest or x2 == y1 or x2 == y2 or x2 == x1:
                                    continue
                                if x1 + x2 <= best_pair_sum:
                                    continue
                                # Check x2 coprime to rest and x1
                                if gcd(x1, x2) != 1:
                                    continue
                                ok2 = True
                                for z in rest:
                                    if gcd(x2, z) != 1:
                                        ok2 = False
                                        break
                                if ok2:
                                    best_pair = (x1, x2)
                                    best_pair_sum = x1 + x2
                        if best_pair is not None:
                            result_set.discard(y1)
                            result_set.discard(y2)
                            result_set.add(best_pair[0])
                            result_set.add(best_pair[1])
                            improved = True

    # Convert to list of ints
    result_list = [int(x) for x in result_set]
    return result_list
