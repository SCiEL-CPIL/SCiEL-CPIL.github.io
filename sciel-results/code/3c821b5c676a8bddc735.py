import time
from math import isqrt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 50.0 # Leave 10s margin for 60s limit
    
    N = 256
    # 1. Generate primes <= N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    p_index = {p: i for i, p in enumerate(primes)}
    
    # 2. Determine prime factors for each n
    # factors[n] is a list of indices of primes that divide n
    factors = [[] for _ in range(N + 1)]
    for n in range(2, N + 1):
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors[n].append(p_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors[n].append(p_index[temp])
            
    # 3. Formulate ILP
    # Variables x_1 ... x_256
    # Objective: Maximize sum(n * x_n)
    c = np.arange(1, N + 1, dtype=float)
    
    # Constraints: For each prime p, sum(x_n for n where p|n) <= 1
    # A is a matrix of shape (num_primes, N)
    A = np.zeros((len(primes), N), dtype=float)
    for n in range(1, N + 1):
        for p_idx in factors[n]:
            A[p_idx, n-1] = 1.0
            
    # Bounds for constraints: 0 <= Ax <= 1
    lb = np.zeros(len(primes))
    ub = np.ones(len(primes))
    constraint = LinearConstraint(A, lb, ub)
    
    # Variable bounds: 0 <= x_i <= 1
    var_bounds = Bounds(0, 1)
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(N)
    
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=var_bounds)
        
        if res.status == 0: # Success
            x = res.x
            # x is a numpy array of floats, should be 0 or 1
            result = []
            for i in range(N):
                if x[i] > 0.5:
                    result.append(int(i + 1))
            
            # Ensure 1 is included? 
            # The ILP will include 1 if it helps. Since 1 has no constraints and adds 1, it will be included.
            # But let's double check. If for some reason it's not, add it.
            if 1 not in result:
                result.append(1)
            
            # Verify validity
            # Check distinct
            if len(result) != len(set(result)):
                # Fallback to primes if something went wrong
                return [1] + primes
            
            # Check coprime
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    if np.gcd(result[i], result[j]) != 1:
                        # Fallback
                        return [1] + primes
            
            return result
        else:
            # Fallback to greedy/primes if ILP fails
            pass
    except Exception:
        pass
    
    # Fallback: Greedy descending with local search or just primes
    # Simple fallback: Primes + 1
    return [1] + primes
