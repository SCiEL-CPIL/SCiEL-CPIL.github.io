import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    # Generate primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Objective: Maximize sum of selected integers
    # Variables x_0 ... x_255 correspond to integers 1 ... 256
    c = np.arange(1, n + 1, dtype=float)
    
    # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
    # A_ub has shape (num_primes, n)
    A_ub = np.zeros((num_primes, n), dtype=float)
    b_ub = np.ones(num_primes, dtype=float)
    
    for j, p in enumerate(primes):
        for i in range(n):
            if (i + 1) % p == 0:
                A_ub[j, i] = 1.0
                
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    
    # Integrality: all variables are integers
    integrality = np.ones(n)
    
    # Solve
    res = milp(c=-c, # Minimize -sum to maximize sum
               constraints=LinearConstraint(A_ub, ub=b_ub),
               integrality=integrality,
               bounds=bounds)
    
    if not res.success:
        # Fallback: Largest power of prime strategy
        # This is a valid solution, just maybe not optimal
        result = [1]
        for p in primes:
            val = p
            while val * p <= n:
                val *= p
            result.append(val)
        return result
        
    x = res.x
    selected = []
    for i in range(n):
        if x[i] > 0.5:
            selected.append(int(i + 1))
            
    return selected
