"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = factorial(12)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    prime_powers = {
        2: 10,
        3: 5,
        5: 2,
        7: 1,
        11: 1
    }
    primes = list(prime_powers.keys())
    prime_idx = {p: i for i, p in enumerate(primes)}
    
    def get_prime_exponents(n):
        """Return dict of prime -> exponent for n."""
        exponents = {}
        for p in primes:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count > 0:
                exponents[p] = count
        return exponents
    
    # Generate all divisors of N greater than 12
    divisors = []
    limit = int(N ** 0.5)
    for i in range(1, limit + 1):
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            other = N // i
            if other > 12 and other != i:
                divisors.append(other)
    
    divisors = sorted(set(divisors))
    n_divisors = len(divisors)
    
    # Precompute prime exponents for each divisor
    divisor_exponents = []
    for d in divisors:
        divisor_exponents.append(get_prime_exponents(d))
    
    # Build ILP using scipy.optimize.milp
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    
    # Variables: x_0, ..., x_{n-1} (binary, select divisor), t (minimize)
    # Total variables: n_divisors + 1
    
    # Objective: minimize t
    c = np.zeros(n_divisors + 1)
    c[-1] = 1.0  # minimize t
    
    # Constraints:
    # 1. For each prime p: sum(x_i * exp_p(div_i)) == exp_p(N)
    # 2. t >= div_i * x_i for all i (epigraph)
    # 3. x_i binary
    
    n_primes = len(primes)
    
    # Constraint matrix for prime exponents (equality)
    A_eq = np.zeros((n_primes, n_divisors + 1))
    b_eq = np.zeros(n_primes)
    for j, d in enumerate(divisors):
        for p_idx, p in enumerate(primes):
            exp = divisor_exponents[j].get(p, 0)
            A_eq[p_idx, j] = exp
        b_eq[p_idx] = prime_powers[p]
    
    # Constraint matrix for t >= div_i * x_i => t - div_i * x_i >= 0
    # This is: -div_i * x_i + 1*t >= 0
    A_ub = np.zeros((n_divisors, n_divisors + 1))
    b_ub = np.zeros(n_divisors)
    for j in range(n_divisors):
        A_ub[j, j] = -float(divisors[j])
        A_ub[j, -1] = 1.0
        b_ub[j] = 0.0
    
    # Bounds: x_i in [0, 1], t in [0, inf)
    lb = np.zeros(n_divisors + 1)
    ub = np.ones(n_divisors + 1)
    ub[-1] = np.inf
    bounds = Bounds(lb, ub)
    
    # Integrality: x_i binary, t continuous
    integrality = np.zeros(n_divisors + 1)
    integrality[:n_divisors] = 1  # binary
    integrality[-1] = 0  # continuous
    
    # Solve ILP
    try:
        result = milp(
            c=c,
            constraints=[
                LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
                LinearConstraint(A_ub, lb=b_ub, ub=np.inf)
            ],
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 40}
        )
        
        if result.status == 0:
            selected = []
            for j in range(n_divisors):
                if result.x[j] > 0.5:
                    selected.append(divisors[j])
            if selected:
                return selected
    except Exception:
        pass
    
    # Fallback: use backtracking search with precomputed divisors
    best_max = float('inf')
    best_factors = []
    
    def search(remaining, start_idx, current_factors):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if current_factors:
                current_max = max(current_factors)
                if current_max < best_max:
                    best_max = current_max
                    best_factors = current_factors[:]
            return
        
        if len(current_factors) >= 12:
            return
            
        for i in range(start_idx, n_divisors):
            f = divisors[i]
            
            if f >= best_max:
                break
                
            if f > remaining:
                break
                
            if remaining % f == 0:
                current_factors.append(f)
                search(remaining // f, i + 1, current_factors)
                current_factors.pop()
                
    search(N, 0, [])
    
    if not best_factors:
        return [N]
        
    return best_factors

# EVOLVE-BLOCK-END
