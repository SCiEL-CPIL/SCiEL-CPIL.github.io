"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    """
    Returns a list of distinct integers from 1..256 such that every pair is coprime,
    maximizing the sum.
    """
    N = 256
    
    # Step 1: Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
        
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Step 2: Setup ILP
    # Variables x_n for n in 1..256
    # Maximize sum(n * x_n)
    # Subject to: for each prime p, sum(x_n for n where p|n) <= 1
    # And x_n in {0,1}
    
    # Objective function coefficients (minimize -sum)
    c = np.array([-n for n in range(1, N + 1)], dtype=float)
    
    # Constraint matrix A: num_primes x N
    # Use sparse matrix for efficiency
    rows = []
    cols = []
    data = []
    for i, p in enumerate(primes):
        for multiple in range(p, N + 1, p):
            rows.append(i)
            cols.append(multiple - 1)
            data.append(1.0)
            
    A = csr_matrix((data, (rows, cols)), shape=(num_primes, N))
    
    # Linear constraints: A @ x <= 1
    lin_cons = LinearConstraint(A, ub=1)
    
    # Bounds: 0 <= x_n <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(N, dtype=int)
    
    # Solve
    selected = None
    try:
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
        if res.status == 0: # Optimal
            x = res.x
            # Extract selected integers
            selected = [i + 1 for i in range(N) if x[i] > 0.5]
        elif res.status == 1: # Infeasible (should not happen)
            pass
        else:
            # Try to recover from other statuses or fallback
            if hasattr(res, 'x') and res.x is not None:
                 # Sometimes status is not 0 but x is available
                 selected = [i + 1 for i in range(N) if res.x[i] > 0.5]
    except Exception:
        pass
        
    if selected is not None and len(selected) > 0:
        # Verify solution validity (safety check)
        valid = True
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if math.gcd(selected[i], selected[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return selected
            
    # Fallback: Greedy + Local Search
    prime_to_power = {}
    for p in primes:
        power = p
        next_power = p * p
        while next_power <= N:
            power = next_power
            next_power *= p
        prime_to_power[p] = power
        
    current_set = set([1])
    current_set.update(prime_to_power.values())
    
    def get_prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    # Greedy improvement: Try to swap elements for better ones
    # We iterate through all numbers in descending order and try to include them
    # if they don't conflict, removing conflicting elements if the gain is positive.
    
    all_numbers = list(range(N, 1, -1))
    
    improved = True
    while improved:
        improved = False
        for n in all_numbers:
            if n in current_set:
                continue
                
            # Check if n conflicts with current_set
            conflicts = []
            for x in current_set:
                if math.gcd(n, x) != 1:
                    conflicts.append(x)
                    
            if not conflicts:
                # No conflicts, just add it? 
                # Wait, if we add n, do we need to remove anything? No.
                # But we started with max prime powers. 
                # If n is coprime to all, adding it increases sum.
                # However, we might have missed it in the initial set.
                # Initial set has 1 and max prime powers.
                # Any number coprime to all max prime powers is a prime power of a prime not in the set?
                # No, primes are all in the set as their own power.
                # So any composite n will share a prime with some prime power p^k in the set.
                # So conflicts is never empty for composites if we start with all prime powers.
                pass
            
            # If conflicts exist, check if swapping is beneficial
            # Remove conflicts, add n.
            # Gain = n - sum(conflicts)
            gain = n - sum(conflicts)
            if gain > 0:
                # Check if the new set is valid (it will be, by construction of conflicts)
                # But we must ensure that removing conflicts doesn't break coprimality among themselves?
                # They were coprime to each other before. Removing them keeps the rest coprime.
                # n is coprime to all non-conflicts by definition.
                # So the new set is valid.
                current_set.difference_update(conflicts)
                current_set.add(n)
                improved = True
                # Restart loop? Or continue?
                # Since we added n, we might enable further swaps?
                # Actually, adding n might block other numbers, but we are maximizing sum.
                # Let's break and restart to be safe with the "improved" flag logic.
                break
                
    return list(current_set)

# EVOLVE-BLOCK-END
