"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # Greedy merging for a strong initial/fallback solution
    slots = list(primes)
    while True:
        if time.time() > deadline - 10:
            break
        best_gain = 0
        best_i = -1
        best_j = -1
        n = len(slots)
        for i in range(n):
            for j in range(i + 1, n):
                merged = slots[i] * slots[j]
                if merged > MAX_N:
                    continue
                gain = merged - slots[i] - slots[j]
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        if best_i == -1 or best_gain <= 0:
            break
        slots[best_i] = slots[best_i] * slots[best_j]
        slots.pop(best_j)
    
    fallback_result = slots + [1]
    # Verify fallback
    valid = True
    for i in range(len(fallback_result)):
        for j in range(i + 1, len(fallback_result)):
            if gcd(fallback_result[i], fallback_result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    if not valid:
        fallback_result = [1] + primes
    
    best_sum = sum(fallback_result)
    best_set = list(fallback_result)
    
    try:
        # Attempt ILP solution for global optimality
        M = MAX_N
        c = np.arange(1, M + 1, dtype=np.float64)
        
        # Build conflict constraints: x_i + x_j <= 1 for gcd(i+1, j+1) > 1
        rows = []
        cols = []
        data = []
        
        for i in range(M):
            val_i = i + 1
            for j in range(i + 1, M):
                val_j = j + 1
                if gcd(val_i, val_j) > 1:
                    idx = len(rows)
                    rows.append(idx)
                    rows.append(idx)
                    cols.append(i)
                    cols.append(j)
                    data.append(1.0)
                    data.append(1.0)
        
        num_constraints = len(data) // 2
        if num_constraints > 0:
            A = lil_matrix((num_constraints, M))
            for k in range(num_constraints):
                A[k, cols[2*k]] = 1
                A[k, cols[2*k+1]] = 1
            A_csr = A.tocsr()
            lincons = LinearConstraint(A_csr, lb=-np.inf, ub=1)
        else:
            lincons = []
        
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(M)
        
        res = milp(c=c, constraints=lincons if lincons else None, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            ilp_set = []
            for i in range(M):
                if res.x[i] > 0.5:
                    ilp_set.append(i + 1)
            
            # Verify coprimality
            valid = True
            for i in range(len(ilp_set)):
                for j in range(i + 1, len(ilp_set)):
                    if gcd(ilp_set[i], ilp_set[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                ilp_sum = sum(ilp_set)
                if ilp_sum > best_sum:
                    best_sum = ilp_sum
                    best_set = ilp_set
                    
    except Exception:
        pass
    
    # Final safety check
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if gcd(best_set[i], best_set[j]) != 1:
                best_set = [1] + primes
                break
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
