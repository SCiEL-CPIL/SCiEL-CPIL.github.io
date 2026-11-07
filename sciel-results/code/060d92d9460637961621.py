"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 256
    
    # Identify all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # We will try to solve this as a MILP problem using scipy if available
    # Variables: x_i for i in 1..N, binary
    # Maximize: sum(i * x_i)
    # Constraints: For each prime p, sum(x_i for i in 1..N if p divides i) <= 1
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # c is the cost vector (we minimize, so we negate weights for maximization)
        # We want to maximize sum(i * x_i), so minimize sum(-i * x_i)
        c = np.array([-float(i) for i in range(1, N + 1)])
        
        # A_ub * x <= b_ub
        # For each prime p_j, sum of x_i where p_j | i <= 1
        rows = []
        cols = []
        data = []
        
        for idx, p in enumerate(primes):
            # Find all numbers i in 1..N divisible by p
            for i in range(1, N + 1):
                if i % p == 0:
                    rows.append(idx)
                    cols.append(i - 1)  # 0-indexed variable
                    data.append(1)
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(num_primes, N))
        b = np.ones(num_primes)
        
        # Bounds for variables: 0 <= x_i <= 1
        lb = np.zeros(N)
        ub = np.ones(N)
        
        # Integrality: all variables are integer (binary)
        integrality = np.ones(N)  # 1 means integer
        
        # Solve MILP
        # method='highs' is the default and robust
        res = milp(c=c,
                   constraints=LinearConstraint(A, ub=b),
                   integrality=integrality,
                   bounds=Bounds(lb, ub),
                   method='highs')
        
        if res.status == 0:  # 0 means success
            # res.x contains the optimal solution
            # Convert back to list of ints
            selected = [i + 1 for i in range(N) if res.x[i] > 0.5]
            
            # Ensure 1 is handled correctly (1 is coprime to everything, weight 1)
            # The ILP should pick 1 if it helps, but since 1 doesn't consume any prime,
            # it's always optimal to include 1 if not conflicting. 
            # Wait, 1 is coprime to all. It adds 1 to the sum. It consumes no primes.
            # So x_1 should always be 1 in the optimal solution.
            # Let's verify. If 1 is not in selected, add it.
            if 1 not in selected:
                selected.append(1)
            
            # Verify validity just in case (safety check)
            selected = list(set(selected))
            selected.sort()
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if math.gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return [int(x) for x in selected]
            else:
                # Fallback if ILP result is invalid (shouldn't happen)
                raise ValueError("ILP result invalid")
                
        else:
            raise RuntimeError(f"MILP solver failed: {res.message}")
            
    except Exception as e:
        # Fallback to a strong heuristic if MILP fails or is not available
        # Heuristic: Greedy with local search / swap
        # Start with prime powers
        result = [1]
        for p in primes:
            power = p
            while power * p <= N:
                power *= p
            result.append(power)
            
        # Local search: Try to swap pairs
        # This is a simplified version of the observed improvement strategy
        # But since we have time, we can do a few iterations
        
        def get_sum(lst):
            return sum(lst)
            
        current_sum = get_sum(result)
        
        # Precompute prime factors for all numbers 1..N
        prime_factors = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            n = i
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    prime_factors[i].append(p)
                    while n % p == 0:
                        n //= p
            if n > 1:
                prime_factors[i].append(n)
                
        def is_coprime_set(lst):
            # Check if all pairs in lst are coprime
            # Optimization: check prime factor overlap
            used_primes = set()
            for num in lst:
                for p in prime_factors[num]:
                    if p in used_primes:
                        return False
                    used_primes.add(p)
            return True
            
        # Simple local search: try replacing one element with another
        # Or replacing two with three?
        # Let's stick to a robust greedy + 1-swap for fallback
        improved = True
        while improved:
            improved = False
            current_set = set(result)
            # Try adding a number not in set
            for cand in range(2, N + 1):
                if cand in current_set:
                    continue
                # Check if cand is coprime to all in result
                ok = True
                for r in result:
                    if math.gcd(cand, r) != 1:
                        ok = False
                        break
                if ok:
                    new_sum = current_sum + cand
                    if new_sum > current_sum:
                        result.append(cand)
                        current_set.add(cand)
                        current_sum = new_sum
                        improved = True
            
            # Try swapping: remove one, add one
            for i in range(len(result)):
                r_i = result[i]
                for cand in range(2, N + 1):
                    if cand in current_set:
                        continue
                    # Check if cand is coprime to all in result except r_i
                    ok = True
                    for j in range(len(result)):
                        if j == i:
                            continue
                        if math.gcd(cand, result[j]) != 1:
                            ok = False
                            break
                    if ok:
                        new_sum = current_sum - r_i + cand
                        if new_sum > current_sum:
                            result[i] = cand
                            current_set.remove(r_i)
                            current_set.add(cand)
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
                    
        return [int(x) for x in result]

# EVOLVE-BLOCK-END
