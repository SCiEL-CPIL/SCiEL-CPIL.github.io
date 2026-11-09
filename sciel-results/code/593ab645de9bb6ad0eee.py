"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """Return a list of distinct ints from 1..256, pairwise coprime, maximizing sum."""
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    N = 256
    
    # Precompute prime factorization for 1..256
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        prime_factors[v] = get_prime_factors(v)
    
    # Build conflict graph: conflicts[v] = set of u such that gcd(u, v) > 1
    conflicts = [set() for _ in range(N + 1)]
    for v in range(1, N + 1):
        for u in range(1, N + 1):
            if u != v and prime_factors[v] & prime_factors[u]:
                conflicts[v].add(u)
    
    # Build initial solution using prime powers + known good replacements
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    result = [1] + list(prime_powers.values())
    
    # Apply known beneficial replacements as initial seed
    if 121 in result and 23 in result:
        result.remove(121); result.remove(23); result.append(253)
    if 49 in result and 31 in result:
        result.remove(49); result.remove(31); result.append(217)
    if 125 in result and 47 in result:
        result.remove(125); result.remove(47); result.append(235)
    if 81 in result and 83 in result:
        result.remove(81); result.remove(83); result.append(249)
    if 169 in result and 19 in result:
        result.remove(169); result.remove(19); result.append(247)
    
    best_set = set(result)
    best_sum = sum(result)
    
    # Try ILP with scipy if available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        import numpy as np
        
        # Collect all conflicting pairs
        constraint_rows = []
        constraint_cols = []
        constraint_vals = []
        
        for v in range(1, N + 1):
            for u in conflicts[v]:
                if u < v:  # Avoid duplicates
                    constraint_rows.append(len(constraint_rows))
                    constraint_cols.extend([u - 1, v - 1])
                    constraint_vals.extend([1, 1])
        
        if constraint_rows:
            A = csr_matrix((constraint_vals, (constraint_rows, constraint_cols)), 
                          shape=(len(constraint_rows), N))
            c = np.array([-i for i in range(1, N + 1)], dtype=float)  # Minimize negative sum
            constraints = LinearConstraint(A, ub=1)
            bounds = Bounds(0, 1)
            integrality = np.ones(N)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.success and res.x is not None:
                ilp_set = set()
                for i, val in enumerate(res.x):
                    if val > 0.5:
                        ilp_set.add(i + 1)
                
                # Verify pairwise coprime
                valid = True
                for i, a in enumerate(ilp_set):
                    for b in list(ilp_set)[i+1:]:
                        if math.gcd(a, b) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(ilp_set) > 0:
                    ilp_sum = sum(ilp_set)
                    if ilp_sum > best_sum:
                        best_set = ilp_set
                        best_sum = ilp_sum
        
        if time.time() > deadline:
            return list(best_set)
        
    except Exception:
        pass
    
    # Local search refinement on the best solution found
    def is_coprime_with_set(x, s):
        fx = prime_factors[x]
        for y in s:
            if fx & prime_factors[y]:
                return False
        return True
    
    current_set = set(best_set)
    current_sum = best_sum
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a single element
        best_add = None
        best_add_gain = 0
        for x in range(N, 0, -1):
            if x not in current_set and is_coprime_with_set(x, current_set):
                gain = x
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add = x
        if best_add is not None:
            current_set.add(best_add)
            current_sum += best_add
            improved = True
            continue
        
        # Try replacing two elements with one larger element
        best_swap = None
        best_swap_gain = 0
        result_list = list(current_set)
        n = len(result_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = result_list[i], result_list[j]
                removed_sum = a + b
                remaining = set(current_set) - {a, b}
                for x in range(N, 0, -1):
                    if x in current_set:
                        continue
                    if is_coprime_with_set(x, remaining):
                        gain = x - removed_sum
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (a, b, x)
                        break  # largest x for this pair
        
        if best_swap is not None:
            a, b, x = best_swap
            current_set.remove(a)
            current_set.remove(b)
            current_set.add(x)
            current_sum = sum(current_set)
            improved = True
    
    if current_sum > best_sum:
        best_set = current_set
        best_sum = current_sum
    
    # Final verification
    result_list = list(best_set)
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            if math.gcd(result_list[i], result_list[j]) != 1:
                # Fallback: return just prime powers for safety
                return [1] + list(prime_powers.values())
    
    return result_list
# EVOLVE-BLOCK-END
