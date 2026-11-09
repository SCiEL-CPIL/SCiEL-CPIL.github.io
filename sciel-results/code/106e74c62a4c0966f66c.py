import math
import time
from collections import deque

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin before 90s kill
    
    target = 210
    max_val = 10210

    # Sieve for SPF
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute square-free part (signature as frozenset of primes with odd exponent)
    def sqfree_part(x):
        temp = x
        sig = set()
        last_p = 0
        count = 0
        while temp > 1:
            p = spf[temp]
            if p == last_p:
                count += 1
            else:
                if count % 2 == 1:
                    sig.add(last_p)
                last_p = p
                count = 1
            temp //= p
        if count % 2 == 1:
            sig.add(last_p)
        return frozenset(sig)

    # Map primes to indices
    prime_index = {}
    primes = []
    for i in range(2, max_val + 1):
        if spf[i] == i:
            prime_index[i] = len(primes)
            primes.append(i)

    def sig_to_mask(sig_set):
        mask = 0
        for p in sig_set:
            mask |= (1 << prime_index[p])
        return mask

    target_mask = sig_to_mask(sqfree_part(target))

    # For each M from 211 upward, add M to the system and check if target is in span
    # Use Gaussian elimination maintaining basis
    basis = {}  # pivot_bit -> (vector_mask, index_mask)
    
    best_solution = None
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            if best_solution:
                return list(best_solution)
            break

        v = sig_to_mask(sqfree_part(M))
        idx_bit = 1 << (M - 211)
        
        temp_v = v
        temp_idx = idx_bit
        
        while temp_v != 0:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_mask, b_idx = basis[p]
                temp_v ^= b_mask
                temp_idx ^= b_idx
            else:
                basis[p] = (temp_v, temp_idx)
                break

        # Check if target is in span
        t_v = target_mask
        t_idx = 0
        
        while t_v != 0:
            p = t_v.bit_length() - 1
            if p in basis:
                b_mask, b_idx = basis[p]
                t_v ^= b_mask
                t_idx ^= b_idx
            else:
                t_v = -1
                break
        
        if t_v == 0:
            # Target is in span. Find minimal cardinality solution.
            sol = _find_minimal_solution(t_idx, M - 211, target, deadline, start)
            if sol and len(sol) <= 100:
                # Verify
                prod = target
                for x in sol:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return [int(x) for x in sol]
            # If we can't find small solution, try next M
            # But if M is small, the solution should be small too
            if M - 210 <= 50:
                # For small ranges, brute force search for small subset
                sol = _brute_force_small(M - 211, target, deadline, start)
                if sol and len(sol) <= 100:
                    prod = target
                    for x in sol:
                        prod *= x
                    if math.isqrt(prod) ** 2 == prod:
                        return [int(x) for x in sol]

    if best_solution:
        return list(best_solution)
    return [840]

def _find_minimal_solution(init_idx, n_elements, target, deadline, start):
    """Find a solution with few elements using BFS on the null space."""
    if time.time() > deadline:
        return None
    
    # init_idx is a bitmask of which elements are in the solution
    # We want to find a solution with minimum popcount
    # The null space vectors allow us to toggle elements
    
    # First, extract the initial solution
    init_sol = []
    for i in range(n_elements):
        if (init_idx >> i) & 1:
            init_sol.append(211 + i)
    
    if len(init_sol) <= 100:
        return init_sol
    
    # For now, just return the initial solution if it's valid size
    # A proper implementation would use null space to reduce size
    return None

def _brute_force_small(n_elements, target, deadline, start):
    """Brute force search for small subset when n_elements is small."""
    if time.time() > deadline:
        return None
    
    # Try subsets of size 1, 2, 3, ...
    # For n_elements <= 30, we can try small subsets
    elements = list(range(211, 211 + n_elements))
    
    def sqfree(x):
        temp = x
        sig = set()
        last_p = 0
        count = 0
        while temp > 1:
            p = temp
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                sig.add(p)
            count = 0
        if count % 2 == 1:
            sig.add(p)
        return frozenset(sig)
    
    target_sf = sqfree(target)
    
    # Try subsets of increasing size
    from itertools import combinations
    for size in range(1, min(6, n_elements + 1)):
        if time.time() > deadline:
            return None
        for combo in combinations(range(n_elements), size):
            prod = target
            for i in combo:
                prod *= elements[i]
            if math.isqrt(prod) ** 2 == prod:
                return [elements[i] for i in combo]
    return None

def _fallback_solution():
    return [840]
