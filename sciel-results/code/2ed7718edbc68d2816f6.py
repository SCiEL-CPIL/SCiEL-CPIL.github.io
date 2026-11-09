import math
import time
from collections import defaultdict

def _sqfree_part(n):
    ps = 1
    i = 2
    while i * i <= n:
        while n % (i * i) == 0:
            n //= (i * i)
            ps *= i
        i += 1
    return n, ps

def construct():
    start = time.time()
    deadline = start + 85.0
    target = 210
    max_val = 10210

    spf = list(range(max_val + 1))
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    prime_index = {}
    primes = []
    for i in range(2, max_val + 1):
        if spf[i] == i:
            prime_index[i] = len(primes)
            primes.append(i)
            
    def get_mask(n):
        mask = 0
        temp = n
        last_p = 0
        count = 0
        while temp > 1:
            p = spf[temp]
            if p == last_p:
                count += 1
            else:
                if count % 2 == 1:
                    mask |= (1 << prime_index[last_p])
                last_p = p
                count = 1
            temp //= p
        if count % 2 == 1:
            mask |= (1 << prime_index[last_p])
        return mask

    signatures = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        signatures[x] = get_mask(x)

    target_mask = signatures[target]

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            return _fallback_solution()
            
        sol = _solve_min_weight(211, M, target_mask, signatures)
        if sol is not None:
            prod = target
            for x in sol:
                prod *= x
            if math.isqrt(prod) ** 2 == prod and len(sol) <= 100:
                return sol

    return _fallback_solution()

def _solve_min_weight(lo, hi, target_mask, signatures):
    n = hi - lo + 1
    if n == 0:
        return None
        
    vectors = [signatures[lo + i] for i in range(n)]
    
    # Gaussian Elimination to find rank, basis, and null space
    # We represent the matrix as list of vectors (rows).
    # We want to solve V^T x = t ? No, sum(x_i v_i) = t.
    # So v_i are columns of the matrix A.
    # We perform GE on the rows of A?
    # Standard GE on the set of vectors to find a basis for the span.
    
    # Basis: pivot_bit -> (vector, index_in_basis)
    # We also need to track the coefficients to reconstruct solutions.
    
    # Let's use a different approach for Null Space:
    # 1. Find a basis for the span of {v_1...v_n}.
    # 2. Check if t is in span.
    # 3. Find particular solution x_p.
    # 4. Find null space basis.
    
    # This is tricky with bitmasks.
    # Let's stick to the "Enumerate Free Variables" approach.
    
    # Perform GE on the vectors.
    # basis[p] = (reduced_vector, original_index, coeff_dict)
    # coeff_dict maps original_indices to coeffs (0/1)
    
    # Actually, simpler:
    # 1. Reduce all vectors to row echelon form, keeping track of which original vectors were used.
    
    basis = {} # pivot -> (mask, frozenset of original indices)
    free_vars = []
    
    # We process vectors one by one.
    # If a vector reduces to 0, it is a linear combination of previous ones.
    # The relation gives a null space vector.
    
    for i in range(n):
        v = vectors[i]
        temp_v = v
        temp_indices = frozenset([i])
        
        while temp_v != 0:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_mask, b_indices = basis[p]
                temp_v ^= b_mask
                temp_indices = frozenset(temp_indices ^ b_indices)
            else:
                basis[p] = (temp_v, temp_indices)
                break
                
        if temp_v == 0:
            # v_i is in span of previous basis vectors.
            # The combination that yields 0 is the current temp_indices?
            # No, temp_indices represents the combination of original vectors that equals the current temp_v.
            # Wait.
            # We started with v_i (index i).
            # We XORed with basis vectors.
            # Basis vectors are combinations of original vectors.
            # So temp_indices is the set of original vectors that sum to temp_v.
            # If temp_v becomes 0, then the sum of vectors in temp_indices is 0.
            # So temp_indices is a null space vector!
            if temp_indices:
                free_vars.append(frozenset(temp_indices))
                
    # Now check if target is in span
    t_v = target_mask
    t_indices = frozenset()
    
    while t_v != 0:
        p = t_v.bit_length() - 1
        if p in basis:
            b_mask, b_indices = basis[p]
            t_v ^= b_mask
            t_indices = frozenset(t_indices ^ b_indices)
        else:
            return None # Not in span
            
    if t_v != 0:
        return None
        
    x_p = t_indices
    
    # Now we have x_p and a list of null space vectors (free_vars).
    # Note: free_vars might not be a basis, but a set of generators.
    # We need a basis for the null space.
    # The number of free_vars is n - rank.
    # Are they linearly independent?
    # In standard GE, each time we find a dependency, we add it.
    # These dependencies form a basis for the null space.
    
    D = len(free_vars)
    
    if D == 0:
        if len(x_p) <= 100:
            return [lo + i for i in x_p]
        return None
        
    if D <= 20:
        # Enumerate all 2^D combinations
        best_weight = 101
        best_sol = None
        
        for mask in range(1 << D):
            current_sol = set(x_p)
            for j in range(D):
                if mask & (1 << j):
                    # XOR current_sol with free_vars[j]
                    fv = free_vars[j]
                    # Efficient XOR of sets
                    # current_sol = current_sol ^ fv
                    # In Python, set XOR is not built-in like ^
                    # We can do: current_sol = (current_sol | fv) - (current_sol & fv)
                    # Or just use symmetric difference
                    current_sol = current_sol.symmetric_difference(fv)
                    
            w = len(current_sol)
            if w < best_weight:
                best_weight = w
                best_sol = current_sol
                
        if best_sol is not None and best_weight <= 100:
            return [lo + i for i in best_sol]
            
    else:
        # D > 20, can't enumerate.
        # Just check if x_p is good enough.
        if len(x_p) <= 100:
            return [lo + i for i in x_p]
            
    return None

def _fallback_solution():
    return [840]
