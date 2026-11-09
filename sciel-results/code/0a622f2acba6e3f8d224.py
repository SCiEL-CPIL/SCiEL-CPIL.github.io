import math
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False

    primes = [i for i, val in enumerate(is_prime) if val]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_mask(x):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << prime_index[p])
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks
    candidate_masks = {}
    for M in range(211, max_val + 1):
        candidate_masks[M] = get_mask(M)

    target_mask = get_mask(n)

    # Strategy: use Gaussian elimination on the linear system over GF(2)
    # but search for minimum weight solution using meet-in-the-middle or
    # iterative refinement with null space basis.

    # First, build the linear system and find a particular solution + null space
    # We work with vectors of length (max_val - 210) for candidates 211..10210
    # Each candidate M has a mask (square-free part parity)
    # We want sum of masks of selected candidates == target_mask (mod 2)
    # This is equivalent to: sum of masks + target_mask == 0

    # Build augmented matrix: [mask | unit vector for each candidate]
    # But with ~10000 candidates, this is too large for direct Gauss-Jordan on all.

    # Alternative: Use the fact that we want to minimize max(selected) - 210
    # So we want to find a subset with small maximum element.
    
    # Let's try: for each possible max value M_max from 211 to max_val,
    # check if there exists a subset of {211, ..., M_max} whose mask sum == target_mask.
    # This is a linear algebra question: is target_mask in the span of {masks[211..M_max]}?
    # If yes, find such a subset.

    # We'll maintain a row-reduced echelon form incrementally.
    
    # basis: dict mapping leading bit -> (vector, set of candidate indices used)
    # But storing sets is expensive. Instead, store as bitmask of candidates (Python int).
    
    basis = {}
    # null_space: list of candidate bitmasks that span the null space
    null_space = []
    
    # t_vec will be the current target after reduction
    # sol_mask will be the current solution bitmask
    
    # Process candidates in order 211, 212, ...
    # At each step, we check if target is in span
    
    # To be efficient, we'll track:
    # - basis: for each leading bit position, the reduced vector and corresponding candidate mask
    # - We maintain the reduced form of target_mask
    
    # Actually, let's use a different approach:
    # Maintain RREF of the candidate masks as we add them one by one.
    # After adding candidate k, check if target_mask is in the column space.
    
    # For efficiency, I'll use a simpler approach:
    # 1. Find any solution using Gauss-Jordan
    # 2. Find null space basis
    # 3. Use local search to minimize the maximum element
    
    # Let me implement Gauss-Jordan on the full system first, but only for a subset
    # that we expect to be small.
    
    # Actually, the key insight: we want to minimize max(selected) - 210.
    # So the optimal solution likely uses small numbers.
    
    # Let's try: for M_max in range(211, min(max_val, 500) + 1):
    #   Build RREF of masks[211..M_max]
    #   Check if target is in span
    #   If yes, find a solution and verify
    
    # This could work if the answer is small.
    
    # But building RREF from scratch each time is O(M_max * num_primes) which is fine.
    
    def check_solvable(max_cand):
        """Check if target_mask is in span of masks[211..max_cand].
        If yes, return a solution bitmask, else None."""
        if time.time() > deadline:
            return None
        
        # Build RREF
        local_basis = {}
        t = target_mask
        
        for M in range(211, max_cand + 1):
            if time.time() > deadline:
                return None
            v = candidate_masks[M]
            h = 1 << (M - 211)
            
            temp_v = v
            temp_h = h
            while temp_v:
                p = temp_v.bit_length() - 1
                if p in local_basis:
                    b_vec, b_hist = local_basis[p]
                    temp_v ^= b_vec
                    temp_h ^= b_hist
                else:
                    break
            
            if temp_v == 0:
                # This is a null space vector; ignore for now
                pass
            else:
                local_basis[p] = (temp_v, temp_h)
                if (t >> p) & 1:
                    t ^= temp_v
        
        # If t == 0, target is in span. Now find the actual solution.
        # We need to redo to get the solution.
        if t != 0:
            return None
        
        # Find solution by back-substitution
        # Actually, during the process, we can track the solution.
        # Let me redo this more carefully.
        
        sol = 0
        t2 = target_mask
        for M in range(211, max_cand + 1):
            if time.time() > deadline:
                return None
            v = candidate_masks[M]
            h = 1 << (M - 211)
            
            temp_v = v
            temp_h = h
            while temp_v:
                p = temp_v.bit_length() - 1
                if p in local_basis:
                    b_vec, b_hist = local_basis[p]
                    temp_v ^= b_vec
                    temp_h ^= b_hist
                else:
                    break
            
            if temp_v == 0:
                continue
            # temp_v has leading bit p, and is not in basis yet (we're rebuilding)
            # Wait, this won't work because local_basis is already built.
            pass
        
        # Let me use a different approach: just solve using the basis we built.
        # The basis vectors are in RREF. To find a solution:
        # Start with sol = 0, residual = target_mask
        # For each basis vector in order of decreasing leading bit:
        #   If residual has that bit, XOR in the basis vector and its history.
        
        sol = 0
        residual = target_mask
        # Sort basis by leading bit descending
        for p in sorted(local_basis.keys(), reverse=True):
            b_vec, b_hist = local_basis[p]
            if (residual >> p) & 1:
                residual ^= b_vec
                sol ^= b_hist
        
        if residual != 0:
            return None
        
        return sol
    
    # Try increasing max_cand
    for max_cand in range(211, min(max_val + 1, 400)):
        if time.time() > deadline:
            break
        
        sol_mask = check_solvable(max_cand)
        
        if sol_mask is not None:
            # Extract the solution
            sol_indices = []
            hh = sol_mask
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_indices.append(211 + idx)
                hh ^= lsb
            
            # Verify
            prod = n
            for x in sol_indices:
                prod *= x
            
            if math.isqrt(prod) ** 2 == prod and 1 <= len(sol_indices) <= 100:
                return [int(x) for x in sol_indices]
    
    # If we haven't found a solution with max <= 400, try a fallback approach
    # Find any solution using the full range, then try to minimize
    
    # Full Gauss-Jordan
    basis = {}
    null_space = []
    sol_mask = 0
    t_vec = target_mask
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        v = candidate_masks[M]
        h = 1 << (M - 211)
        
        temp_v = v
        temp_h = h
        while temp_v:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                break
        
        if temp_v == 0:
            null_space.append(temp_h)
        else:
            basis[p] = (temp_v, temp_h)
            if (t_vec >> p) & 1:
                t_vec ^= temp_v
                sol_mask ^= temp_h
        
        if t_vec == 0 and sol_mask:
            weight = bin(sol_mask).count('1')
            if 1 <= weight <= 100:
                sol_indices = []
                hh = sol_mask
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_indices.append(211 + idx)
                    hh ^= lsb
                
                prod = n
                for x in sol_indices:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return [int(x) for x in sol_indices]
            
            # Try to reduce weight using null space
            improved = True
            while improved:
                improved = False
                for z in null_space:
                    new_mask = sol_mask ^ z
                    if bin(new_mask).count('1') < bin(sol_mask).count('1'):
                        sol_mask = new_mask
                        improved = True
            
            weight = bin(sol_mask).count('1')
            if 1 <= weight <= 100:
                sol_indices = []
                hh = sol_mask
                while hh:
                    lsb = hh & -hh
                    idx = lsb.bit_length() - 1
                    sol_indices.append(211 + idx)
                    hh ^= lsb
                
                prod = n
                for x in sol_indices:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return [int(x) for x in sol_indices]
    
    # Fallback: simple known solution
    # 210 = 2 * 3 * 5 * 7
    # 4*210 = 840, and 840 * 840 / 210 = 840, not helpful
    # 210 * 210 * 4 = 210 * 840, not a square
    # 210 * 420 = 210 * 2 * 210 = 2 * 210^2, not a square
    # 210 * 840 = 210 * 4 * 210 = 4 * 210^2 = (2*210)^2 = 420^2. Yes!
    # So {840} works: 210 * 840 = 420^2
    if time.time() < deadline:
        return [840]
    
    return [840]
