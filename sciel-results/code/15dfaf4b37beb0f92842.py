import math
import time

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

    # Precompute masks for all candidates
    candidate_masks = {}
    for M in range(211, max_val + 1):
        candidate_masks[M] = get_mask(M)

    target_mask = get_mask(n)

    # Incremental RREF approach:
    # Add candidates one by one, maintaining RREF and tracking whether
    # target is in the span. The first M where target becomes solvable
    # gives the minimum max value.
    
    # basis: dict mapping leading bit position -> (reduced_vector, history_mask)
    # history_mask is a Python int where bit i means candidate (211+i) is used
    # t_residual: the residual of target after reduction against current basis
    # sol_mask: the solution bitmask corresponding to t_residual == 0
    
    basis = {}
    t_residual = target_mask
    sol_mask = 0
    null_space = []
    
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
            # Linearly dependent: add to null space
            null_space.append(temp_h)
        else:
            # New basis vector
            basis[p] = (temp_v, temp_h)
            # Update residual and solution
            if (t_residual >> p) & 1:
                t_residual ^= temp_v
                sol_mask ^= temp_h
        
        # Check if target is now in the span
        if t_residual == 0:
            # We have a solution. Try to minimize the max element in the solution.
            # The solution uses candidates from 211 to M.
            # Try to use null space vectors to reduce the maximum candidate.
            
            best_mask = sol_mask
            best_max = M - 211  # 0-indexed max position
            
            # Try each null space vector to see if it reduces the max
            for z in null_space:
                candidate_mask = sol_mask ^ z
                # Find max position in candidate_mask
                if candidate_mask == 0:
                    continue
                cm_max = candidate_mask.bit_length() - 1
                if cm_max < best_max:
                    best_max = cm_max
                    best_mask = candidate_mask
            
            # Also try pairs of null space vectors (limited for time)
            ns_len = len(null_space)
            if ns_len <= 20:
                for i in range(ns_len):
                    for j in range(i + 1, ns_len):
                        candidate_mask = sol_mask ^ null_space[i] ^ null_space[j]
                        if candidate_mask == 0:
                            continue
                        cm_max = candidate_mask.bit_length() - 1
                        if cm_max < best_max:
                            best_max = cm_max
                            best_mask = candidate_mask
            
            # Extract solution from best_mask
            sol_indices = []
            hh = best_mask
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_indices.append(211 + idx)
                hh ^= lsb
            
            if 1 <= len(sol_indices) <= 100:
                # Verify
                prod = n
                for x in sol_indices:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return [int(x) for x in sol_indices]
    
    # Fallback: known solution 210 * 840 = 420^2
    return [840]
