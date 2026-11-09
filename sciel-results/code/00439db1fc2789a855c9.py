import math
import time

def _squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if n > 1:
        result *= n
    return result

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    max_val = 10210
    target_kernel = 210
    
    # Precompute kernels for all numbers up to max_val
    # This is O(N * sqrt(N)) which is fine for N=10210
    kernels = [0] * (max_val + 1)
    for i in range(1, max_val + 1):
        kernels[i] = _squarefree_kernel(i)
        
    # We need to represent the kernels as bit vectors for GF(2) operations.
    # The primes involved are those <= max_val.
    # Let's map each prime to a bit index.
    # We can simply use the prime value itself as the bit position? 
    # No, max prime is ~10000, so we'd need 10000-bit integers. That's fine.
    # Or we can map to indices 0..P-1.
    # Using the prime value directly as bit position is simpler and Python ints are arbitrary precision.
    # Bit 0 is 1, Bit 1 is 2, Bit 2 is 3, etc? No, bit p corresponds to prime p.
    # So mask for kernel K is sum(1 << p for p in primes_of_K).
    
    # Precompute bit masks for each kernel
    # To do this efficiently, we can just convert the kernel value to a mask.
    # Since the kernel is square-free, its prime factors are distinct.
    # We need a function kernel_to_mask(k).
    
    # Let's precompute masks for all i in 211..max_val
    masks = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        k = kernels[i]
        mask = 0
        temp = k
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                mask |= (1 << d)
                while temp % d == 0:
                    temp //= d
            d += 1 if d == 2 else 2
        if temp > 1:
            mask |= (1 << temp)
        masks[i] = mask
        
    # Target mask for 210
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << p)
        
    # Linear Basis
    # basis[p] = (vector, history_mask)
    # vector has highest bit p
    # history_mask is a bitset of indices (0..max_val-211) corresponding to numbers used
    # To save memory and time, we can store history masks as Python ints.
    # Index i corresponds to number 211 + i.
    
    basis = {}
    
    # We iterate M from 211 to max_val
    # For each M, we add masks[M] to the basis
    # Then we check if target_mask is in the span
    
    # Optimization: We don't need to check every M if we can jump?
    # No, we must find the minimal M.
    
    # However, checking if target is in span takes O(P) steps.
    # P is number of primes ~ 1000.
    # Total steps ~ 10000 * 1000 = 10^7 operations. Should be fast in Python?
    # Bit operations on large ints are fast in C.
    
    # Let's iterate M
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        v = masks[M]
        h = 1 << (M - 211) # History bit for this number
        
        # Insert v into basis
        temp_v = v
        temp_h = h
        while temp_v:
            # Find highest set bit
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                basis[p] = (temp_v, temp_h)
                break
        
        # Check if target is in span
        # Reduce target using basis
        t_v = target_mask
        t_h = 0
        # We need to iterate bits of t_v from highest to lowest?
        # The standard reduction is: while t_v: p = highest bit; if p in basis: t_v ^= basis[p].vec; t_h ^= basis[p].hist
        # This works because basis[p] has highest bit p.
        while t_v:
            p = t_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                t_v ^= b_vec
                t_h ^= b_hist
            else:
                break
                
        if t_v == 0:
            # Solution found
            # t_h is the mask of indices
            # Extract indices
            sol_indices = []
            hh = t_h
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_indices.append(211 + idx)
                hh ^= lsb
                
            sol = sol_indices
            # Verify size
            if len(sol) <= 100:
                # Verify product is square
                prod = 210
                for x in sol:
                    prod *= x
                if _is_square(prod):
                    return [int(x) for x in sol]
            # If size > 100, we might need to continue?
            # But M is increasing. If the first valid M gives a large set, 
            # is it possible that a larger M gives a smaller set?
            # Yes. The basis changes.
            # So we should continue the loop if len(sol) > 100.
            # However, for small M, the dimension is small, so solution size is small.
            # It is unlikely to exceed 100 for small M.
            pass
            
    # Fallback
    # If no solution found within time, return a known valid one
    # From Inspiration 2: [224, 240] is valid. Max 240.
    return [224, 240]
