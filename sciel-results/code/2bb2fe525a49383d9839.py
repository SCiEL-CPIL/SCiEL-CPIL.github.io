import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    # 1. Sieve of Eratosthenes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute parity vectors (bitsets) for 210 and 211..10210
    def get_parity_bitset(n):
        bits = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << i)
        if temp > 1:
            # temp is a prime factor
            if temp in prime_index:
                bits |= (1 << prime_index[temp])
        return bits

    target_bits = get_parity_bitset(210)
    
    # Precompute bitsets for 211..10210
    bitsets = [0] * (limit - 210)
    for i in range(211, limit + 1):
        bitsets[i - 211] = get_parity_bitset(i)
        
    # 3. Iterate M from 1 upwards
    max_M = 1000
    for M in range(1, max_M + 1):
        if time.time() > deadline:
            break
            
        # Vectors are bitsets[0] ... bitsets[M-1]
        # We want to solve sum(x_i * v_i) = target_bits
        # Gaussian elimination on the set of vectors
        
        # We will perform Gaussian elimination to find a basis and check if target is in span.
        # We also want to find a particular solution and null space.
        
        # Let's use a standard approach:
        # Maintain a list of (pivot_col, vector, original_index)
        # Actually, we just need to solve the system.
        
        # We'll use an array of the M vectors and perform elimination on them.
        # But we need to track the coefficients.
        
        # Alternative: 
        # We have M equations? No, we have 1 equation in M variables? No.
        # We have a system of linear equations over GF(2).
        # The "variables" are the selection of numbers.
        # The "constraints" are the primes.
        
        # Let's use a dictionary or list to track the basis.
        # basis[pivot] = (vector, coeff_mask)
        # where vector is the bitset of the number, and coeff_mask is the bitset of which original numbers were XORed to get this basis vector.
        
        basis = {}
        # We iterate through the M numbers
        for i in range(M):
            v = bitsets[i]
            # Try to reduce v using existing basis
            changed = True
            while changed:
                changed = False
                # Find the highest set bit in v
                if v == 0:
                    break
                # Get the highest bit position
                hb = v.bit_length() - 1
                if hb in basis:
                    # XOR with basis vector
                    bv, bcoeff = basis[hb]
                    v ^= bv
                    # Update coefficient mask? 
                    # Wait, we are finding a particular solution.
                    # If we just want to check if target is in span, we can do it differently.
                    # Let's restart the logic.
                    
        # Let's use a cleaner Gaussian Elimination method for $Ax=b$.
        # We have M columns (vectors) and we want to find x.
        # We can perform elimination on the vectors themselves.
        
        # Let's store the vectors and their indices.
        # We will find a basis for the column space.
        # basis[bit] = (vector, index_mask)
        # index_mask is a bitmask of which of the M vectors were used.
        
        basis = {}
        # We process each vector i in 0..M-1
        for i in range(M):
            v = bitsets[i]
            mask = 1 << i
            
            # Reduce v
            while v != 0:
                hb = v.bit_length() - 1
                if hb in basis:
                    bv, bmask = basis[hb]
                    v ^= bv
                    mask ^= bmask
                else:
                    break
                    
            if v != 0:
                hb = v.bit_length() - 1
                basis[hb] = (v, mask)
                
        # Now check if target_bits is in the span
        # Reduce target_bits using the basis
        t = target_bits
        while t != 0:
            hb = t.bit_length() - 1
            if hb in basis:
                bv, bmask = basis[hb]
                t ^= bv
            else:
                break
                
        if t != 0:
            # No solution
            continue
            
        # Solution exists.
        # We have a particular solution?
        # The reduction of target_bits didn't track the mask.
        # Let's re-do to track the mask for the particular solution.
        
        basis = {}
        for i in range(M):
            v = bitsets[i]
            mask = 1 << i
            while v != 0:
                hb = v.bit_length() - 1
                if hb in basis:
                    bv, bmask = basis[hb]
                    v ^= bv
                    mask ^= bmask
                else:
                    break
            if v != 0:
                hb = v.bit_length() - 1
                basis[hb] = (v, mask)
                
        t = target_bits
        p_mask = 0
        while t != 0:
            hb = t.bit_length() - 1
            if hb in basis:
                bv, bmask = basis[hb]
                t ^= bv
                p_mask ^= bmask
            else:
                break
                
        if t != 0:
            continue
            
        # p_mask is a particular solution.
        # Now find the null space.
        # The null space is the set of masks m such that sum(m_i * v_i) = 0.
        # We can find a basis for the null space by looking at the free variables.
        # However, we can just brute force if the nullity is small.
        # Nullity = M - rank(basis)
        rank = len(basis)
        nullity = M - rank
        
        # If nullity is small, brute force
        if nullity <= 20:
            # We need to identify the free variables.
            # The pivot columns are the keys of basis.
            # But basis keys are bit positions in the vector space, not variable indices.
            # This is tricky.
            
            # Alternative: 
            # We can just try all subsets? No, M can be up to 100.
            # But we expect M to be small.
            
            # Let's find the null space basis properly.
            # We need to know which variables are free.
            # In the elimination process, the variables corresponding to the pivot rows are leading.
            # But our basis is keyed by the highest set bit of the *vector*, which corresponds to a prime.
            # This doesn't directly tell us which *numbers* (variables) are pivots.
            
            # Let's use a standard matrix form for small M.
            # Since M is small (we expect M < 50), we can represent the system as a list of bitsets of length M?
            # No, the vectors are long (primes).
            
            # Let's just check the weight of p_mask.
            weight = bin(p_mask).count('1')
            if weight <= 100:
                # We have a valid solution.
                # Extract the indices
                selected = []
                for i in range(M):
                    if p_mask & (1 << i):
                        selected.append(211 + i)
                # Verify
                # Product check is expensive, but we can check parity.
                # We already know parity works.
                return selected
            
            # If weight > 100, we might be able to find a better one.
            # But for now, let's just continue to next M if we can't find a good one quickly.
            # Or we can try to minimize weight.
            
        # If nullity is large, we can't brute force.
        # But if M is small, nullity is likely small.
        # If M is large, we might have already found a solution in a previous step.
        
    # If no solution found within max_M, return a dummy or best effort?
    # The problem guarantees a solution exists?
    # "Short intervals supplying a square-product extension"
    # It implies it exists.
    
    # Fallback: return [4*210] if it fits? 840 <= 10210.
    # 210 * 840 = 210 * 4 * 210 = 4 * 210^2 = (2*210)^2. Square.
    # Max is 840. 840 - 210 = 630.
    return [840]

# EVOLVE-BLOCK-END
