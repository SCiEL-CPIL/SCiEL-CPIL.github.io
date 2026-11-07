# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # Step 1: Sieve for SPF
    limit = MAX_VAL
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Step 2: Kernels
    kernel = [0] * (limit + 1)
    for x in range(211, limit + 1):
        mask = 0
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        kernel[x] = mask
        
    # Target kernel for 210
    kernel_210 = 0
    temp = 210
    while temp > 1:
        p = spf[temp]
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count % 2 == 1:
            kernel_210 |= (1 << prime_to_idx[p])
    target_mask = kernel_210
    
    # Step 3: Iterate M
    # We will use a list to store vectors for current M
    # But we can also maintain the basis incrementally and just extract null space when needed?
    # Extracting null space from an incremental basis is tricky.
    # Better: For each M, if target is in span, we perform a full GE on the vectors 211..M to get null space.
    # Since M is small, this is fast.
    
    # To check if target is in span efficiently without full GE every time,
    # we can use the incremental basis check first.
    
    basis = [None] * num_primes
    combo = [0] * num_primes
    
    def add_to_basis(vec, idx_mask):
        v = vec
        c = idx_mask
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                basis[hb] = v
                combo[hb] = c
                return True
            else:
                v ^= basis[hb]
                c ^= combo[hb]
        return False
    
    def is_in_span(target):
        v = target
        c = 0
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                return False, 0
            v ^= basis[hb]
            c ^= combo[hb]
        return True, c
    
    for M in range(211, limit + 1):
        idx_mask = 1 << (M - 211)
        add_to_basis(kernel[M], idx_mask)
        
        found, sol_mask = is_in_span(target_mask)
        if found:
            # We have a solution. Now find the minimum weight solution.
            # Get all vectors for 211..M
            n = M - 210
            vectors = [kernel[211 + i] for i in range(n)]
            
            # Perform Gaussian Elimination to find particular solution and null space basis
            # Matrix A (num_primes x n) is too big to store explicitly?
            # We can store vectors as rows? No, we solve A x = b.
            # Let's augment the target vector.
            # We will do GE on the system of equations.
            # Actually, we can just use the vectors and target.
            
            # Let's create a list of (vector, index_bit)
            # We want to solve sum x_i v_i = target.
            
            # Standard GE on the vectors to find pivot columns.
            # We'll work with a copy of the vectors to determine pivot structure.
            # But we need the actual coefficients.
            
            # Alternative: Use the basis we already built?
            # The basis `basis` and `combo` give us one solution `sol_mask`.
            # To find the null space, we need the relationships between the original vectors.
            
            # Let's do a fresh GE for the small system of size n.
            # n is at most ~100. num_primes is ~1200.
            # We can represent the system as a list of bitmasks.
            
            # We want to find all x in GF(2)^n such that sum x_i v_i = 0.
            # This is the kernel of the linear map.
            
            # Let's build a reduced row echelon form of the matrix with rows = vectors?
            # No, the vectors are in GF(2)^P. We have n vectors.
            # We want the null space of the n x P matrix? No, the map is from GF(2)^n to GF(2)^P.
            # The null space is the set of x such that A x = 0.
            
            # We can find the null space by performing GE on the transpose?
            # Or just standard GE on the vectors to find linear dependencies.
            
            # Let's use a simple algorithm:
            # 1. Find a basis for the span of {v_1, ..., v_n}.
            # 2. Identify which vectors are dependent.
            # 3. The dependencies give the null space vectors.
            
            # Actually, there is a simpler way.
            # We have a particular solution x0 (from sol_mask).
            # We want to minimize weight(x0 + z) for z in NullSpace.
            # If n is small, we can just brute force all 2^n possibilities?
            # If n <= 20, 2^20 = 1M. Doable.
            # If n <= 30, 2^30 = 1B. Too slow.
            # But n is likely small when M is minimal.
            # Let's check n. If n <= 25, we can brute force the null space if we can find its dimension.
            # Or, we can just try to reduce the weight of x0 by swapping with other vectors?
            
            # Let's stick to the robust method:
            # Find the null space basis.
            
            # Algorithm to find null space basis of {v_1, ..., v_n}:
            # Perform GE on the matrix V (n rows, P cols).
            # Identify pivot rows.
            # The non-pivot rows (or columns?) correspond to free variables.
            # Actually, we are solving V^T x = 0? No.
            # We have vectors v_i. We want x such that sum x_i v_i = 0.
            # This is equivalent to finding the left null space of the matrix with rows v_i.
            
            # Let's just use the incremental basis idea but track the dependencies.
            # When we add v_i to the basis and it reduces to 0, the combination used is a null vector.
            # When it reduces to a new basis vector, it's independent.
            
            # So, during the incremental build for this M, we can collect all null vectors.
            # But we already built the basis globally.
            # Let's rebuild the basis for just 211..M and collect null vectors.
            
            local_basis = [None] * num_primes
            local_combo = [0] * num_primes
            null_vectors = []
            
            for i in range(n):
                v = vectors[i]
                c = 1 << i
                # Reduce v
                while v:
                    hb = v.bit_length() - 1
                    if local_basis[hb] is None:
                        local_basis[hb] = v
                        local_combo[hb] = c
                        break
                    else:
                        v ^= local_basis[hb]
                        c ^= local_combo[hb]
                if v == 0:
                    # c is a null vector
                    if c != 0:
                        null_vectors.append(c)
            
            # Now we have null_vectors.
            # We also have a particular solution x0 from the global basis?
            # No, we need a particular solution for this specific system.
            # We can get it by reducing target_mask using local_basis.
            
            v_target = target_mask
            x0 = 0
            while v_target:
                hb = v_target.bit_length() - 1
                if local_basis[hb] is None:
                    # Should not happen if is_in_span said True
                    break
                v_target ^= local_basis[hb]
                x0 ^= local_combo[hb]
            
            if v_target != 0:
                continue # Inconsistent, should not happen
            
            # Now minimize weight(x0 + sum c_i n_i)
            # If len(null_vectors) is small, we can iterate.
            # Max null vectors is n - rank.
            # Let's try to reduce x0 using null vectors greedily?
            # Or if k = len(null_vectors) <= 20, iterate all 2^k.
            
            k = len(null_vectors)
            if k <= 20:
                best_x = x0
                best_weight = bin(x0).count('1')
                # Iterate all combinations of null vectors
                # We can do this by iterating mask from 0 to 2^k - 1
                for mask in range(1, 1 << k):
                    z = 0
                    mm = mask
                    j = 0
                    while mm:
                        if mm & 1:
                            z ^= null_vectors[j]
                        mm >>= 1
                        j += 1
                    x = x0 ^ z
                    w = bin(x).count('1')
                    if w < best_weight:
                        best_weight = w
                        best_x = x
                x0 = best_x
            
            # Check if best_weight <= 100
            if best_weight <= MAX_SIZE:
                selected = []
                m = x0
                offset = 211
                i = 0
                while m:
                    if m & 1:
                        selected.append(offset + i)
                    m >>= 1
                    i += 1
                
                # Verify
                prod = 1
                for x in selected:
                    prod *= x
                total = prod * 210
                root = math.isqrt(total)
                if root * root == total:
                    return selected
            
    return [840]

# EVOLVE-BLOCK-END
