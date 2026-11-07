import math
import time

def construct():
    deadline = time.time() + 85
    
    max_val = 10210
    
    # 1. Sieve primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Compute squarefree kernel bitmask for each number
    # We need K(x) for x in [211, max_val]
    # K(x) is product of primes with odd exponent.
    # Bitmask: bit i is 1 if prime p_i has odd exponent in x.
    
    def get_kernel_bitmask(n):
        mask = 0
        f = n
        for p in primes:
            if p * p > f:
                break
            if f % p == 0:
                exp = 0
                while f % p == 0:
                    f //= p
                    exp += 1
                if exp % 2 == 1:
                    mask |= (1 << prime_to_idx[p])
        if f > 1:
            # f is a prime > sqrt(n)
            if f in prime_to_idx:
                mask |= (1 << prime_to_idx[f])
            else:
                # This shouldn't happen if primes list is complete up to max_val
                # But f could be > max_val? No, f <= n <= max_val.
                pass
        return mask

    # Precompute masks for all numbers in range
    masks = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        masks[i] = get_kernel_bitmask(i)
        
    target_mask = get_kernel_bitmask(210)
    
    # 3. Incremental Linear Basis
    # basis_vec[i] is the bitmask of the basis vector with pivot i
    # basis_src[i] is the set of original numbers (indices) that form basis_vec[i]
    # We use a list of tuples or separate lists.
    # Since we don't know the max pivot in advance, we can use a dict or list.
    # Max pivot is len(primes) ~ 1000.
    
    basis_vec = [0] * len(primes)
    basis_src = [set() for _ in range(len(primes))]
    
    def add_to_basis(vec, src):
        """
        Adds vec (with source src) to the basis.
        Returns True if vec was independent (added to basis), False if dependent.
        """
        v = vec
        s = set(src) # Copy source set
        
        # Reduce v by existing basis vectors
        # We iterate from highest pivot to lowest? Or lowest to highest?
        # Standard Gaussian elimination:
        # For each bit from high to low:
        #   if v has bit i:
        #     if basis_vec[i] exists:
        #       v ^= basis_vec[i]
        #       s = symmetric_diff(s, basis_src[i])
        #     else:
        #       # v is a new basis vector with pivot i
        #       basis_vec[i] = v
        #       basis_src[i] = s
        #       return True
        # if v becomes 0, it's dependent
        # return False
        
        # We need to know the highest set bit of v to be efficient.
        # Python int.bit_length() gives this.
        
        while v:
            # Find highest set bit
            i = v.bit_length() - 1
            if basis_vec[i]:
                v ^= basis_vec[i]
                # Symmetric difference
                s = s.symmetric_difference(basis_src[i])
            else:
                # Add to basis
                basis_vec[i] = v
                basis_src[i] = s
                return True
        return False

    def is_in_span(target):
        """
        Checks if target is in the span.
        Returns the set of original numbers if found, else None.
        """
        v = target
        s = set()
        
        # Reduce target by basis vectors
        # We must process in a specific order?
        # Actually, any order works for checking membership, 
        # but for extraction, we need to ensure we use the basis vectors correctly.
        # Standard reduction:
        # Iterate i from high to low.
        # if v has bit i and basis_vec[i] exists:
        #   v ^= basis_vec[i]
        #   s = symmetric_diff(s, basis_src[i])
        
        # However, we need to be careful. The basis is not necessarily in reduced row echelon form (RREF)
        # where each pivot is the only one in its column.
        # In the insertion method above, when we add a new vector, we reduce it by *existing* vectors.
        # So the new basis vector has 0s in all positions corresponding to *existing* pivots.
        # This means the basis is in a form where pivot i is the highest set bit of basis_vec[i],
        # and basis_vec[i] has 0s in all positions j > i? No.
        # Let's trace:
        # Suppose we have basis_vec[2] = 0b100 (pivot 2).
        # New vec = 0b110.
        # Highest bit 2. basis_vec[2] exists.
        # v ^= basis_vec[2] => 0b010.
        # Highest bit 1. basis_vec[1] doesn't exist.
        # Add basis_vec[1] = 0b010.
        # Now basis_vec[2] = 0b100, basis_vec[1] = 0b010.
        # This is RREF-like.
        
        # So, to reduce target:
        # We should iterate from high to low.
        # If target has bit i, and basis_vec[i] exists, XOR it.
        
        # Optimization: Only check bits that are set in target.
        # But target changes.
        
        # Simpler: Iterate i from len(primes)-1 down to 0.
        # if v & (1 << i):
        #   if basis_vec[i]:
        #     v ^= basis_vec[i]
        #     s = s.symmetric_difference(basis_src[i])
        #   else:
        #     return None # Not in span
        
        # This is O(len(primes)) which is ~1000.
        
        for i in range(len(primes) - 1, -1, -1):
            if v & (1 << i):
                if basis_vec[i]:
                    v ^= basis_vec[i]
                    s = s.symmetric_difference(basis_src[i])
                else:
                    return None
        if v == 0:
            return s
        return None

    # Iterate M
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        # Add number M to basis
        add_to_basis(masks[M], {M})
        
        # Check if target is in span
        sol = is_in_span(target_mask)
        
        if sol is not None:
            # Verify solution
            # 1. Distinct? Yes, by construction.
            # 2. Size 1..100?
            if 1 <= len(sol) <= 100:
                # 3. Product * 210 is perfect square?
                prod = 210
                for x in sol:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    # 4. Minimize max? We are iterating M upwards, so this is minimal.
                    return [int(x) for x in sol]
            # If size > 100, we might need to find another solution?
            # The linear algebra solution is unique for a given basis?
            # No, the basis is fixed, but the representation of Target is unique in the basis.
            # However, if we had different basis choices, we might get different solutions.
            # But we are just looking for *a* solution.
            # If the unique solution in this basis has size > 100, is there another?
            # The solution space is an affine space.
            # But for a fixed M, the set of available numbers is fixed.
            # The solutions are all subsets that XOR to Target.
            # If the "canonical" one is too big, maybe another is smaller?
            # Unlikely for small M.
            # For now, if size > 100, we continue searching for larger M?
            # Or we can try to find a smaller solution by looking for dependencies?
            # Let's just return it if valid, or continue.
            # Given the constraints, it's highly probable the first found solution is small.
            pass

    # Fallback
    return [840]
