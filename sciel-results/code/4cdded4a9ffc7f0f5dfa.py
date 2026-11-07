"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_kernel(n, primes):
    """Return bitmask of primes with odd exponent in n."""
    mask = 0
    for i, p in enumerate(primes):
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1
        if cnt:
            mask |= (1 << i)
    # If remaining n > 1, it's a prime factor not in our list
    # We'll handle this by extending the prime list if needed
    if n > 1:
        # This means there's a prime factor we didn't account for
        # For correctness, we need to include it
        pass
    return mask, n

def _factorize(n, primes):
    """Factor n using given primes. Return (dict of prime->exponent, remaining)."""
    factors = {}
    for p in primes:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        if cnt:
            factors[p] = cnt
    return factors, n

def construct():
    deadline = time.time() + 75  # 75s safety margin
    
    # 210 = 2 * 3 * 5 * 7
    base_primes = [2, 3, 5, 7]
    base_kernel = 0
    for i, p in enumerate(base_primes):
        base_kernel |= (1 << i)  # 210 has all four primes with odd exponent
    
    # We need the XOR of selected numbers' kernels to equal base_kernel
    # Each number n in (210, M] has a kernel over some set of primes
    
    # Strategy: for each M from 211 upward, check if base_kernel is in the
    # GF(2) span of the kernels of numbers in (210, M]
    
    # First, let's collect kernels incrementally
    # We need to handle primes beyond the base 4
    
    # Collect all primes up to 10210
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    all_primes = sieve(10210)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    num_primes = len(all_primes)
    
    # Precompute kernel for 210
    # 210 = 2*3*5*7
    kernel_210 = 0
    for p in [2, 3, 5, 7]:
        kernel_210 |= (1 << prime_to_idx[p])
    
    # For each M, we add the kernel of M to our set and check if kernel_210 is in the span
    # Use Gaussian elimination to maintain the row-reduced echelon form
    
    # Better approach: collect all kernels up to some M, then do GF(2) elimination
    # to find if kernel_210 is in the span, and if so, find a minimal subset
    
    # Let's search M from 211 upward
    # For efficiency, we'll build up the kernel vectors and maintain a basis
    
    # Store kernels as integers (bitmasks)
    kernels = {}  # number -> kernel bitmask
    numbers_in_range = []  # (number, kernel)
    
    # Precompute kernels for all numbers in (210, 10210]
    for n in range(211, 10211):
        temp = n
        mask = 0
        for i, p in enumerate(all_primes):
            if p * p > temp:
                break
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt ^= 1
            if cnt:
                mask |= (1 << i)
        if temp > 1:
            # temp is a prime factor
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
            else:
                # Should not happen since we sieved up to 10210
                pass
        kernels[n] = mask
        numbers_in_range.append((n, mask))
    
    # Now search for smallest M such that kernel_210 is in the span of kernels[211..M]
    # Use incremental Gaussian elimination
    
    best_result = None
    best_max = 10211
    
    # For each candidate max M, check if we can form kernel_210 from a subset of {211,...,M}
    # and find the actual subset
    
    # Efficient approach: maintain a basis and track which vectors are in the span
    # When kernel_210 enters the span, find the subset via backtracking
    
    # Let's use a different approach: for each M from 211 to 10210,
    # check if kernel_210 is in the span of {kernels[211], ..., kernels[M]}
    
    # Build basis incrementally
    basis = []  # list of (pivot, vector) in row echelon form
    # vector_to_numbers: track which original vectors contributed
    
    # Actually, let's just do: for each M, collect vectors, do GF(2) elimination,
    # check if kernel_210 is reducible to 0
    
    # For speed, let's be smarter: 
    # The kernel of n is determined by its squarefree part.
    # n and n*k^2 have the same kernel for any k.
    
    # Let's try small M first with brute force subset search for small sets
    
    # For M close to 210, the set is small, so we can enumerate subsets
    # For larger M, use GF(2) methods
    
    # Phase 1: Try M from 211 to 250 with subset enumeration (up to 40 elements)
    # For 40 elements, 2^40 is too large. But we can use meet-in-the-middle or GF(2).
    
    # Phase 2: Use GF(2) to find if a solution exists, then extract the subset
    
    # Let's use GF(2) Gaussian elimination for each M
    
    def gf2_reduce(target, vectors):
        """Check if target is in the span of vectors. If so, return the coefficients."""
        # vectors: list of (index, bitmask)
        # Build augmented system: [vectors | target]
        # Actually, we want to find c such that XOR(c_i * v_i) = target
        
        # Standard GF(2) elimination
        # Represent as list of bitmasks
        vecs = [v for _, v in vectors]
        n = len(vecs)
        
        # Gaussian elimination
        # We'll do forward elimination
        mat = vecs[:]  # copy
        # target as column
        # Actually, let's just check if target is in the span by reducing target with the basis
        
        # Build basis from vectors
        basis = []
        for v in vecs:
            x = v
            for b in basis:
                x = min(x, x ^ b, key=int.bit_length)
            if x != 0:
                # Insert x into basis
                basis.append(x)
                # Re-sort basis by pivot
                basis.sort(key=lambda b: b.bit_length() - 1)
        
        # Now reduce target
        x = target
        for b in sorted(basis, key=lambda b: -b.bit_length()):
            if x.bit_length() > b.bit_length() - 1:
                x ^= b
        
        return x == 0
    
    # This is getting complex. Let me use a cleaner approach.
    
    # For each M, collect vectors, do proper GF(2) elimination to check span membership
    # and if in span, find the actual combination.
    
    # Let's precompute and search
    # Key optimization: we only need to check M values where a new "interesting" vector is added
    
    # Actually, the simplest correct approach:
    # For M from 211 to 10210:
    #   vectors = [kernels[n] for n in range(211, M+1)]
    #   Check if kernel_210 is in span(vectors)
    #   If yes, find the subset and return it
    
    # But this is O(M * M * bitops) which might be slow. Let's optimize.
    
    # Incremental: maintain basis as we add vectors one by one
    
    basis = []  # list of (pivot_bit, vector)
    # For finding the actual subset, we need to track linear combinations
    
    # Let's use a different strategy: 
    # 1. Find the smallest M where kernel_210 is in the span
    # 2. Then find a small subset that achieves it
    
    # Step 1: Find smallest M
    # Maintain a basis with tracking
    
    class GF2Space:
        def __init__(self):
            self.basis = {}  # pivot -> vector
            self.combo = {}  # pivot -> frozenset of indices that make this vector
        
        def add(self, idx, vector):
            """Add vector at index idx. Returns True if it's new (independent)."""
            x = vector
            combo = {idx}
            # Reduce
            while x != 0:
                pivot = x.bit_length() - 1
                if pivot in self.basis:
                    x ^= self.basis[pivot]
                    combo = combo ^ self.combo[pivot]
                else:
                    self.basis[pivot] = x
                    self.combo[pivot] = combo
                    return True
            return False
        
        def contains(self, target):
            """Check if target is in the span. If so, return the combo."""
            x = target
            combo = frozenset()
            while x != 0:
                pivot = x.bit_length() - 1
                if pivot in self.basis:
                    x ^= self.basis[pivot]
                    combo = combo ^ self.combo[pivot]
                else:
                    return None
            return combo
        
        def size(self):
            return len(self.basis)
    
    space = GF2Space()
    
    for M in range(211, 10211):
        if time.time() > deadline:
            break
        # Add kernel of M
        k = kernels[M]
        space.add(M, k)
        
        # Check if kernel_210 is in the span
        combo = space.contains(kernel_210)
        if combo is not None:
            # Found a solution! The selected numbers are those in combo
            selected = sorted(combo)
            # Verify: all in (210, M], distinct
            # The max is M (or possibly less if M's vector wasn't needed)
            actual_max = max(selected)
            
            # Verify the solution
            prod = 1
            for n in selected:
                prod *= n
            total = 210 * prod
            if _is_perfect_square(total):
                if actual_max < best_max:
                    best_max = actual_max
                    best_result = selected
            
            # We found the smallest M where it's in the span, 
            # but the actual subset might not include M
            # Let's also try to find a smaller max by checking if 
            # kernel_210 is in the span of a subset not including the largest elements
            
            # For now, return this solution
            # But we can try to minimize the max further
            break
    
    if best_result is not None:
        # Try to improve: can we find a subset with smaller max?
        # The current solution has max = best_max
        # Check if there's a solution with max < best_max
        # This would mean kernel_210 is in the span of {211, ..., best_max-1}
        
        # Let's search for better solutions with smaller max
        # Reset and search up to best_max - 1
        space2 = GF2Space()
        for M in range(211, best_max):
            if time.time() > deadline:
                break
            k = kernels[M]
            space2.add(M, k)
        
        combo2 = space2.contains(kernel_210)
        if combo2 is not None:
            selected2 = sorted(combo2)
            actual_max2 = max(selected2)
            prod = 1
            for n in selected2:
                prod *= n
            if _is_perfect_square(210 * prod) and actual_max2 < best_max:
                best_max = actual_max2
                best_result = selected2
    
    if best_result is not None:
        # Final verification
        prod = 1
        for n in best_result:
            prod *= n
        assert _is_perfect_square(210 * prod)
        assert all(isinstance(n, int) for n in best_result)
        assert all(n > 210 and n <= 10210 for n in best_result)
        assert len(best_result) >= 1 and len(best_result) <= 100
        assert len(set(best_result)) == len(best_result)  # distinct
        
        return list(best_result)
    
    # Fallback: single element 840 (210 * 4)
    return [840]

# EVOLVE-BLOCK-END
