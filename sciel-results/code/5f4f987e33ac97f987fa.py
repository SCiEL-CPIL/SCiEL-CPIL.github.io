"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Precompute squarefree kernels for all values in (210, MAX_VAL]
    kernels = {}
    for v in range(211, MAX_VAL + 1):
        kernels[v] = squarefree_kernel(v)
    
    # We need: product of kernels of selected numbers == sf(210) = 210
    # (since sf(210 * prod) = sf(210) * sf(prod) mod squares, and we need this to be 1)
    # Actually: sf(210 * prod) = sf(210) XOR sf(prod) in terms of prime parity
    # We need sf(210 * prod) = 1, so sf(prod) must equal sf(210) = 210
    
    # Strategy: Try increasing max values from 211 upward.
    # For each candidate max M, check if there's a subset of {211, ..., M} 
    # whose kernel product equals 210.
    
    # Use linear algebra over GF(2) on the prime factorizations of kernels.
    # Primes up to 10210 that appear in any kernel:
    # Let's find all primes <= 10210 first
    
    # Actually, let's use a more direct approach:
    # For each candidate max M (from 211 upward), check if 210 is in the 
    # span (over GF(2)) of the kernel vectors of {211, ..., M}.
    
    # Represent each kernel as a bitmask of its prime factors.
    # First, collect all primes up to 10210
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(MAX_VAL)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Convert each kernel to a bitmask (using Python int as arbitrary-precision bitset)
    kernel_masks = {}
    for v in range(211, MAX_VAL + 1):
        mask = 0
        k = kernels[v]
        for p in primes:
            if p * p > k:
                break
            if k % p == 0:
                mask |= (1 << prime_to_idx[p])
                while k % p == 0:
                    k //= p
        if k > 1:
            mask |= (1 << prime_to_idx[k])
        kernel_masks[v] = mask
    
    # Target: kernel of 210 = 210 itself (since 210 is squarefree)
    target_mask = 0
    k = 210
    for p in primes:
        if p * p > k:
            break
        if k % p == 0:
            target_mask |= (1 << prime_to_idx[p])
            while k % p == 0:
                k //= p
    if k > 1:
        target_mask |= (1 << prime_to_idx[k])
    
    # Now search for minimum M such that target_mask is in span of {kernel_masks[v] : 211 <= v <= M}
    # Use incremental Gaussian elimination
    
    # We'll do incremental: add vectors one by one, maintain row-echelon form
    # When target becomes reachable, we've found our answer
    
    # For efficiency, use a list of basis vectors and check membership
    basis = []  # list of (pivot_bit, vector) in reduced row echelon form
    
    def insert_and_check(vec):
        """Try to reduce vec using basis. Return True if vec is in span, else insert."""
        v = vec
        for pivot_bit, bv in basis:
            if v & (1 << pivot_bit):
                v ^= bv
        if v == 0:
            return True  # in span
        # Find pivot for v
        # Get highest set bit
        pivot = v.bit_length() - 1
        # Insert into basis at correct position (maintain sorted by pivot)
        basis.append((pivot, v))
        basis.sort(key=lambda x: x[0])
        # Reduce existing basis vectors with this new one
        for i, (pb, bv) in enumerate(basis):
            if pb == pivot:
                continue
            if bv & (1 << pivot):
                basis[i] = (pb, bv ^ v)
        return False
    
    # Start with empty basis, add vectors from 211 upward
    # Track which vectors were added for solution reconstruction
    added_vectors = []  # list of (value, mask)
    
    best_solution = None
    best_max = MAX_VAL + 1
    
    for M in range(211, min(MAX_VAL + 1, 3000)):  # Limit search for speed
        if time.time() > deadline:
            break
        
        vec = kernel_masks[M]
        was_in_span = insert_and_check(vec)
        added_vectors.append((M, vec))
        
        # Check if target is now in the span
        if was_in_span:
            # Target was already in span before adding M, so M is not needed
            # But we need to check if target was in span BEFORE this addition
            # Actually, insert_and_check returns True if vec is in span of PREVIOUS basis
            # So if it returns True, target might or might not be in span
            # Let me restructure: check target after each insertion
            pass
        
        # Check if target is in current span
        t = target_mask
        for pivot_bit, bv in basis:
            if t & (1 << pivot_bit):
                t ^= bv
        if t == 0:
            # Target is in span! Now extract the solution.
            # We need to find which subset of {211, ..., M} produces target_mask
            # Use backtracking with the basis
            
            # Rebuild: for each vector in added_vectors, check if it's needed
            # Actually, let's use a different approach:
            # For each vector in order, try to reduce target using it
            
            # Better: do a greedy extraction
            # We have the basis in reduced form. Let's trace back.
            # Actually, simplest: brute force small subsets
            
            # Since we're looking for minimum max, and M is small, 
            # let's try all subsets of {211, ..., M} of size 1, 2, 3, ...
            
            # For size 1: check each v in 211..M
            found = False
            for v in range(211, M + 1):
                if kernel_masks[v] == target_mask:
                    best_solution = [v]
                    best_max = M
                    found = True
                    break
            
            if not found:
                # Size 2
                for i in range(211, M + 1):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, M + 1):
                        if kernel_masks[i] ^ kernel_masks[j] == target_mask:
                            best_solution = [i, j]
                            best_max = M
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Size 3
                for i in range(211, M + 1):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, M + 1):
                        if time.time() > deadline:
                            break
                        for k in range(j + 1, M + 1):
                            if kernel_masks[i] ^ kernel_masks[j] ^ kernel_masks[k] == target_mask:
                                best_solution = [i, j, k]
                                best_max = M
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            if found:
                # Verify
                prod = N
                for x in best_solution:
                    prod *= x
                if is_perfect_square(prod):
                    return best_solution
                else:
                    # Shouldn't happen, but continue searching
                    pass
            else:
                # Try size 4 if M is small enough
                if M - 211 <= 10:
                    for i in range(211, M + 1):
                        if time.time() > deadline:
                            break
                        for j in range(i + 1, M + 1):
                            if time.time() > deadline:
                                break
                            for k in range(j + 1, M + 1):
                                if time.time() > deadline:
                                    break
                                for l in range(k + 1, M + 1):
                                    if kernel_masks[i] ^ kernel_masks[j] ^ kernel_masks[k] ^ kernel_masks[l] == target_mask:
                                        best_solution = [i, j, k, l]
                                        best_max = M
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        prod = N
                        for x in best_solution:
                            prod *= x
                        if is_perfect_square(prod):
                            return best_solution
    
    # Fallback: single number 840
    return [840]
# EVOLVE-BLOCK-END
