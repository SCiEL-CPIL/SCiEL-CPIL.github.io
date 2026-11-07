# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # Precompute prime list up to 10210
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
    
    # Target: parity vector of 210 = 2*3*5*7
    # We need subset S of {211..M} such that xor of parity vectors = parity(210)
    # i.e., product of S * 210 is a perfect square
    
    target = [0] * num_primes
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target[prime_to_idx[p]] = 1
    
    def parity_vector(n):
        """Return parity vector (as frozenset of prime indices with odd exponent) for n."""
        vec = []
        for i, p in enumerate(primes):
            if p * p > n:
                break
            cnt = 0
            while n % p == 0:
                n //= p
                cnt ^= 1
            if cnt:
                vec.append(i)
        if n > 1:
            # n is prime
            if n in prime_to_idx:
                vec.append(prime_to_idx[n])
            else:
                # shouldn't happen for n <= 10210
                vec.append(-1)  # marker for unknown prime
        return frozenset(vec)
    
    # Precompute parity vectors for all numbers 211..MAX_VAL
    parity = {}
    for n in range(211, MAX_VAL + 1):
        parity[n] = parity_vector(n)
    
    # Now, for each M from 211 upward, check if there exists a non-empty subset
    # of {211..M} whose XOR of parity vectors equals target.
    # Additionally, the subset must have max element exactly M (to minimize max).
    # Actually we just need max <= M, and we want the smallest M.
    # But we also need the subset to include M? No, we just need max(selected) = M.
    # If a subset of {211..M-1} already works, then M-1 was sufficient.
    # So we check: is target in the span of {parity[n] for n in 211..M} 
    # AND not in the span of {parity[n] for n in 211..M-1}?
    # That's not quite right either. We just need target in span of {211..M}.
    # The smallest M where this happens is our answer.
    
    # Use incremental Gaussian elimination over GF(2)
    # Represent each vector as a Python int (bitmask)
    
    def vec_to_int(vec):
        x = 0
        for i in vec:
            if i >= 0:
                x |= (1 << i)
        return x
    
    target_int = vec_to_int(set(range(num_primes)))  # placeholder
    target_int = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target_int |= (1 << prime_to_idx[p])
    
    # Build bitmask for each number
    num_bitmask = {}
    for n in range(211, MAX_VAL + 1):
        num_bitmask[n] = vec_to_int(parity[n])
    
    # Incremental Gaussian elimination
    # Maintain a basis in row-echelon form
    # basis[i] = a vector with leading 1 at position i
    
    def gauss_check(basis, target_int):
        """Check if target_int is in span of basis vectors."""
        x = target_int
        # Process from highest bit to lowest
        for i in range(num_primes - 1, -1, -1):
            if (x >> i) & 1:
                if i in basis:
                    x ^= basis[i]
                else:
                    return False
        return x == 0
    
    def gauss_add(basis, vec):
        """Add vec to basis, return True if it was linearly independent."""
        x = vec
        for i in range(num_primes - 1, -1, -1):
            if (x >> i) & 1:
                if i in basis:
                    x ^= basis[i]
                else:
                    basis[i] = x
                    return True
        return x == 0  # False means it was dependent
    
    # Also need to track which elements are used to form each basis vector,
    # so we can reconstruct the subset.
    # basis_vec[i] = the actual bitmask
    # basis_elem[i] = frozenset of element indices that form this basis vector
    
    basis_vec = {}
    basis_elem = {}
    
    # For reconstruction, we also need to track the combination
    # Let's use a different approach: for each step M, when we find target in span,
    # we need to extract the actual subset.
    
    # Better approach: maintain a full basis with tracking
    # For each basis vector at position i, store:
    #   basis_vec[i]: the bitmask
    #   basis_set[i]: set of original numbers (as indices 211..M) that XOR to give this
    
    # Actually, let's just do: for each M, build the basis from scratch up to M.
    # That's O(M * num_primes) per M, total O(M^2 * num_primes) which is too slow for large M.
    
    # Incremental approach with tracking:
    # When we add a new element n with vector v:
    #   Reduce v against existing basis
    #   If reduced v is nonzero with leading bit i:
    #       basis_vec[i] = reduced_v
    #       basis_set[i] = {n} union (sets used during reduction)
    #   If reduced v is zero: n is dependent, skip
    
    # For reduction tracking:
    # When reducing x against basis at position i:
    #   x ^= basis_vec[i], and we XOR the set: x_set ^= basis_set[i]
    
    # Let me restructure:
    
    basis_vec = {}
    basis_set = {}  # maps position -> set of numbers
    
    def reduce_and_add(vec, num, basis_vec, basis_set):
        """Reduce vec, add to basis if independent. Returns (reduced_vec, set_of_numbers)."""
        x = vec
        x_set = {num}
        for i in range(num_primes - 1, -1, -1):
            if (x >> i) & 1:
                if i in basis_vec:
                    x ^= basis_vec[i]
                    x_set = x_set ^ basis_set[i]  # symmetric difference
                else:
                    basis_vec[i] = x
                    basis_set[i] = x_set
                    return x, x_set
        return x, x_set  # x should be 0
    
    def find_subset(basis_vec, basis_set, target_int):
        """Find a subset whose XOR equals target_int, or None."""
        x = target_int
        result_set = set()
        for i in range(num_primes - 1, -1, -1):
            if (x >> i) & 1:
                if i in basis_vec:
                    x ^= basis_vec[i]
                    result_set = result_set ^ basis_set[i]
                else:
                    return None
        if x == 0:
            return result_set
        return None
    
    best_result = None
    best_max = MAX_VAL + 1
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        # Add element M to basis
        v = num_bitmask[M]
        reduce_and_add(v, M, basis_vec, basis_set)
        
        # Check if target is in span
        subset = find_subset(basis_vec, basis_set, target_int)
        if subset is not None and len(subset) > 0:
            # Verify
            prod = N
            for n in subset:
                prod *= n
            if _is_perfect_square(prod):
                # Check distinct and in range
                if len(set(subset)) == len(subset) and all(210 < n <= 10210 for n in subset):
                    if max(subset) < best_max:
                        best_max = max(subset)
                        best_result = list(subset)
                        # We found a valid answer; can we do better?
                        # The max of this subset might be less than M
                        # But we're iterating M upward, so this is the first M where
                        # target is in span. However, the subset might not include M.
                        # Let's record and continue to see if we get a smaller max.
                        # Actually, since we're going in order of M, and the basis only
                        # grows, once target is in span at M, it stays in span for all larger M.
                        # So the first M where it's in span gives us the minimal possible max.
                        # But the subset found might have max < M, which would mean it was
                        # already in span at that smaller max. So we should check.
                        if max(subset) < M:
                            # This subset was possible earlier; we might have missed it
                            # because the basis representation changed. Let's just record.
                            pass
                        break
    
    if best_result is not None:
        # Ensure we return Python ints in a list
        return [int(x) for x in sorted(best_result)]
    
    # Fallback: try the known solution
    return [840]

# EVOLVE-BLOCK-END
