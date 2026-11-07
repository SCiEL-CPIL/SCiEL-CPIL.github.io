"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def construct():
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # We need a set S of distinct integers such that:
    # 210 < x <= 10210 for all x in S
    # |S| <= 100
    # product(S) * 210 is a perfect square
    # Minimize max(S) - 210
    
    # Let P = product(S). We require P * 210 = k^2.
    # This implies that the square-free part of P must be the square-free part of 210.
    # 210 = 2 * 3 * 5 * 7.
    # So, the product of the elements in S must have the same square-free part as 210.
    # In other words, if we write each integer x as x = s(x) * y^2 where s(x) is square-free,
    # then the product of s(x) for x in S must be square-free and equal to 210.
    # Actually, more precisely:
    # Let x = 2^a2 * 3^a3 * 5^a5 * 7^a7 * ... 
    # The condition is that the parity of the exponents of primes in P * 210 is all even.
    # Let v_p(n) be the exponent of prime p in n.
    # We need v_p(P) + v_p(210) to be even for all p.
    # This means v_p(P) % 2 == v_p(210) % 2.
    # For p=2,3,5,7: v_p(210)=1, so v_p(P) must be odd.
    # For p > 7: v_p(210)=0, so v_p(P) must be even.
    
    # Strategy:
    # We want to minimize max(S). Let M = max(S). We iterate M from 211 upwards.
    # For a fixed M, we want to find a subset S of {211, ..., M} such that the parity condition holds.
    # To minimize the number of elements (or just to find *any* valid set), we can use linear algebra over GF(2).
    # Each number x in [211, M] can be represented by a vector of the parities of the exponents of all primes <= M.
    # However, the number of primes is large.
    # But note that we only care about the square-free kernel.
    # Let's map each integer x to its square-free part s(x).
    # The condition is that the product of s(x) for x in S is 210.
    # Since s(x) is square-free, the product of s(x) is square-free if and only if no prime appears in more than one s(x) with odd multiplicity?
    # No. The product of square-free numbers is not necessarily square-free.
    # Example: s(2)=2, s(8)=2. Product = 4 = 2^2.
    # The condition is on the *total* exponent parity.
    
    # Let's use the vector space approach over GF(2) for the exponents of primes.
    # The primes involved are those <= M.
    # Let P_M be the set of primes <= M.
    # For each x in [211, M], define vector v_x in GF(2)^{|P_M|} where the component for prime p is v_p(x) % 2.
    # We need a subset S such that sum_{x in S} v_x = v_210 (component-wise mod 2).
    # v_210 has 1s for primes 2,3,5,7 and 0s elsewhere.
    
    # We want to find such a subset S.
    # To minimize M, we check M = 211, 212, ...
    # For each M, we form the matrix A where rows are the numbers 211..M and columns are primes <= M.
    # We want to solve A * c = b where c is the selection vector (0/1) and b is the target vector.
    # We also want to minimize the "cost" of the solution? No, the task says "Minimize max(selected integers)-210".
    # This is equivalent to finding the smallest M such that a solution exists.
    # Once a solution exists for M, we can pick any valid subset.
    # However, we also have the constraint |S| <= 100.
    # Gaussian elimination will give us a basis. We can find a particular solution.
    # The particular solution might have many elements.
    # But usually, the dimension of the space is small compared to the number of numbers.
    # Wait, the number of primes up to 10210 is about 1200.
    # The number of integers is M - 210.
    # If M is small, e.g., 211, there is 1 number.
    # If M is large, there are many numbers.
    
    # Let's try to find the minimal M.
    # We can iterate M from 211 to 10210.
    # For each M, we check if there is a subset of {211, ..., M} satisfying the condition.
    # This is a linear system over GF(2).
    # We need to be careful about performance.
    # 10210 - 210 = 10000 numbers.
    # Primes up to 10210: ~1200.
    # Building the matrix for each M from scratch is O(N * P * log N) which might be too slow if done for every M.
    
    # Alternative:
    # We can just pick a specific set of numbers that works.
    # The problem asks to *construct* a list. It doesn't say we must find the absolute minimum M via search, but the score is based on `max(selected) - 210`. So we DO want the minimum M.
    
    # Let's analyze the structure.
    # We need product(S) * 210 to be a square.
    # This is equivalent to: product(S) = 210 * k^2.
    # So we need to find a set of distinct integers > 210 whose product is 210 times a square.
    
    # Simple construction:
    # Pick numbers that are themselves squares times a factor from {2,3,5,7}.
    # For example, if we pick 210 * 2^2 = 840, 210 * 3^2 = 1890, etc.
    # Product of {840} = 840. 840 * 210 = 840 * 210 = 4 * 210^2. Square!
    # So S = [840] is a valid solution.
    # max(S) - 210 = 840 - 210 = 630.
    # Can we do better?
    
    # What if we pick smaller numbers?
    # We need product(S) to have the same square-free part as 210.
    # Square-free part of 210 is 210.
    # So product(S) must be 210 * square.
    
    # Let's try to find a set with max element < 840.
    # Try M = 211.
    # Numbers: 211.
    # 211 * 210 = 211 * 210. 211 is prime. 210 = 2*3*5*7.
    # Product = 2*3*5*7*211. Not a square.
    
    # Try M = 212.
    # Numbers: 211, 212.
    # 212 = 4 * 53 = 2^2 * 53. Square-free part 53.
    # 211 is prime. Square-free part 211.
    # Subsets:
    # {211}: sf=211. Need 210. No.
    # {212}: sf=53. Need 210. No.
    # {211, 212}: sf = 211 * 53. Need 210. No.
    
    # We need the product of square-free parts to be 210 (mod squares).
    # i.e., xor of the prime exponent vectors equals the vector of 210.
    
    # Let's write a solver that finds the minimal M.
    # We will use Gaussian elimination on the vectors of the numbers 211..M.
    # The target is the vector of 210.
    # We need to check if the target is in the span of the vectors.
    # If yes, we extract a solution.
    # We also need to ensure the solution size <= 100.
    # Since the rank is at most the number of primes, and we are adding vectors one by one,
    # the solution found by Gaussian elimination (back substitution) will have weight at most the rank.
    # The rank is bounded by the number of primes involved.
    # For M < 840, the primes are small.
    
    # Let's implement the search for minimal M.
    
    # Step 1: Generate primes up to 10210.
    # Step 2: Precompute the square-free part vector (or just the square-free part value) for each number 211..10210.
    # Actually, representing as a set of primes (square-free kernel) is easier for XOR.
    # Or a bitset.
    # Primes up to 10210: ~1200. Python integers can handle bitsets of 1200 bits easily.
    
    # Map each prime to an index.
    # For each number x, compute its square-free kernel as a bitmask.
    
    # Then, for M from 211 to 10210:
    #   Add the bitmask of M to the list.
    #   Check if target_mask (210) is in the span of the masks of 211..M.
    #   To do this efficiently, we maintain a basis in row-echelon form.
    #   When we add a new vector, we try to reduce it with the existing basis.
    #   If it reduces to 0, it's dependent.
    #   If not, add it to the basis.
    #   We also need to track the original indices to reconstruct the subset.
    
    # Standard linear basis construction:
    # basis[i] = (vector, mask_of_indices)
    # where vector has leading bit at position i (or similar).
    
    # Let's refine the "check if in span" part.
    # We maintain a basis for the subspace spanned by the vectors seen so far.
    # We also maintain for each basis vector, the combination of original indices that produce it.
    # When we want to check if target is in the span, we try to reduce the target using the basis.
    # If it reduces to 0, it is in the span.
    # The combination of indices is the XOR of the index masks of the basis vectors used.
    
    # This allows us to find a solution as soon as the target becomes reachable.
    # The first M for which this happens is the minimal max element.
    
    # Complexity:
    # M goes up to ~1000. (840 is a candidate, maybe we can find smaller).
    # Each step involves XOR operations on 1200-bit integers.
    # This is very fast in Python.
    
    # Let's code this.
    
    # 1. Sieve of Eratosthenes for primes up to 10210.
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute square-free kernel bitmask for each number in 211..10210
    # kernel[x] = bitmask
    kernel = [0] * (limit + 1)
    for x in range(2, limit + 1):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime
            mask |= (1 << prime_to_idx[temp])
        kernel[x] = mask
    
    # Target mask for 210
    target_mask = kernel[210]
    
    # 3. Iterate M from 211 to limit
    # Maintain basis
    # basis is a list of [bitmask, index_mask]
    # We want to organize basis such that basis[i] has the highest set bit at position i?
    # Or just standard Gaussian elimination insertion.
    
    # Let's use a dictionary or list of size num_primes.
    # basis[i] stores the vector with leading bit i.
    # And combo[i] stores the index mask for that vector.
    
    basis = [None] * num_primes
    combo = [0] * num_primes
    
    def add_to_basis(vec, idx_mask):
        # Returns True if vec was added to basis, False if it was reduced to 0 (dependent)
        v = vec
        c = idx_mask
        while v:
            # Find highest set bit
            # In Python, v.bit_length() - 1 gives the index of the highest set bit
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
        # Returns (bool, index_mask)
        v = target
        c = 0
        while v:
            hb = v.bit_length() - 1
            if basis[hb] is None:
                return False, 0
            v ^= basis[hb]
            c ^= combo[hb]
        return True, c
    
    # Iterate M
    for M in range(211, limit + 1):
        # Add M to the system
        idx_mask = 1 << (M - 211)  # Bit position for index M
        # Note: M can be up to 10210, so index can be up to 10000.
        # Python integers handle this fine.
        
        add_to_basis(kernel[M], idx_mask)
        
        found, sol_mask = is_in_span(target_mask)
        if found:
            # Extract the solution
            # sol_mask has bits set for the selected numbers
            # We need to convert sol_mask back to a list of integers
            selected = []
            m = sol_mask
            offset = 211
            i = 0
            while m:
                if m & 1:
                    selected.append(offset + i)
                m >>= 1
                i += 1
            
            # Check constraints
            # Distinct: Yes, by construction.
            # > 210: Yes, min is 211.
            # <= 10210: Yes, M <= 10210.
            # Size <= 100:
            # The weight of sol_mask is the number of selected elements.
            # In Gaussian elimination, the solution found is a linear combination of basis vectors.
            # The weight can be up to the rank.
            # Rank <= num_primes.
            # But usually much smaller for small M.
            # Let's check the size.
            if len(selected) <= 100:
                # Verify the condition
                prod = 1
                for x in selected:
                    prod *= x
                total = prod * 210
                # Check if perfect square
                root = math.isqrt(total)
                if root * root == total:
                    return selected
                else:
                    # Should not happen if logic is correct
                    # But if it does, we might have a bug.
                    # Let's continue searching? Or return anyway?
                    # The task says "Any broken rule makes the whole result invalid".
                    # So we must ensure validity.
                    # If verification fails, we should not return.
                    # But theoretically it should pass.
                    pass
            else:
                # If size > 100, we might need to find a different solution?
                # The problem doesn't ask to minimize size, just max element.
                # But if the minimal max element solution has size > 100, it's invalid.
                # We would need to find another solution with the same max element but smaller size?
                # Or a larger max element with smaller size?
                # The score is max(selected) - 210.
                # If the first valid M gives a set with size > 100, that M is not a valid candidate.
                # We must continue to next M?
                # No, if M is the minimal max element for *any* valid set, and that set has size > 100,
                # then M is not achievable. We must find the next M that yields a valid set of size <= 100.
                # However, usually the weight is small.
                # Let's assume for now that the first found solution is valid.
                # If not, we would need a more complex search (e.g. finding min weight vector in a coset).
                # Given the constraints, let's hope the weight is small.
                pass

    # Fallback: return a known valid solution
    # S = [840]
    return [840]

# EVOLVE-BLOCK-END
