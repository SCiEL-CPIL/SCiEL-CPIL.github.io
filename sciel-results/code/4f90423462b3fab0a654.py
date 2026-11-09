import math
import time

# Precomputation constants
MAX_N = 10210
START_N = 211
TARGET = 210

def _isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    x = int(math.isqrt(n))
    while x * x > n:
        x -= 1
    while (x + 1) * (x + 1) <= n:
        x += 1
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # 1. Sieve primes up to MAX_N
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(MAX_N)) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
                
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute square-free masks
    # mask[n] is an integer where bit i is 1 if prime[i] divides n to an odd power
    masks = [0] * (MAX_N + 1)
    for n in range(START_N, MAX_N + 1):
        temp = n
        m = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    m |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime
            if temp in prime_to_idx:
                m |= (1 << prime_to_idx[temp])
        masks[n] = m
        
    # Target mask
    target_mask = masks[TARGET] # Wait, TARGET is 210, but we need sf(210).
    # My loop starts at 211. I need sf(210) specifically.
    # 210 = 2 * 3 * 5 * 7.
    # Let's just calculate it manually or extend the loop.
    # 210 is not in [211, 10210].
    # sf(210):
    # 210 / 2 = 105 (cnt 1) -> bit for 2
    # 105 / 3 = 35 (cnt 1) -> bit for 3
    # 35 / 5 = 7 (cnt 1) -> bit for 5
    # 7 / 7 = 1 (cnt 1) -> bit for 7
    # So target_mask has bits for 2, 3, 5, 7.
    
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_to_idx[p])
        
    # 3. Gaussian Elimination
    # basis is a list of tuples (pivot_mask, comb_mask)
    # pivot_mask has a single highest bit set (the pivot)
    # comb_mask is a bitmask of the numbers used to form this basis vector
    # We store basis sorted by pivot bit (highest bit)
    
    basis = []
    
    def reduce_vector(v, c):
        """
        Reduces v using the current basis.
        Returns (reduced_v, updated_c).
        """
        for i, (b_mask, b_comb) in enumerate(basis):
            # Check if the pivot bit of b_mask is set in v
            # The pivot bit is the highest set bit in b_mask
            # We can find it by b_mask & -b_mask? No, that's lowest.
            # We stored them such that each basis vector has a unique pivot.
            # Let's just check if v has the same highest bit as b_mask.
            # Actually, standard GE:
            # If v has the bit corresponding to the pivot of b_mask, XOR them.
            
            # To make it fast, we can store the pivot bit explicitly.
            # But for now, let's assume basis is sorted by pivot bit descending.
            # And we only check if the highest bit of v matches the pivot of b_mask.
            
            # Optimization: Store (pivot_bit, full_mask, comb)
            # Let's refactor basis to be a list of [pivot_bit, mask, comb]
            
            pass # Placeholder, will implement properly below
            
    # Let's re-structure for clarity and speed
    # basis: list of [pivot_bit, mask, comb]
    # We maintain basis in reduced row echelon form (RREF) or just row echelon?
    # Row echelon is sufficient for checking span membership and finding a solution.
    # We insert new vectors into the basis.
    
    basis = [] # List of [pivot_bit, mask, comb]
    
    def add_to_basis(mask, comb):
        v = mask
        c = comb
        # Reduce v by existing basis vectors
        # Sort basis by pivot bit descending to reduce efficiently
        # Actually, we can just iterate through all.
        # Since N is small, O(N^2) is fine.
        
        # To ensure we can reduce any vector, we should keep basis in a form
        # where each basis vector has a unique pivot bit, and no other basis vector
        # has that bit set. (RREF-like for the pivot columns).
        
        # 1. Reduce v by existing basis
        for i in range(len(basis)):
            pivot_bit, b_mask, b_comb = basis[i]
            if v & pivot_bit:
                v ^= b_mask
                c ^= b_comb
                if v == 0:
                    return False # Dependent
        
        if v == 0:
            return False # Dependent
            
        # 2. Find the pivot bit for the new vector
        # The highest set bit in v
        # v.bit_length() gives position of highest bit + 1
        # So highest bit is 1 << (v.bit_length() - 1)
        pivot_bit = 1 << (v.bit_length() - 1)
        
        # 3. Check if a basis vector already has this pivot bit
        # If so, we should have reduced it. Since we reduced against all, 
        # and if there was a basis vector with this pivot, v would have had that bit cleared.
        # So this pivot bit is new.
        
        # 4. Add to basis
        # We also need to eliminate this new pivot bit from other basis vectors
        # to maintain RREF (optional but good for consistency, not strictly required for span check)
        # However, if we don't eliminate, future reductions might be slightly less efficient 
        # or we might need to be careful. 
        # Standard GE for span check:
        # Just keeping a basis where each vector has a unique leading term is enough.
        
        # Let's insert into basis
        # We can just append.
        basis.append([pivot_bit, v, c])
        
        # Optional: Eliminate this pivot bit from other basis vectors to keep them reduced
        # This makes the basis "nicer" but adds O(N) cost per insertion.
        # Given N=10000, O(N^2) total is 10^8, which might be too slow in Python.
        # Let's skip full RREF and just use row echelon.
        # In row echelon, we just need to ensure that when reducing a new vector,
        # we use basis vectors that have pivots *above* (higher bit) or *below*?
        # Standard: Reduce using basis vectors with pivots higher than the current highest bit of v?
        # No, we reduce v by any basis vector that has a bit set in v.
        # To be systematic, we should sort basis by pivot bit.
        
        return True
        
    # Improved Add:
    # We will maintain `basis` as a list of [pivot_bit, mask, comb].
    # We will not maintain full RREF, but we will ensure that when we add a new vector,
    # we reduce it against all existing basis vectors.
    # This ensures the new vector is independent.
    
    # We need a function to check if target is in span.
    # def is_in_span(target_mask):
    #    v = target_mask
    #    for [p, m, c] in basis:
    #       if v & p:
    #          v ^= m
    #    return v == 0
    
    # Wait, if basis is not RREF, reducing target against basis in arbitrary order might not work?
    # Yes, if basis is in row echelon form (each pivot is unique and to the right of previous),
    # then reducing from top to bottom works.
    # My `add_to_basis` doesn't guarantee order.
    
    # Let's enforce order.
    # Sort basis by pivot_bit descending after each addition? O(N log N) per step -> O(N^2 log N). Too slow.
    
    # Alternative: 
    # Use a dictionary `pivot_to_basis` mapping pivot_bit to (mask, comb).
    # When adding a new vector:
    # 1. While v != 0:
    #    a. Find highest bit p in v.
    #    b. If p is in pivot_to_basis:
    #       v ^= pivot_to_basis[p][0]
    #       c ^= pivot_to_basis[p][1]
    #    c. Else:
    #       # This is a new pivot
    #       pivot_to_basis[p] = (v, c)
    #       # We should also eliminate p from other basis vectors? 
    #       # If we don't, the "reduction" step for future vectors might be tricky?
    #       # No, if we always reduce by the *highest* bit present in the vector,
    #       # and we store basis by that highest bit, it works perfectly.
    #       # This is essentially maintaining the basis in a form where
    #       # each basis vector is the unique vector in the span with that specific highest bit
    #       # and no other basis vector has that bit? 
    #       # Actually, if we don't eliminate p from others, other basis vectors might have bit p set.
    #       # But when reducing a new vector w, we look at highest bit of w.
    #       # If it's p, we XOR with basis[p].
    #       # This clears bit p.
    #       # Then we look at next highest bit.
    #       # This works regardless of whether other basis vectors have bit p set,
    #       # as long as basis[p] is the *only* one we use to clear p.
    #       # And since we pick the highest bit of w, and basis[p] has p as its highest bit,
    #       # XORing them clears p and doesn't affect bits higher than p.
    #       # So this strategy is valid.
    #       break
        
    # This dictionary approach is O(D) per reduction, where D is number of set bits in v.
    # Average D is small.
    
    pivot_to_basis = {}
    
    def add_vector(mask, comb):
        v = mask
        c = comb
        while v != 0:
            # Get highest set bit
            p_bit = 1 << (v.bit_length() - 1)
            if p_bit in pivot_to_basis:
                b_mask, b_comb = pivot_to_basis[p_bit]
                v ^= b_mask
                c ^= b_comb
            else:
                # New pivot
                pivot_to_basis[p_bit] = (v, c)
                return True
        return False
        
    def get_solution(target_mask):
        v = target_mask
        c = 0
        while v != 0:
            p_bit = 1 << (v.bit_length() - 1)
            if p_bit in pivot_to_basis:
                b_mask, b_comb = pivot_to_basis[p_bit]
                v ^= b_mask
                c ^= b_comb
            else:
                return None # Not in span
        return c
        
    # Iterate
    for x in range(START_N, MAX_N + 1):
        if time.time() > deadline:
            break
            
        m = masks[x]
        # comb is a single bit for x
        # Bit index for x is x - START_N
        c = 1 << (x - START_N)
        
        add_vector(m, c)
        
        # Check if target is in span
        sol_comb = get_solution(target_mask)
        
        if sol_comb is not None:
            # Extract numbers
            numbers = []
            # sol_comb has bits set for the numbers used
            # Bit i corresponds to number START_N + i
            # We need to iterate through set bits
            # Since N is ~10000, we can just loop
            for i in range(MAX_N - START_N + 1):
                if sol_comb & (1 << i):
                    numbers.append(START_N + i)
                    
            # Verify solution
            if numbers:
                # Check distinctness (guaranteed by bit mask)
                # Check range
                # Check count <= 100
                if len(numbers) <= 100:
                    # Verify square
                    prod = 210
                    for n in numbers:
                        prod *= n
                    s = _isqrt(prod)
                    if s * s == prod:
                        return numbers
                else:
                    # If count > 100, we might need to simplify, 
                    # but for minimal M, this is unlikely.
                    # If it happens, we break and try to find a simpler solution?
                    # For now, if it's the first M, we assume it's valid.
                    # But the problem says 1..100.
                    # If the first M gives >100, it's invalid.
                    # We should continue searching?
                    # No, if M is minimal, any solution must use numbers <= M.
                    # If all solutions use >100 numbers, then no valid solution exists for this M.
                    # We should continue to M+1.
                    pass
                    
    # Fallback
    return [840]

construct()
