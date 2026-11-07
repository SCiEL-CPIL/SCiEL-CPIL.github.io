"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit+1) if is_prime[i]]

def _parity_vector(n, primes, prime_to_idx):
    """Compute parity vector of n over the given primes."""
    vec = [0] * len(primes)
    temp = n
    for i, p in enumerate(primes):
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        if count % 2 == 1:
            vec[i] = 1
    if temp > 1:
        # temp is a prime factor
        if temp in prime_to_idx:
            vec[prime_to_idx[temp]] = 1
    return vec

def _gaussian_elimination(rows, target, max_size=100):
    """
    Find a subset of rows (by index) such that XOR of those rows = target.
    rows: list of bitsets (Python ints)
    target: bitset (Python int)
    Returns list of indices or None.
    """
    n = len(rows)
    m = target.bit_length()  # number of bits
    
    # We want to solve: sum of selected rows = target (mod 2)
    # This is equivalent to: target is in the span of rows
    # Use Gaussian elimination to find a representation
    
    # Build augmented matrix: [row | 1] for each row, and we want to reach [target | 1]
    # Actually, let's use a different approach: 
    # Perform Gaussian elimination on the rows, tracking which original rows were used
    
    # Better approach: treat this as finding x such that A*x = b over GF(2)
    # where A is the matrix with rows as columns (transpose), and b = target
    
    # Let's use the standard approach: 
    # We have n vectors in GF(2)^m. We want a subset that XORs to target.
    # This is solvable iff target is in the span of the vectors.
    
    # Use Gaussian elimination on the vectors, keeping track of the combination
    
    # Represent each vector as an int, and also track the "recipe" (which original vectors combine to make it)
    
    # For efficiency, let's do row-reduction style but on the vector space
    
    # Actually, simplest: augment each row with a unit vector in GF(2)^n to track the combination
    # But n can be up to 10000, so this is expensive.
    
    # Alternative: just check if target is in the span, and if so, find ONE solution
    # We don't need the MINIMAL solution necessarily, just ANY solution with size <= 100
    
    # Let's do Gaussian elimination properly
    
    # Build the matrix as list of (value, recipe) where recipe is a frozenset of indices
    # This could be memory-intensive for large n
    
    # For now, let's just check feasibility first, then find a solution
    
    # Check if target is in span of rows
    # Gaussian elimination on the rows
    
    basis = []  # list of (pivot_col, value)
    for i, row in enumerate(rows):
        val = row
        for pivot_col, basis_val in basis:
            if val & (1 << pivot_col):
                val ^= basis_val
        if val == 0:
            continue
        # Find highest set bit
        pivot_col = val.bit_length() - 1
        # Check if we already have a basis vector with this pivot
        found = False
        for j, (pc, bv) in enumerate(basis):
            if pc == pivot_col:
                # Reduce val using bv, or reduce bv using val
                # For now, just skip (this means the row is dependent)
                found = True
                break
        if not found:
            basis.append((pivot_col, val))
    
    # Now check if target can be reduced to 0 using basis
    val = target
    for pivot_col, basis_val in sorted(basis, key=lambda x: -x[0]):
        if val & (1 << pivot_col):
            val ^= basis_val
    if val != 0:
        return None  # Not in span
    
    # Target is in the span. Now find a solution.
    # This is tricky without tracking recipes. Let's use a different approach.
    
    # For finding a solution, let's use the fact that we can express target as a combination
    # of basis vectors, and each basis vector is a combination of original rows.
    
    # Let me redo this with recipe tracking, but be careful about memory.
    
    return None  # Placeholder - need to implement properly

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline for safety
    
    n = P['n']  # 210
    max_possible = 10210
    
    # Precompute primes up to max_possible
    primes = _sieve_primes(max_possible)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Target parity vector: 210 = 2 * 3 * 5 * 7
    target = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_idx:
            target |= (1 << prime_to_idx[p])
    
    # Binary search on max_val
    lo = n + 1  # 211
    hi = max_possible  # 10210
    
    best_result = None
    
    # First, let's try a direct approach for small ranges
    # For each max_val, check if there's a valid subset
    
    # Given time constraints, let's try a greedy/heuristic approach first
    # Then refine with GF(2) if needed
    
    # For now, let's just try to find ANY valid set with small max_val
    # Start with max_val = 211, 212, etc.
    
    for max_val in range(lo, hi + 1):
        if time.time() > deadline:
            break
        
        # Compute parity vectors for all numbers in [n+1, max_val]
        num_vars = max_val - n  # number of candidate numbers
        if num_vars > 10000:
            break
        
        rows = []
        for num in range(n + 1, max_val + 1):
            vec = _parity_vector(num, primes, prime_to_idx)
            # Convert to int bitset
            bitset = 0
            for i, bit in enumerate(vec):
                if bit:
                    bitset |= (1 << i)
            rows.append(bitset)
        
        # Check if target is in the span of rows
        # Use Gaussian elimination
        basis = {}  # pivot_col -> value
        for row in rows:
            val = row
            for pivot_col in sorted(basis.keys(), reverse=True):
                if val & (1 << pivot_col):
                    val ^= basis[pivot_col]
            if val == 0:
                continue
            pivot_col = val.bit_length() - 1
            if pivot_col not in basis:
                basis[pivot_col] = val
        
        # Check if target is reducible to 0
        val = target
        for pivot_col in sorted(basis.keys(), reverse=True):
            if val & (1 << pivot_col):
                val ^= basis[pivot_col]
        
        if val == 0:
            # Target is in the span! Find a solution.
            # For now, just return a simple solution: try to find a small subset
            # Greedy: try single elements, then pairs, etc.
            
            # First, check if any single element equals target
            for i, row in enumerate(rows):
                if row == target:
                    result = [n + 1 + i]
                    return result
            
            # Check pairs
            for i in range(len(rows)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(rows)):
                    if (rows[i] ^ rows[j]) == target:
                        result = [n + 1 + i, n + 1 + j]
                        return result
            
            # For larger subsets, use a more systematic approach
            # This is getting complex; for now, let's just indicate feasibility
            # and use a simple construction
            
            # Actually, let's try to find a solution using the basis
            # This requires tracking recipes, which is complex
            # For now, skip and try next max_val
            
            continue
    
    # If we didn't find anything, return a default
    return [4 * n]
# EVOLVE-BLOCK-END
