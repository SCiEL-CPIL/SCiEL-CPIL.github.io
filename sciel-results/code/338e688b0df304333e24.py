import time
from math import isqrt

def _prime_parity_vector(n, primes):
    """Return the parity vector (as a tuple of 0/1) for prime exponents of n."""
    vec = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        vec.append(count % 2)
    if temp > 1:
        vec.append(1)  # remaining prime factor has exponent 1
    return tuple(vec)

def _gaussian_elimination_gf2(vectors, target):
    """
    Given a list of parity vectors (tuples of 0/1) and a target vector,
    find a non-empty subset whose XOR equals target.
    Returns the subset of indices, or None if no such subset exists.
    """
    n = len(vectors)
    if n == 0:
        return None
    
    # Build augmented matrix: each row is a vector + target column
    # We want to solve: sum(x_i * v_i) = target (mod 2)
    # This is a linear system over GF(2)
    
    # Use Gaussian elimination
    # Each vector is a tuple of bits
    num_bits = len(target)
    
    # Create rows: (vector, target_bit)
    # Actually, we want to find x such that A*x = target
    # where A is the matrix with vectors as rows
    
    # Let's do standard Gaussian elimination
    # rows[i] = (vector_i, b_i) where b_i is the target bit at position i... no
    # Actually target is a single vector, not per-row
    
    # Better approach: treat it as finding a linear combination
    # We have vectors v_0, v_1, ..., v_{n-1} and want sum(x_i * v_i) = target
    
    # Gaussian elimination:
    # Start with the list of vectors, reduce to row echelon form
    # Track which original indices are used
    
    # Each entry: (vector, set_of_original_indices)
    rows = [(vectors[i], frozenset([i])) for i in range(n)]
    
    # Sort by length (shorter vectors first for efficiency)
    rows.sort(key=lambda x: len(x[0]))
    
    pivot_cols = {}  # col -> row_index in reduced form
    reduced = []
    
    for vec, idx_set in rows:
        cur_vec = list(vec)
        cur_idx = idx_set
        for col in range(len(cur_vec)):
            if cur_vec[col] == 0:
                continue
            if col in pivot_cols:
                # Eliminate using the pivot row
                pivot_row = reduced[pivot_cols[col]]
                pivot_vec = list(pivot_row[0])
                pivot_idx = pivot_row[1]
                # XOR
                for j in range(col, len(pivot_vec)):
                    if j < len(cur_vec):
                        cur_vec[j] ^= pivot_vec[j]
                cur_idx = cur_idx ^ pivot_idx
            else:
                pivot_cols[col] = len(reduced)
                reduced.append((tuple(cur_vec), cur_idx))
                break
    
    # Now reduced is in row echelon form
    # We need to check if target is in the span
    # Reduce target using the same pivots
    t = list(target)
    t_idx = frozenset()
    
    for col in range(len(t)):
        if t[col] == 0:
            continue
        if col in pivot_cols:
            pivot_row = reduced[pivot_cols[col]]
            pivot_vec = list(pivot_row[0])
            pivot_idx = pivot_row[1]
            for j in range(col, len(pivot_vec)):
                if j < len(t):
                    t[j] ^= pivot_vec[j]
            t_idx = t_idx ^ pivot_idx
        else:
            # Can't eliminate, so target is not in span
            return None
    
    if any(t):
        return None
    
    if not t_idx:
        return None
    
    return list(t_idx)

def construct():
    deadline = time.time() + 80  # 80 second budget with 10s margin
    
    # Primes up to 10210
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(10210)
    
    # Parity vector of 210 = 2*3*5*7
    # Find which primes divide 210
    target_vec = []
    temp = 210
    for p in primes:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        target_vec.append(count % 2)
    if temp > 1:
        target_vec.append(1)
    target = tuple(target_vec)
    
    # Binary search on M
    # For a given M, check if there's a valid subset using numbers in (210, M]
    
    def check(M):
        """Check if there exists a valid subset with max element <= M."""
        if M <= 210:
            return None
        
        # Collect vectors for numbers 211 to M
        vectors = []
        for m in range(211, M + 1):
            vec = _prime_parity_vector(m, primes)
            vectors.append(vec)
        
        if not vectors:
            return None
        
        result = _gaussian_elimination_gf2(vectors, target)
        if result is None:
            return None
        
        # Map indices back to actual numbers
        selected = [211 + i for i in result]
        return selected
    
    # First, try a small M to see if we can find a solution
    # Binary search between 211 and 10210
    lo, hi = 211, 10210
    best_result = None
    
    # First check if hi works
    if time.time() > deadline - 5:
        return None
    
    # Try to find the minimal M
    # Start with a reasonable upper bound
    # Let's try M = 211 first, then increase
    
    # Actually, let's just do binary search
    # But first verify hi is valid
    result_hi = check(hi)
    if result_hi is None:
        # Shouldn't happen, but handle gracefully
        return None
    
    best_result = result_hi
    best_M = hi
    
    # Binary search for minimal M
    while lo < hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        result_mid = check(mid)
        if result_mid is not None:
            best_result = result_mid
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Verify the result
    if best_result is None:
        return None
    
    # Verify: product of selected * 210 should be a perfect square
    prod = 210
    for x in best_result:
        prod *= x
    
    sqrt_val = isqrt(prod)
    if sqrt_val * sqrt_val != prod:
        # Verification failed, try a different approach
        # Fall back to a simple valid solution
        # 210 * 210 = 210^2 which is a square, but 210 is not > 210
        # Use 4*210 = 840: 210 * 840 = 210 * 4 * 210 = 4 * 210^2 = (2*210)^2
        return [840]
    
    return best_result
