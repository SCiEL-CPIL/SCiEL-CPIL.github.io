"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_kernel(n):
    """Return the squarefree kernel of n as a frozenset of primes with odd exponent."""
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp % 2 == 1:
                factors[d] = 1
        d += 1
    if temp > 1:
        factors[temp] = 1
    return frozenset(factors.keys())

def _kernels_to_vectors(kernels, all_primes):
    """Convert list of kernels to list of bit vectors (tuples of 0/1)."""
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    vectors = []
    for k in kernels:
        vec = [0] * len(all_primes)
        for p in k:
            if p in prime_to_idx:
                vec[prime_to_idx[p]] = 1
        vectors.append(tuple(vec))
    return vectors

def _gaussian_elimination_find_subset(vectors, target_vec):
    """
    Find a subset of vectors whose XOR equals target_vec using Gaussian elimination.
    Returns the indices of the selected vectors, or None if no solution.
    Uses row reduction to find a solution.
    """
    n = len(vectors)
    if n == 0:
        return None
    
    m = len(vectors[0]) if vectors else 0
    
    # Work with augmented matrix: each row is (vector, index)
    # We'll do Gaussian elimination to find a solution
    
    # Create augmented rows: (vector_bits, original_index)
    rows = []
    for i, v in enumerate(vectors):
        rows.append((v, i))
    
    # Gaussian elimination
    pivot_cols = []
    row_idx = 0
    reduced_rows = []
    
    for col in range(m):
        if row_idx >= len(rows):
            break
        # Find pivot
        pivot = None
        for i in range(row_idx, len(rows)):
            if rows[i][0][col] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        # Swap
        rows[row_idx], rows[pivot] = rows[pivot], rows[row_idx]
        # Eliminate
        for i in range(row_idx + 1, len(rows)):
            if rows[i][0][col] == 1:
                # XOR row i with row row_idx
                v1, idx1 = rows[row_idx]
                v2, idx2 = rows[i]
                new_v = tuple(a ^ b for a, b in zip(v1, v2))
                rows[i] = (new_v, idx1 ^ idx2)  # This doesn't work for tracking
        pivot_cols.append(col)
        row_idx += 1
    
    # Actually, let me use a cleaner approach with explicit tracking
    # Re-do with proper Gaussian elimination that tracks the combination
    
    # Start fresh
    # We want to find coefficients c_i in {0,1} such that XOR of c_i * v_i = target
    # This is a linear system over GF(2)
    
    # Build augmented matrix: A | b where A is n x m, b is target
    # Actually, we have m equations (one per prime) and n unknowns (one per vector)
    # So: sum_j A[j][i] * x_i = target[j] for each j
    
    # Let's set up: for each prime j, equation: sum over i where v_i has bit j set: x_i = target[j]
    
    # Use standard Gaussian elimination on the system
    # Rows = equations, columns = variables (x_0, ..., x_{n-1})
    
    # Build coefficient matrix: eq_j[i] = 1 if vector i has prime j in its kernel
    # RHS: target[j]
    
    # For efficiency, only include primes that appear in any vector or in target
    active_primes = set()
    for v in vectors:
        for j, bit in enumerate(v):
            if bit:
                active_primes.add(j)
    for j, bit in enumerate(target_vec):
        if bit:
            active_primes.add(j)
    
    active_primes = sorted(active_primes)
    num_eqs = len(active_primes)
    
    if num_eqs == 0:
        # No constraints, any single vector works if target is empty
        if all(b == 0 for b in target_vec):
            return [0] if n > 0 else None
        return None
    
    # Build system: for each active prime j, equation over variables x_0..x_{n-1}
    # Coefficient for variable i in equation j: 1 if vectors[i] has bit at position active_primes[j]
    # RHS for equation j: target_vec[active_primes[j]]
    
    # Gaussian elimination
    # Matrix: num_eqs rows, n columns + 1 (RHS)
    # We'll store as list of (coeff_tuple, rhs)
    
    mat = []
    for j_idx, j in enumerate(active_primes):
        coeff = []
        for i in range(n):
            coeff.append(vectors[i][j])
        rhs = target_vec[j]
        mat.append((tuple(coeff), rhs))
    
    # Gaussian elimination
    rank = 0
    pivot_cols_used = []
    
    for col in range(n):
        if rank >= num_eqs:
            break
        # Find pivot in column col
        pivot_row = None
        for r in range(rank, num_eqs):
            if mat[r][0][col] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        # Swap
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        # Eliminate
        for r in range(num_eqs):
            if r != rank and mat[r][0][col] == 1:
                c1, r1 = mat[rank]
                c2, r2 = mat[r]
                new_c = tuple(a ^ b for a, b in zip(c1, c2))
                mat[r] = (new_c, r1 ^ r2)
        pivot_cols_used.append(col)
        rank += 1
    
    # Check consistency
    for r in range(rank, num_eqs):
        if mat[r][1] == 1:
            # Inconsistent: 0 = 1
            return None
    
    # Back-substitute to find a solution
    # Set free variables to 0, solve for pivot variables
    x = [0] * n
    pivot_col_to_row = {col: rank - 1 - i for i, col in enumerate(pivot_cols_used)}
    # Actually, let's map: for each pivot column, which row has it as pivot
    col_to_pivot_row = {}
    for i, col in enumerate(pivot_cols_used):
        col_to_pivot_row[col] = i  # row i in the reduced system
    
    # Back substitute from bottom
    for i in range(rank - 1, -1, -1):
        col = pivot_cols_used[i]
        coeff, rhs = mat[i]
        # x[col] + sum_{j > col, j is pivot} coeff[j]*x[j] = rhs
        # Since we eliminated, only pivot columns after col have non-zero coeff
        val = rhs
        for j in pivot_cols_used[i+1:]:
            if coeff[j] == 1:
                val ^= x[j]
        x[col] = val
    
    # Check that this solution works
    # Verify: XOR of selected vectors should equal target
    selected_indices = [i for i in range(n) if x[i] == 1]
    
    if len(selected_indices) == 0:
        if all(b == 0 for b in target_vec):
            return [0]  # Empty product is 1, 210*1=210 not square... 
            # Actually if target is empty, we need product kernel = empty, meaning product is perfect square
            # 210 * (perfect square) is not a perfect square since 210 is not
            # So empty solution is invalid unless target is empty AND we interpret differently
            # Wait, target is {2,3,5,7}, so target is not empty
            return None
        return None
    
    # Verify the solution
    result_vec = [0] * m
    for idx in selected_indices:
        for j in range(m):
            result_vec[j] ^= vectors[idx][j]
    
    if tuple(result_vec) != tuple(target_vec):
        return None
    
    return selected_indices

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # 210 = 2 * 3 * 5 * 7
    target_kernel = frozenset([2, 3, 5, 7])
    
    # Precompute kernels for all numbers in (210, MAX_VAL]
    kernels = {}
    for x in range(N + 1, MAX_VAL + 1):
        kernels[x] = _squarefree_kernel(x)
    
    # Collect all primes that appear in any kernel
    all_primes_set = set()
    for k in kernels.values():
        all_primes_set.update(k)
    all_primes = sorted(all_primes_set)
    
    # Build vector for target
    target_vec = [0] * len(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    for p in target_kernel:
        if p in prime_to_idx:
            target_vec[prime_to_idx[p]] = 1
    target_vec = tuple(target_vec)
    
    best_result = None
    best_M = MAX_VAL + 1
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        # Available numbers: (210, M]
        available = list(range(N + 1, M + 1))
        if len(available) > MAX_SIZE:
            available = available[-MAX_SIZE:]  # Take largest MAX_SIZE numbers? 
            # Actually, we should take all, but limit selection to MAX_SIZE
            # For Gaussian elimination, we can have more variables than needed
            # But we need to ensure we don't select more than MAX_SIZE
            # Let's keep all available but track size
        
        # Actually, let's keep all available numbers as variables
        available = list(range(N + 1, M + 1))
        
        # Convert to vectors
        vectors = []
        for x in available:
            k = kernels[x]
            vec = [0] * len(all_primes)
            for p in k:
                if p in prime_to_idx:
                    vec[prime_to_idx[p]] = 1
            vectors.append(tuple(vec))
        
        # Find subset using Gaussian elimination
        indices = _gaussian_elimination_find_subset(vectors, target_vec)
        
        if indices is not None and 1 <= len(indices) <= MAX_SIZE:
            # Verify
            prod = N
            for idx in indices:
                prod *= available[idx]
            if _is_perfect_square(prod):
                best_result = [available[idx] for idx in indices]
                best_M = M
                break
    
    if best_result is None:
        # Fallback: find any valid solution
        # Use a simple constructive approach
        # 210 * 210 = 210^2 is a perfect square, but 210 is not > 210
        # Try: 210 * 441 = 210 * 21^2. 441 > 210, 441 <= 10210
        # 210 * 441 = 210 * 441 = 92610. sqrt(92610) = 304.3... not integer
        # 210 = 2*3*5*7. Need product to have odd exponents for 2,3,5,7.
        # Try x = 2*3*5*7 = 210, but not > 210.
        # Try x = 2*3*5*7 * k^2. For k=2: 210*4 = 840. 210*840 = 210^2 * 4 = (210*2)^2 = 420^2. Perfect square!
        # 840 > 210, 840 <= 10210. So [840] works.
        # max - 210 = 840 - 210 = 630
        best_result = [840]
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
