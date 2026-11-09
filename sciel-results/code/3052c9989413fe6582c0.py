"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    import sys
    from functools import lru_cache
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    candidates = list(range(2, 41))
    
    def factorize(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
        
    vecs = [factorize(n) for n in candidates]
    n = len(candidates)
    
    best_size = 0
    best_set = []
    
    # We use a recursive branch and bound to find the maximum independent set.
    # State: index to start from, current basis (list of vectors)
    # We maintain the basis in a way that allows quick independence checking.
    # For simplicity and speed given small N, we check rank of basis + new_vec.
    
    def rank_of_matrix(matrix):
        # matrix is list of tuples/lists of ints
        # Compute rank over Q (using fractions or Gaussian elimination with exact arithmetic)
        if not matrix:
            return 0
        m = len(matrix)
        n_cols = len(matrix[0])
        # Copy matrix to list of lists
        mat = [list(row) for row in matrix]
        rank = 0
        for col in range(n_cols):
            # Find pivot
            pivot_row = -1
            for row in range(rank, m):
                if mat[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            # Swap rows
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            # Eliminate
            pivot_val = mat[rank][col]
            for row in range(m):
                if row != rank and mat[row][col] != 0:
                    factor = mat[row][col] / pivot_val
                    for c in range(col, n_cols):
                        mat[row][c] -= factor * mat[rank][c]
            rank += 1
        return rank

    def is_independent(vec, basis):
        # Check if vec is independent of basis
        # basis is a list of vectors
        new_basis = basis + [vec]
        return rank_of_matrix(new_basis) == len(basis) + 1

    def search(idx, basis):
        nonlocal best_size, best_set
        current_size = len(basis)
        if current_size > best_size:
            best_size = current_size
            best_set = basis[:]
            
        # Pruning
        if current_size + (n - idx) <= best_size:
            return
            
        # If we reached the theoretical max (num_primes), we can stop
        if best_size == num_primes:
            return

        for i in range(idx, n):
            if is_independent(vecs[i], basis):
                search(i + 1, basis + [vecs[i]])
                if best_size == num_primes:
                    return

    search(0, [])
    
    # Reconstruct numbers from vectors
    # We need to map vectors back to original numbers.
    # Since vectors might not be unique (e.g. 4 and 8 have different vectors, but what if two numbers had same vector? 
    # No, prime factorization is unique. So each vector corresponds to exactly one number in 2..40?
    # Yes, Fundamental Theorem of Arithmetic.
    
    vec_to_num = {}
    for i in range(n):
        vec_to_num[vecs[i]] = candidates[i]
        
    result = [vec_to_num[v] for v in best_set]
    
    # Verify
    from itertools import combinations
    def verify(nums):
        seen = set()
        seen.add(1)
        k = len(nums)
        for r in range(1, k + 1):
            for combo in combinations(nums, r):
                prod = 1
                for x in combo:
                    prod *= x
                if prod in seen:
                    return False
                seen.add(prod)
        return True

    if not verify(result):
        # Fallback to primes
        result = primes[:]
        
    return result

# EVOLVE-BLOCK-END
