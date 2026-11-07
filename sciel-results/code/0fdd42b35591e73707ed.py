"""Rewritten program using linear independence over prime exponents."""

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Map each number in 2..40 to its exponent vector
    # Vector has length num_primes
    def get_vector(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        # If n > 1, it has a prime factor > 37, which is impossible for n <= 40
        # except if the range was larger, but here n <= 40.
        return vec

    candidates = list(range(2, 41))
    vectors = {n: get_vector(n) for n in candidates}
    
    # We want to find the largest linearly independent subset.
    # Since the dimension is 12, the max size is 12.
    # We can just pick the primes, as they form a basis.
    # However, to be robust and follow the "Gaussian Elimination" strategy,
    # we perform a greedy selection with Gaussian elimination to find a basis.
    
    basis = []
    selected = []
    
    # Sort candidates to ensure determinism, though order doesn't change the max size here
    # because the primes are a subset and form a basis.
    # Any set of 12 linearly independent vectors is a basis.
    
    for n in candidates:
        v = vectors[n]
        # Check if v is linearly independent of current basis
        # Perform Gaussian elimination on [basis + [v]]
        # If rank increases, add to basis
        
        # Create a list of vectors to eliminate
        # We maintain the basis in row-echelon form (simplified)
        # For small dimensions, we can just check if v is in the span.
        
        # Simple check: Add v to a copy of basis, run elimination, check rank.
        # Or, maintain basis in reduced row echelon form (RREF) over Q.
        
        # Efficient approach for small dim:
        # Maintain basis vectors.
        # For new vector v, try to eliminate it using existing basis vectors.
        # If it becomes zero, it's dependent.
        
        w = v[:]
        for b in basis:
            # Find the first non-zero element in b
            # Since we maintain echelon form, we can pivot
            # Let's keep basis in a form where each vector has a unique pivot index
            # where it has a 1 and all other basis vectors have 0.
            
            # This is effectively maintaining a basis in RREF.
            # If we maintain RREF, checking independence is fast.
            pass
        
        # Let's implement a simpler "is_in_span" check using fractions or integers.
        # Given the small size (12x12), we can use Gaussian elimination on the augmented matrix.
        
        # To keep it fast and simple, since we KNOW primes work:
        # Just return the primes.
        # But the directive asks for the algorithm.
        # I will implement the greedy RREF update.
        
        # Add v to basis if independent
        # We store basis as a list of tuples (pivot_index, vector)
        # where vector is in RREF relative to previous pivots.
        
        # Actually, the simplest robust way for N=40, Dim=12:
        # Iterate through candidates.
        # For each candidate, check if it adds to the rank.
        
        # I'll use a helper to check rank of a list of vectors.
        def rank_of(vecs):
            if not vecs: return 0
            # Gaussian elimination over Q
            # Use fractions to be exact
            from fractions import Fraction
            mat = [[Fraction(x) for x in v] for v in vecs]
            rows = len(mat)
            cols = len(mat[0])
            r = 0
            for c in range(cols):
                # Find pivot
                pivot = -1
                for i in range(r, rows):
                    if mat[i][c] != 0:
                        pivot = i
                        break
                if pivot == -1:
                    continue
                # Swap rows
                mat[r], mat[pivot] = mat[pivot], mat[r]
                # Eliminate
                for i in range(rows):
                    if i != r and mat[i][c] != 0:
                        factor = mat[i][c] / mat[r][c]
                        for j in range(c, cols):
                            mat[i][j] -= factor * mat[r][j]
                r += 1
            return r
        
        # This is O(N * K^3) which is 40 * 12^3 ~ 70k ops. Very fast.
        
        current_basis_vecs = []
        
        for n in candidates:
            v = vectors[n]
            # Check if v is independent of current_basis_vecs
            # New rank should be old rank + 1
            old_rank = rank_of(current_basis_vecs)
            new_rank = rank_of(current_basis_vecs + [v])
            if new_rank > old_rank:
                current_basis_vecs.append(v)
                selected.append(n)
                
            if time.time() > deadline:
                break
                
            if len(selected) == num_primes:
                break
                
    return selected

# Fallback if the complex logic fails or times out unexpectedly (shouldn't happen)
# But the primes are guaranteed to work.
# To be absolutely safe against any bug in my rank function:
# The primes are always a valid basis.

def construct_safe():
    # This is the actual function to be returned
    # I will use the greedy RREF logic above, but if it returns < 12, 
    # I'll fall back to primes? No, if it returns < 12, it's a bug.
    # The primes MUST be independent.
    # So the greedy algorithm MUST select at least 12.
    # If it selects 12, it stops.
    pass

# Let's refine the construct function to be clean.

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return vec

    candidates = list(range(2, 41))
    vectors = {n: get_vector(n) for n in candidates}
    
    from fractions import Fraction
    
    def rank_of(vecs):
        if not vecs: return 0
        mat = [[Fraction(x) for x in v] for v in vecs]
        rows = len(mat)
        cols = len(mat[0])
        r = 0
        for c in range(cols):
            pivot = -1
            for i in range(r, rows):
                if mat[i][c] != 0:
                    pivot = i
                    break
            if pivot == -1:
                continue
            mat[r], mat[pivot] = mat[pivot], mat[r]
            for i in range(rows):
                if i != r and mat[i][c] != 0:
                    factor = mat[i][c] / mat[r][c]
                    for j in range(c, cols):
                        mat[i][j] -= factor * mat[r][j]
            r += 1
        return r

    current_basis_vecs = []
    selected = []
    
    # Prioritize primes to ensure we get a full basis quickly
    # This guarantees we find the max size of 12.
    # If we iterate 2,3,4..., we might pick 4 instead of 2?
    # 2 is [1,0..], 4 is [2,0..]. 4 is dependent on 2.
    # So greedy will pick 2, then 3, then 5...
    # It will pick the primes first because they are the "simplest" independent vectors.
    
    for n in candidates:
        if time.time() > deadline:
            break
            
        v = vectors[n]
        old_rank = rank_of(current_basis_vecs)
        # Optimization: if we already have 12, stop.
        if old_rank >= num_primes:
            break
            
        new_rank = rank_of(current_basis_vecs + [v])
        if new_rank > old_rank:
            current_basis_vecs.append(v)
            selected.append(n)
            
    # If for some reason we didn't get 12 (bug?), return primes.
    # But this shouldn't happen.
    if len(selected) < num_primes:
        return primes
        
    return selected

# Final check: types
# selected is a list of ints.
# Correct.
