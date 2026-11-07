"""
Program to construct a set of integers in [2, 40] with distinct subset products.
Approach: Greedy selection of numbers whose prime factorization exponent vectors
are linearly independent. This maximizes the size of the set.
"""

def construct():
    import math
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        """Return the exponent vector of n with respect to the primes list."""
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return vec
    
    def rank_of_vectors(vectors):
        """Calculate the rank of a list of integer vectors using Gaussian elimination."""
        if not vectors:
            return 0
        # Convert to a list of lists (rows) for easy manipulation
        mat = [v[:] for v in vectors]
        rows = len(mat)
        cols = len(mat[0])
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot_row = None
            for r in range(rank, rows):
                if mat[r][col] != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            
            # Swap pivot row to current rank row
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            
            # Eliminate this column in all other rows
            for r in range(rows):
                if r != rank and mat[r][col] != 0:
                    factor = mat[r][col]
                    for c in range(col, cols):
                        mat[r][c] -= factor * mat[rank][c]
            
            rank += 1
        return rank

    selected_numbers = []
    selected_vectors = []
    
    for n in range(2, 41):
        v = get_vector(n)
        # Check if adding v increases the rank
        current_rank = rank_of_vectors(selected_vectors)
        new_vectors = selected_vectors + [v]
        new_rank = rank_of_vectors(new_vectors)
        
        if new_rank > current_rank:
            selected_numbers.append(n)
            selected_vectors.append(v)
            
            # Optimization: if we reached the max possible rank (num_primes), we can stop
            # because we can't get more independent vectors.
            if new_rank == num_primes:
                break

    return selected_numbers
