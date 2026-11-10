def construct():
    from fractions import Fraction
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def get_vector(n):
        vec = []
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            vec.append(Fraction(count))
        return vec

    selected_vectors = []
    selected_numbers = []
    
    for n in range(2, 41):
        v = get_vector(n)
        # Check if v is independent of selected_vectors
        # Create a matrix with selected_vectors and v, check rank
        # Or simply try to add v to the span and see if it increases rank
        
        # Efficient check: Add v to list, compute rank of new list, compare to old rank
        # Since dim is small, just do Gaussian elimination on the list of vectors
        
        new_vectors = selected_vectors + [v]
        
        def rank(vecs):
            if not vecs:
                return 0
            # Gaussian elimination
            mat = [list(v) for v in vecs]
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
                # Swap
                mat[r], mat[pivot] = mat[pivot], mat[r]
                # Eliminate
                for i in range(r + 1, rows):
                    if mat[i][c] != 0:
                        factor = mat[i][c] / mat[r][c]
                        for j in range(c, cols):
                            mat[i][j] -= factor * mat[r][j]
                r += 1
                if r == rows:
                    break
            return r

        old_rank = rank(selected_vectors)
        new_rank = rank(new_vectors)
        
        if new_rank > old_rank:
            selected_vectors = new_vectors
            selected_numbers.append(n)
            
    return selected_numbers
