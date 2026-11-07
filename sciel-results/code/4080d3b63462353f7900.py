"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 40
    numbers = list(range(2, n + 1))
    
    # 1. Find primes up to n
    primes = []
    for num in numbers:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
            
    # 2. Compute exponent vectors for each number
    # Vector is a list of exponents for each prime in 'primes'
    vectors = []
    for num in numbers:
        vec = [0] * len(primes)
        temp = num
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        vectors.append(vec)
        
    # 3. Find a maximal linearly independent subset
    # We use a greedy approach with rank checking.
    # Since the dimension is small (12), we can maintain the basis in row-echelon form.
    # Or simpler: just check if the new vector is in the span of the selected ones.
    
    selected_indices = []
    basis = [] # List of vectors in the selected set, kept in a form that allows easy span check?
    # Actually, just checking rank of [selected_vectors + [new_vector]] vs [selected_vectors] is easy.
    
    def rank_of_matrix(mat):
        # mat is list of lists of Fractions
        if not mat:
            return 0
        rows = len(mat)
        cols = len(mat[0])
        # Copy matrix
        m = [row[:] for row in mat]
        r = 0
        for c in range(cols):
            # Find pivot
            pivot_r = -1
            for i in range(r, rows):
                if m[i][c] != 0:
                    pivot_r = i
                    break
            if pivot_r == -1:
                continue
            # Swap
            m[r], m[pivot_r] = m[pivot_r], m[r]
            # Eliminate
            pivot_val = m[r][c]
            for i in range(r, rows):
                if i != r and m[i][c] != 0:
                    factor = m[i][c] / pivot_val
                    for j in range(c, cols):
                        m[i][j] -= factor * m[r][j]
            r += 1
        return r

    # Greedy selection
    for i in range(len(numbers)):
        if time.time() > deadline:
            break
        # Convert current vector to Fractions for exact arithmetic
        new_vec = [Fraction(x) for x in vectors[i]]
        
        # Current selected vectors
        current_vecs = [[Fraction(x) for x in vectors[j]] for j in selected_indices]
        
        rank_before = rank_of_matrix(current_vecs) if current_vecs else 0
        
        # Check if adding new_vec increases rank
        # We can just compute rank of combined matrix
        combined = current_vecs + [new_vec]
        rank_after = rank_of_matrix(combined)
        
        if rank_after > rank_before:
            selected_indices.append(i)
            
    # The selected_indices corresponds to the numbers
    result = [numbers[i] for i in selected_indices]
    
    return result

# EVOLVE-BLOCK-END
