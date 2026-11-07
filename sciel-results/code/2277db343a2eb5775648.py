import time
from fractions import Fraction

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 10s buffer for 60s limit

    # Candidates: 2 to 40
    candidates = list(range(2, 41))
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Map number to its prime factorization vector (exponents)
    # We use a list of tuples for the vectors
    vectors = []
    for n in candidates:
        exponents = []
        temp_n = n
        for p in primes:
            count = 0
            while temp_n % p == 0:
                temp_n //= p
                count += 1
            exponents.append(count)
        vectors.append(tuple(exponents))
        
    # We want to find the largest linearly independent subset of these vectors.
    # We use Gaussian Elimination over Q.
    # Basis is a list of (pivot_index, vector_tuple, original_index)
    # We maintain the basis in row-echelon form.
    
    basis = []
    
    for orig_idx, vec in enumerate(vectors):
        if time.time() > deadline:
            break
            
        # Copy vector to modify
        current = [Fraction(x) for x in vec]
        
        # Eliminate against existing basis vectors
        # Basis is kept sorted by pivot index
        for i, (b_pivot, b_vec, b_orig) in enumerate(basis):
            if current[b_pivot] != 0:
                factor = current[b_pivot] / b_vec[b_pivot]
                for j in range(b_pivot, num_primes):
                    current[j] -= factor * b_vec[j]
                    
        # Check if dependent (all zeros)
        if all(x == 0 for x in current):
            continue
            
        # Find pivot
        pivot_idx = -1
        for i in range(num_primes):
            if current[i] != 0:
                pivot_idx = i
                break
                
        if pivot_idx == -1:
            continue
            
        # Add to basis, maintaining sorted order by pivot
        # Insert at correct position
        insert_pos = len(basis)
        for i, (b_pivot, _, _) in enumerate(basis):
            if b_pivot > pivot_idx:
                insert_pos = i
                break
                
        basis.insert(insert_pos, (pivot_idx, tuple(current), orig_idx))
        
    # Extract the original indices of the independent vectors
    independent_indices = [b[2] for b in basis]
    
    # Map back to values
    result = [candidates[i] for i in independent_indices]
    
    # Ensure Python int type
    result = [int(x) for x in result]
    
    return result
