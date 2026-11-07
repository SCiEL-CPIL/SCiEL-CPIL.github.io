# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    """
    candidates = list(range(2, 41))
    n = len(candidates)
    
    # Precompute prime exponent vectors
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def get_vec(x):
        vec = [0] * num_primes
        for p, i in prime_to_idx.items():
            while x % p == 0:
                vec[i] += 1
                x //= p
        return tuple(vec)
    
    # Get vectors for all candidates
    vecs = [get_vec(x) for x in candidates]
    
    # Use Gaussian elimination over Q to find the maximum linearly independent subset
    # We process candidates in a specific order to maximize the size (though any basis has size = rank)
    # Since we want to maximize the number of selected integers, and the max is the rank,
    # any basis of size Rank is optimal.
    
    # We can improve the set by ensuring we pick the "simplest" or earliest elements?
    # The task just says maximize number. So size is the metric.
    
    # Let's find a basis.
    basis = []  # List of (original_index, vector)
    
    # We iterate through candidates. To make the result deterministic and potentially "nicer",
    # we can sort by value (already is) or by number of prime factors?
    # Standard Gaussian elimination picks the first independent vector as pivot.
    
    for idx, vec in enumerate(vecs):
        w = list(vec)
        is_independent = True
        
        for b_idx, b_vec in basis:
            if w[b_idx] != 0:
                factor = w[b_idx] / b_vec[b_idx]
                for j in range(num_primes):
                    w[j] -= factor * b_vec[j]
        
        # Check if zero
        if all(x == 0 for x in w):
            is_independent = False
        else:
            # Find pivot
            pivot_idx = -1
            for j in range(num_primes):
                if w[j] != 0:
                    pivot_idx = j
                    break
            
            # Normalize
            if w[pivot_idx] != 1:
                inv_pivot = 1 / w[pivot_idx]
                for j in range(num_primes):
                    w[j] *= inv_pivot
            
            basis.append((pivot_idx, tuple(w)))
            # We store the original candidate index to retrieve the value later
            # But we need to know which original candidate this corresponds to.
            # The loop variable `idx` is the index in `candidates` (0..38)
            # So the value is candidates[idx]
            
            # Actually, I need to store the value or index that made it into the basis.
            # Let's keep a separate list.
            
    # The above logic is slightly flawed because I didn't store which original value was added.
    # Let's redo the basis construction to track the original values.
    
    basis_values = []
    basis_vecs = []  # (pivot_idx, vector)
    
    for val, vec in zip(candidates, vecs):
        w = list(vec)
        is_independent = True
        
        for pivot_idx, b_vec in basis_vecs:
            if w[pivot_idx] != 0:
                factor = w[pivot_idx] / b_vec[pivot_idx]
                for j in range(num_primes):
                    w[j] -= factor * b_vec[j]
        
        if all(x == 0 for x in w):
            is_independent = False
        else:
            pivot_idx = -1
            for j in range(num_primes):
                if w[j] != 0:
                    pivot_idx = j
                    break
            
            if w[pivot_idx] != 1:
                inv_pivot = 1 / w[pivot_idx]
                for j in range(num_primes):
                    w[j] *= inv_pivot
            
            basis_vecs.append((pivot_idx, tuple(w)))
            basis_values.append(val)
            
    # basis_values contains the integers that form a linearly independent set.
    # This set guarantees distinct subset products.
    # Size is the rank of the matrix of prime exponents.
    # This is the maximum possible size.
    
    # Verify types and range
    result = [int(x) for x in basis_values]
    
    # Sanity check: all in 2..40
    # They are from candidates, so this is guaranteed.
    
    return result
# EVOLVE-BLOCK-END
