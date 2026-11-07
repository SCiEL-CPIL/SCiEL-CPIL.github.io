"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s limit

    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Construct the set of 16 elements
    # Block 2: {2, 4, 16} -> vectors {1, 2, 4}
    # Block 3: {3, 9} -> vectors {1, 2}
    # Block 5: {5, 25} -> vectors {1, 2}
    # Block 7: {7} -> vector {1}
    # Block 11: {11}
    # Block 13: {13}
    # Block 17: {17}
    # Block 19: {19}
    # Block 23: {23}
    # Block 29: {29}
    # Block 31: {31}
    # Block 37: {37}
    
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify validity (optional, but good for safety)
    # Check if all subset products are distinct
    # For k=16, 2^16 = 65536 products.
    # This is fast enough to verify in Python.
    
    def is_valid(S):
        prods = set()
        k = len(S)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= S[i]
            if prod in prods:
                return False
            prods.add(prod)
        return True

    if is_valid(result):
        return result
    else:
        # Fallback to primes if verification fails (should not happen)
        return primes

# EVOLVE-BLOCK-END
