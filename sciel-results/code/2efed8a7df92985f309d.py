"""Program for kissing number in dimension 11 using Leech lattice construction."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct 165 vectors in R^11 forming a kissing configuration.
    
    Approach:
    1. Include all 22 vectors of the form (±2, 0, ..., 0).
    2. Find a maximal set of 143 weight-4 vectors (four ±1's) such that
       all pairwise dot products are <= 2.
    3. Use a fast randomized greedy algorithm with bitset operations to find
       the maximal independent set in the conflict graph of weight-4 vectors.
    """
    start_time = time.time()
    deadline = start_time + 115

    # Generate weight-1 vectors (norm^2 = 4)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    v1_arr = np.array(v1_list, dtype=np.int64)

    # Generate weight-4 vectors (norm^2 = 4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    v4_arr = np.array(v4_list, dtype=np.int64)
    N4 = len(v4_arr) # 5280

    # Compute conflict matrix for weight-4 vectors
    # Conflict if dot product > 2
    dots4 = v4_arr @ v4_arr.T
    conflicts4 = dots4 > 2
    np.fill_diagonal(conflicts4, False)

    # Convert conflict matrix to list of Python integers (bitsets)
    # This allows very fast bitwise operations for greedy selection
    conflict_bits = []
    # Pack the boolean rows into bytes, then to int
    # np.packbits packs 8 booleans into 1 byte
    packed_conflicts = np.packbits(conflicts4, axis=1)
    for i in range(N4):
        # Convert the byte array to a Python int
        # We need to handle the byte order correctly
        # packbits packs MSB first? No, default is MSB first in the byte?
        # Actually, let's just use int.from_bytes
        b = packed_conflicts[i]
        val = int.from_bytes(b, byteorder='big')
        conflict_bits.append(val)

    # Randomized Greedy with Bitsets
    rng = np.random.default_rng(42)
    best_size = 0
    best_indices = []
    
    # Number of restarts. Each restart is very fast.
    # 5280 bits is ~660 bytes. Bitwise AND is very fast.
    n_restarts = 2000
    if time.time() > deadline - 10:
        n_restarts = 100

    for _ in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        # Shuffle indices
        order = rng.permutation(N4)
        
        # Initialize available mask to all 1s
        # (1 << N4) - 1 creates a mask with N4 bits set
        avail = (1 << N4) - 1
        selected = []
        
        for idx in order:
            # Check if idx is still available
            if (avail >> idx) & 1:
                selected.append(idx)
                # Remove idx and its conflicts from avail
                avail &= ~conflict_bits[idx]
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected

    # Combine with weight-1 vectors
    # The weight-1 vectors are all compatible with each other and with any weight-4 vector.
    # Proof:
    # w1_i = 2 * e_i. w1_j = 2 * e_j.
    # w1_i . w1_j = 4 if i=j (same vector, not in set), 0 if i!=j.
    # w1_i . w4 = +/- 2 if i in support(w4), 0 otherwise.
    # All <= 2.
    
    result_v4 = v4_arr[best_indices]
    final_vectors = np.vstack([v1_arr, result_v4])
    
    return final_vectors

# EVOLVE-BLOCK-END
