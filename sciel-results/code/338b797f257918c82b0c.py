"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 90.0  # 90s wall-clock limit, use 85s to be safe
    
    # Quadratic residues modulo 17
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    # R = {1, 2, 4, 8, 9, 13, 15, 16}
    R = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Construct the 17x17 circulant matrix
    # Row i has 1s at positions (j - i) % 17 in R
    # We will store rows as integers for efficiency
    rows = []
    for i in range(17):
        word = 0
        for j in range(17):
            if (j - i) % 17 in R:
                word |= (1 << j)
        rows.append(word)
        
    # Perform Gaussian elimination to find a basis for the row space
    # We work with integers representing bit vectors
    basis = []
    # Sort rows by highest set bit to make elimination efficient? 
    # Or just iterate.
    # Standard Gaussian elimination over GF(2)
    # We want to find the rank and a basis.
    # Since we expect rank 9, we can just reduce the 17 rows.
    
    # To make it faster, we can just pick independent rows.
    # But Gaussian elimination is robust.
    # Let's implement a simple reduction.
    
    # We'll maintain a list of pivot rows.
    # pivot_rows[i] is a row with the highest set bit at position i.
    # Since length is 17, positions are 0..16.
    pivot_rows = [None] * 17
    
    for r in rows:
        if r == 0:
            continue
        # Find the highest set bit
        # Python int bit_length gives position of highest bit + 1
        hb = r.bit_length() - 1
        if pivot_rows[hb] is None:
            pivot_rows[hb] = r
        else:
            r = r ^ pivot_rows[hb]
            # Continue reducing if r is not 0
            while r != 0:
                hb = r.bit_length() - 1
                if pivot_rows[hb] is None:
                    pivot_rows[hb] = r
                    r = 0
                else:
                    r = r ^ pivot_rows[hb]
                    
    # Collect non-zero pivot rows
    basis = [p for p in pivot_rows if p is not None]
    
    # The dimension is len(basis)
    # Expected 9
    k = len(basis)
    
    # Generate all 2^k linear combinations
    words = []
    # Iterate through all subsets of the basis
    for mask in range(1 << k):
        word = 0
        for i in range(k):
            if (mask >> i) & 1:
                word ^= basis[i]
        
        # Extend by parity bit
        # Parity bit is the sum of bits in word mod 2
        # word is 17 bits.
        parity = word.bit_count() % 2
        # Append parity bit at position 17
        word_18 = word | (parity << 17)
        words.append(word_18)
        
    # Verify distinctness (should be distinct by construction of basis)
    # Verify distance (should be >= 6)
    # The construction guarantees this for EQRC.
    
    return words

# EVOLVE-BLOCK-END
