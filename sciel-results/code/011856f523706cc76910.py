"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """Construct the extended quadratic-residue code QR(17) extended to length 18.
    
    This gives a [18, 9, 6] linear code with 512 codewords, which achieves
    A_2(18, 6) = 512.
    """
    p = 17
    # Quadratic residues mod 17: {1, 2, 4, 8, 16, 15, 13, 9}
    qr = set()
    for i in range(1, p):
        if pow(i, (p - 1) // 2, p) == 1:
            qr.add(i)
    # qr = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Build the 9x17 generator matrix for QR(17)
    # First row: position 0 is 1, positions that are QR get 1
    first_row = [0] * p
    first_row[0] = 1
    for r in qr:
        first_row[r] = 1
    
    # Generate 9 cyclic shifts (rows 0..8)
    k = 9
    gen_rows = []
    for shift in range(k):
        row = [first_row[(i + shift) % p] for i in range(p)]
        gen_rows.append(row)
    
    # Verify the generator matrix has rank k (should be 9)
    # Convert to list of integers for XOR operations
    gen_ints = []
    for row in gen_rows:
        val = 0
        for j, bit in enumerate(row):
            if bit:
                val |= (1 << j)
        gen_ints.append(val)
    
    # Enumerate all 2^9 = 512 codewords
    words = []
    for mask in range(1 << k):
        word = 0
        for i in range(k):
            if mask & (1 << i):
                word ^= gen_ints[i]
        # Extend with parity bit (bit 17)
        parity = word.bit_count() % 2
        word_full = word | (parity << p)
        words.append(word_full)
    
    # Verify all pairs have distance >= 6
    # For a linear code, we only need to check nonzero codewords have weight >= 6
    min_weight = 18
    for w in words:
        wt = w.bit_count()
        if wt > 0 and wt < min_weight:
            min_weight = wt
    # All nonzero words should have weight >= 6
    assert min_weight >= 6, f"Min weight {min_weight} < 6"
    
    # Verify distinctness
    assert len(set(words)) == len(words), "Duplicate words found"
    
    return words
# EVOLVE-BLOCK-END
