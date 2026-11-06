"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct the Extended Quadratic Residue Code of length 18.
    This is a linear [18, 9, 6] code with 512 codewords.
    """
    p = 17
    k = 9
    n = p + 1  # 18
    
    # Quadratic residues modulo 17
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    Q = {1, 4, 9, 16, 8, 2, 15, 13}
    
    # Base vector of length 17 with 1s at quadratic residues
    base = [0] * p
    for q in Q:
        base[q] = 1
        
    # Generate the 9x17 generator matrix rows (cyclic shifts)
    gen_rows = []
    for shift in range(k):
        row = [0] * p
        for i in range(p):
            row[i] = base[(i + shift) % p]
        gen_rows.append(row)
        
    words = []
    # Iterate over all 2^9 combinations
    for mask in range(1 << k):
        codeword = [0] * p
        for bit_idx in range(k):
            if (mask >> bit_idx) & 1:
                row = gen_rows[bit_idx]
                for i in range(p):
                    codeword[i] ^= row[i]
        
        # Calculate parity bit for extension
        parity = sum(codeword) % 2
        
        # Construct the 18-bit integer
        # The task says "18-bit words". We need to map the 18 bits to an integer.
        # Standard binary encoding: bit 0 is LSB.
        # Let's place the 17 QR bits in positions 0..16 and the parity bit in position 17.
        # Or any consistent mapping. The Hamming distance is invariant under bit permutation.
        # We will use: bits 0..16 from codeword, bit 17 is parity.
        
        val = 0
        for i in range(p):
            if codeword[i]:
                val |= (1 << i)
        if parity:
            val |= (1 << p) # p is 17, so bit 17
            
        words.append(val)
        
    return words
# EVOLVE-BLOCK-END
