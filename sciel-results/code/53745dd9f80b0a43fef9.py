"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = 60.0 - 5.0  # 55 seconds safety margin

    p = 17
    # Compute quadratic residues modulo 17
    Q = set()
    for i in range(1, p):
        Q.add((i * i) % p)
    
    # Construct the base vector u of length 17
    # u_i = 1 if i is in Q, else 0
    u = 0
    for i in range(p):
        if i in Q:
            u |= (1 << i)
            
    # Generate the 9 basis vectors for the [17, 9, 5] QR code
    # by cyclic shifting u
    basis = []
    for k in range(9):
        # Cyclic right shift by k
        shifted = (u >> k) | ((u & ((1 << k) - 1)) << (p - k))
        basis.append(shifted)
        
    # Generate all 2^9 = 512 codewords
    codewords = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
                
        # Extend by a parity bit to get length 18 and distance 6
        parity = word.bit_count() & 1
        extended_word = (word << 1) | parity
        codewords.append(extended_word)
        
    return codewords
# EVOLVE-BLOCK-END
