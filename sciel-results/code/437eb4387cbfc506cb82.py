import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Generator polynomial for the [17,9,5] cyclic code (unextended QR code of length 17).
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Integer representation: bit 8, 5, 4, 3, 0 are set.
    g = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 0)
    
    # Precompute the 9 basis vectors for the [17,9,5] code.
    basis_17 = []
    for i in range(9):
        vec = g << i
        basis_17.append(vec)
        
    # Extend each basis vector to length 18 by adding an overall parity bit.
    basis_18 = []
    for vec in basis_17:
        # Calculate parity of the 17-bit vector
        # Using bit_count is faster and cleaner than bin().count()
        weight = vec.bit_count()
        parity = weight & 1
        ext_vec = vec
        if parity:
            ext_vec |= (1 << 17)
        basis_18.append(ext_vec)
        
    words = []
    
    # Generate all 512 codewords
    for msg in range(512):
        c_val = 0
        m = msg
        for i in range(9):
            if m & 1:
                c_val ^= basis_18[i]
            m >>= 1
            
        words.append(c_val)
        
        if time.time() > deadline:
            return words
            
    return words
