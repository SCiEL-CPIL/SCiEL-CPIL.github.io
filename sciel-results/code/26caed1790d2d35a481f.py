def construct():
    # Quadratic residues mod 23
    Q = [1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18]
    
    # Generate base row for QR code [23, 12, 7]
    # Length 23
    base_row = 0
    base_row |= 1 << 0
    for q in Q:
        base_row |= 1 << q
        
    # Generate 12 generator rows by cyclic shifts
    gen_rows = []
    for i in range(12):
        # Cyclic shift right by i
        # (x >> i) | ((x << (23 - i)) & ((1 << 23) - 1))
        if i == 0:
            r = base_row
        else:
            r = (base_row >> i) | ((base_row << (23 - i)) & ((1 << 23) - 1))
        gen_rows.append(r)
        
    # Extend to [24, 12, 8] by adding parity bit
    gen_rows_ext = []
    for r in gen_rows:
        # Count bits
        w = r.bit_count()
        parity = w % 2
        r_ext = r | (parity << 23)
        gen_rows_ext.append(r_ext)
        
    # Generate all 4096 codewords
    words_24 = []
    for m in range(4096):
        w = 0
        for i in range(12):
            if (m >> i) & 1:
                w ^= gen_rows_ext[i]
        words_24.append(w)
        
    # Puncture twice (remove bits 22 and 23)
    # We keep bits 0..21
    mask_22 = (1 << 22) - 1
    words_22 = [w & mask_22 for w in words_24]
    
    # Shorten 4 times (remove bit 0, then new bit 0, etc.)
    # This effectively removes bits 0, 1, 2, 3 from the 22-bit word
    # And keeps only words where those bits were 0
    
    current_words = words_22
    
    for _ in range(4):
        next_words = []
        for w in current_words:
            if (w & 1) == 0:
                # Remove bit 0 by shifting right
                next_words.append(w >> 1)
        current_words = next_words
        
    # Result should be 256 words of length 18
    return current_words
