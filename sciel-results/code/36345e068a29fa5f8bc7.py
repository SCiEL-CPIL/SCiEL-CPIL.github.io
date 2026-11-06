"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    
    # Attempt to construct the Extended Quadratic Residue Code QR(17)
    # This is a linear code [18, 9, 6] with 512 codewords.
    
    # Generator polynomials for QR(17) in GF(2)
    # The roots are the non-residues mod 17.
    # Two possible generator polynomials (factors of x^17+1 of degree 8).
    g1 = 0b100111001 # x^8 + x^5 + x^4 + x^3 + 1
    g2 = 0b101101101 # x^8 + x^7 + x^6 + x^4 + 1
    
    def generate_qr_code(g_poly):
        codes = []
        for msg in range(512):
            cw = 0
            for i in range(9):
                if (msg >> i) & 1:
                    cw ^= (g_poly << i)
            # Extend with parity bit
            parity = bin(cw).count('1') % 2
            cw_ext = cw | (parity << 17)
            codes.append(cw_ext)
        return codes
    
    # Try first generator
    code1 = generate_qr_code(g1)
    min_d1 = N
    for cw in code1:
        if cw != 0:
            w = cw.bit_count()
            if w < min_d1:
                min_d1 = w
                
    if min_d1 >= MIN_DIST:
        return code1
        
    # Try second generator
    code2 = generate_qr_code(g2)
    min_d2 = N
    for cw in code2:
        if cw != 0:
            w = cw.bit_count()
            if w < min_d2:
                min_d2 = w
                
    if min_d2 >= MIN_DIST:
        return code2
        
    # Fallback: Reed-Muller RM(1,4) padded to 18 bits
    # RM(1,4) is [16, 5, 8] code.
    basis = [
        0b1111111111111111,
        0b1010101010101010,
        0b0110011001100110,
        0b0011001100110011,
        0b0001000100010001
    ]
    
    rm_codes = []
    for m in range(32):
        cw = 0
        for i in range(5):
            if (m >> i) & 1:
                cw ^= basis[i]
        # Pad to 18 bits (append 2 zeros)
        rm_codes.append(cw)
        
    # Verify RM code
    min_d_rm = N
    for cw in rm_codes:
        if cw != 0:
            w = cw.bit_count()
            if w < min_d_rm:
                min_d_rm = w
                
    if min_d_rm >= MIN_DIST:
        return rm_codes
        
    return [0]

# EVOLVE-BLOCK-END
