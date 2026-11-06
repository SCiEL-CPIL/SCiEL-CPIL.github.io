import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    
    # Construct the extended QR(17) code
    # p = 17
    p = 17
    # Quadratic residues modulo 17
    # x^2 mod 17 for x in 1..8
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    # residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Create the base vector v of length 17
    # v[i] = 1 if i in residues else 0
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    for _ in range((p - 1) // 2): # 8 shifts
        basis.append(current)
        # Rotate left by 1 bit within p bits
        # (current << 1) | (current >> (p-1))
        # Mask to p bits
        mask = (1 << p) - 1
        current = ((current << 1) | (current >> (p - 1))) & mask
        
    # Add the all-ones vector
    basis.append((1 << p) - 1)
    
    # Generate all 2^9 = 512 codewords
    code = []
    num_basis = len(basis)
    for mask in range(1 << num_basis):
        word = 0
        for i in range(num_basis):
            if mask & (1 << i):
                word ^= basis[i]
        
        # Compute weight of the 17-bit word
        weight = word.bit_count()
        
        # Determine parity bit
        # If weight is odd, add 1 to make it even (total weight even)
        # If weight is even, add 0
        parity = weight % 2
        
        # Form 18-bit integer
        # The 17 bits are the original, the 18th bit is the parity
        # The task says "18-bit words". Standard representation:
        # Usually, the parity bit is appended.
        # Integer value: (word << 1) | parity
        full_word = (word << 1) | parity
        
        code.append(full_word)
        
    return code
