import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    
    # Deadline guard
    start_time = time.time()
    deadline = 90.0
    
    # Construct the extended QR(17) code
    p = 17
    
    # Quadratic residues modulo 17
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Create the base vector v of length 17
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
            
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    mask_p = (1 << p) - 1
    for _ in range((p - 1) // 2):
        basis.append(current)
        current = ((current << 1) | (current >> (p - 1))) & mask_p
        
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
                
        weight = word.bit_count()
        parity = weight % 2
        full_word = (word << 1) | parity
        code.append(full_word)
        
    return code
