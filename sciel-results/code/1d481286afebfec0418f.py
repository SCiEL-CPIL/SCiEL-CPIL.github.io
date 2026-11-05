"""Program constructing a large binary code of length 17, distance >= 6."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

import numpy as np

N = 17
MIN_D = 6
DEADLINE = 58.0  # safety margin before 60s kill


def _hamming(u, v):
    return (u ^ v).bit_count()


def construct():
    start = time.time()
    rng = random.Random(42)

    # Strategy 1: Construct the [17,9,6] linear code (256 words)
    # The [17,9,6] code is the shortened [18,9,6] extended QR code.
    # We construct the [17,9,5] QR code generator matrix G_17.
    # Then append a parity bit to get [18,9,6].
    # Then shorten by deleting the last bit to get [17,9,6].
    
    p = 17
    QR = set()
    for x in range(1, p):
        if pow(x, (p-1)//2, p) == 1:
            QR.add(x)
    # QR = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Standard QR code generator matrix construction:
    # G[i][j] = 1 if (j+1) is a QR and (i+1) is a QR? No.
    # The parity check matrix H (8x17) has entries H[i][j] = 1 if (j+1)^(2^i) is in QR.
    # Let's build H and then find a basis for the null space (which is G).
    
    H = np.zeros((8, 17), dtype=int)
    for i in range(8):
        for j in range(17):
            # j+1 ranges from 1 to 17.
            # In F_17, 17 is 0. 0 is not a QR.
            # For j=16 (x=17=0), H[i][16] = 0.
            val = (j + 1) % p
            if val == 0:
                H[i][j] = 0
            else:
                # Check if val^(2^i) is a QR
                power = pow(val, (1 << i), p)
                H[i][j] = 1 if power in QR else 0
                
    # Find the null space of H over F_2.
    # H is 8x17. Rank should be 8. Nullity 9.
    # We can find a basis by Gaussian elimination.
    
    # Convert H to a list of rows for easy manipulation
    H_rows = [list(row) for row in H]
    
    # Gaussian Elimination to find basis for null space
    # We want vectors v such that H v = 0.
    # We perform row reduction on H to find pivot columns.
    # The non-pivot columns correspond to free variables.
    
    # We work with the matrix H as integers mod 2.
    mat = [row[:] for row in H_rows]
    rank = 0
    pivot_cols = []
    
    for col in range(17):
        if rank == 8:
            break
        # Find a row with 1 in this column
        pivot_row = -1
        for r in range(rank, 8):
            if mat[r][col] == 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
            
        # Swap rows
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        pivot_cols.append(col)
        
        # Eliminate this column from other rows
        for r in range(8):
            if r != rank and mat[r][col] == 1:
                for c in range(col, 17):
                    mat[r][c] ^= mat[rank][c]
        
        rank += 1
        
    # The pivot columns are pivot_cols.
    # The free columns are the rest.
    # We can construct 9 basis vectors.
    
    free_cols = [c for c in range(17) if c not in pivot_cols]
    
    G = []
    for i, free_col in enumerate(free_cols):
        v = [0] * 17
        v[free_col] = 1
        # For each pivot column, determine the value
        for j, p_col in enumerate(pivot_cols):
            # The equation is sum_{c} mat[j][c] v[c] = 0
            # v[p_col] = sum_{c != p_col} mat[j][c] v[c]
            # Since we are in F_2, sum is XOR.
            val = 0
            for c in range(17):
                if c != p_col:
                    val ^= mat[j][c] * v[c]
            v[p_col] = val
        G.append(v)
        
    # G is a list of 9 vectors of length 17.
    # This is the generator matrix for the [17,9,5] code.
    
    # Verify min distance of this code
    min_d_17 = 17
    # Generate all 256 words
    words_17 = []
    for mask in range(256):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(17):
                    if G[i][j]:
                        w ^= (1 << j)
        words_17.append(w)
        
    # Check min distance (sample check for speed, or full check if fast)
    # 256 words, ~32k pairs. Fast enough.
    for a in range(len(words_17)):
        for b in range(a+1, len(words_17)):
            d = _hamming(words_17[a], words_17[b])
            if d < min_d_17:
                min_d_17 = d
            if min_d_17 < 5:
                break
        if min_d_17 < 5:
            break
            
    if min_d_17 >= 5:
        # Extend to [18,9,6] by adding parity bit
        ext_words = []
        for w in words_17:
            parity = w.bit_count() % 2
            ext_w = w | (parity << 17)
            ext_words.append(ext_w)
            
        # Shorten by deleting the last bit (bit 17)
        # This gives a [17,9,6] code
        code = set()
        for w in ext_words:
            shortened = w & ((1 << 17) - 1)
            code.add(shortened)
            
        # Verify min distance of shortened code
        code_list = list(code)
        min_d_final = 17
        for a in range(len(code_list)):
            for b in range(a+1, len(code_list)):
                d = _hamming(code_list[a], code_list[b])
                if d < min_d_final:
                    min_d_final = d
                if min_d_final < 6:
                    break
            if min_d_final < 6:
                break
                
        if min_d_final >= 6 and len(code_list) > 100:
            # We have a valid large code
            return code_list
    else:
        # Fallback to greedy search
        pass
        
    # Fallback: Greedy Random Search
    code = [0]
    code_set = {0}
    
    # Add random words
    while time.time() - start < DEADLINE - 2:
        candidate = rng.getrandbits(17)
        if candidate in code_set:
            continue
        
        valid = True
        for c in code:
            if _hamming(candidate, c) < MIN_D:
                valid = False
                break
        
        if valid:
            code.append(candidate)
            code_set.add(candidate)
            
    return code

# EVOLVE-BLOCK-END
