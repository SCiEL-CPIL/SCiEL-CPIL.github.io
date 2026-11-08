import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = 3, 4, 8

    # Try rank 80 with a random restart approach
    # First, try to construct a rank-80 decomposition
    
    # Known: rank(3x4x8) is between 78 and 88
    # Let's try a hybrid approach: schoolbook for some parts, Strassen-like for others
    
    # Strategy: Use the fact that we can decompose 3x4x8 as:
    # - 1x4x8 part: 32 terms (schoolbook)
    # - 2x4x8 part: can be done with Strassen-like methods
    
    # For 2x4x8, we can view it as two 2x4x4 blocks
    # Each 2x4x4 can be done in 12 terms (using a 2x2x2 Strassen on 2x2x2 blocks with 4x4 padding)
    # Actually, 2x4x4 has rank at most 2*4*4/2 = 16, but with Strassen we can do better.
    
    # Let me think differently. Use a block decomposition:
    # Split the 4 columns into two 2-column blocks
    # 3x4x8 = 3x2x8 + 3x2x8 (two halves)
    # Each 3x2x8 can be handled
    
    # 3x2x8: rank is at most 3*2*8 = 48, but with Strassen-like:
    # Split 8 into four 2-blocks
    # 3x2x2 has rank 7 (Strassen)
    # So 3x2x8 = 4 * (3x2x2) = 4*7 = 28 terms
    # Total: 28 + 28 = 56 terms for the 2x4x8 part
    # Plus 32 for the 1x4x8 part = 88 terms total
    
    # That's what we already have. To get 80, we need to save 8 terms.
    
    # Alternative: Use a different splitting.
    # What if we use a 2x2x2 Strassen on a 2x2x2 block that overlaps?
    
    # Let's try: handle i=0,1 together using Strassen on 2x4x8
    # 2x4x8: split j into two 2-blocks, k into four 2-blocks
    # For each (jb, kb) pair, we need a 2x2x2 multiplication
    # 2x2x2 has rank 7 (Strassen)
    # So 2 * 4 * 7 = 56 terms for 2x4x8
    # Plus 32 for 1x4x8 = 88 terms. Same as before.
    
    # To save terms, we need a smarter decomposition.
    
    # Let's try a random search for rank 80
    # We'll use a simple greedy/random approach to find a valid decomposition
    
    # Actually, let's just try to verify if rank 80 is achievable
    # by attempting a construction.
    
    # Known result: [3,4,8] tensor rank is 80 (per some literature)
    # Let's try to construct it.
    
    # One approach: use the Winograd scheme
    # 3x4x8 = (2+1) x 4 x (2*4)
    # Use Winograd F_2 transform for the 2x2 blocks
    
    # Let me try a concrete construction with rank 80:
    
    rank = 80
    U = [[0] * rank for _ in range(n * m)]  # 12 x 80
    V = [[0] * rank for _ in range(m * p)]  # 32 x 80
    W = [[0] * rank for _ in range(p * n)]  # 24 x 80
    
    col = 0
    
    # Part 1: Handle i=0 with schoolbook on 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3][col] = 1
            col += 1
    
    # Part 2: Handle i=1,2 with a Strassen-like decomposition
    # 2x4x8: split j into [0,1] and [2,3], k into [0,1],[2,3],[4,5],[6,7]
    # For each of the 2*4=8 blocks, use 7-term Strassen
    # But we can potentially share some terms between adjacent blocks
    
    # Standard 2x2x2 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 2x2x2, the factor matrices are:
    # U factors (7 terms, each is 2x2):
    sU = [
        [1, 0, 0, 1],  # M1: A11+A22
        [0, 0, 1, 1],  # M2: A21+A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A11
        [0, 1, 0, -1], # M7: A12-A22
    ]
    sV = [
        [1, 0, 0, 1],  # B11+B22
        [1, 0, 0, 0],  # B11
        [0, 1, 0, -1], # B21-B22
        [-1, 0, 1, 0], # B12-B11
        [0, 0, 0, 1],  # B22
        [1, 1, 0, 0],  # B11+B12
        [0, 0, 1, 1],  # B21+B22
    ]
    sW = [
        [1, 0, 0, 1],  # C11=M1-M4+M5-M7
        [0, 0, 1, -1], # C12=M5+M7
        [0, 1, 0, 1],  # C21=M2+M4
        [1, 0, 1, 0],  # C22=M1-M2+M3+M6
        [-1, 1, 0, 0], # (for combining)
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    # Actually, the W matrix for 2x2x2 Strassen is:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M5 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    sW = [
        [1, 0, 0, 1],   # M1 contributes to C11 and C22
        [0, 0, 1, -1],  # M2 contributes to C21 and C22
        [0, 1, 0, 1],   # M3 contributes to C22
        [1, 0, 1, 0],   # M4 contributes to C11 and C21
        [-1, 1, 0, 0],  # M5 contributes to C11 and C12
        [0, 0, 0, 1],   # M6 contributes to C22
        [1, 0, 0, 0],   # M7 contributes to C11 and C12
    ]
    # Let me redo this properly. The output C is 2x2:
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M5 + M7
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # So W[t][li] where li encodes (i', k'):
    # li = 0: (0,0), li = 1: (0,1), li = 2: (1,0), li = 3: (1,1)
    sW = [
        [1, 0, 0, 1],   # M1: C00 +1, C11 +1
        [0, 0, 1, -1],  # M2: C10 +1, C11 -1
        [0, 1, 0, 1],   # M3: C01 +1, C11 +1
        [1, 0, 1, 0],   # M4: C00 +1, C10 +1
        [-1, 1, 0, 0],  # M5: C00 -1, C01 +1
        [0, 0, 0, 1],   # M6: C11 +1
        [1, 0, 0, 0],   # M7: C00 +1, C01 -1
    ]
    
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                for li in range(4):
                    if sU[t][li]:
                        li_p = li // 2
                        lj_p = li % 2
                        gi = 1 + li_p
                        gj = js + lj_p
                        U[gi * 4 + gj][col] += sU[t][li]
                for lv in range(4):
                    if sV[t][lv]:
                        lj_p = lv // 2
                        lk_p = lv % 2
                        gj = js + lj_p
                        gk = ks + lk_p
                        V[gj * 8 + gk][col] += sV[t][lv]
                for lw in range(4):
                    if sW[t][lw]:
                        li_p = lw // 2
                        lk_p = lw % 2
                        gi = 1 + li_p
                        gk = ks + lk_p
                        W[gk * 3 + gi][col] += sW[t][lw]
                col += 1
    
    # We've used 32 + 56 = 88 terms, but rank is 80.
    # We need to save 8 terms. Let's try to merge some terms.
    
    # Since col = 88 > 80, we have a problem. Let's return the rank-88 solution.
    if col > rank:
        # Fall back to rank 88
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * 8 + k][col] = 1
                W[k * 3][col] = 1
                col += 1
        
        for jb in range(2):
            js = jb * 2
            for kb in range(4):
                ks = kb * 2
                for t in range(7):
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
                    for li in range(4):
                        if sU[t][li]:
                            li_p = li // 2
                            lj_p = li % 2
                            gi = 1 + li_p
                            gj = js + lj_p
                            U[gi * 4 + gj][col] += sU[t][li]
                    for lv in range(4):
                        if sV[t][lv]:
                            lj_p = lv // 2
                            lk_p = lv % 2
                            gj = js + lj_p
                            gk = ks + lk_p
                            V[gj * 8 + gk][col] += sV[t][lv]
                    for lw in range(4):
                        if sW[t][lw]:
                            li_p = lw // 2
                            lk_p = lw % 2
                            gi = 1 + li_p
                            gk = ks + lk_p
                            W[gk * 3 + gi][col] += sW[t][lw]
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
