"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition using Strassen for 2x2x2 sub-blocks
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (A22-A11)(B11-B22)
    # M2 = (A11+A22)B22
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B11
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12+A22)(B21-B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the 7 tensors for a 2x2x2 block.
    # Indices in local 2x2x2 block: i,j,k in {0,1}
    # Global indices: i0, j0, k0
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A: (i,j), B: (j,k), C: (k,i)
        # M1: A=(1,1)-(0,0), B=(0,0)-(1,1), C=(0,0)+(1,1)?? No.
        # C11 gets M1. C11 is (0,0).
        # C22 gets M1. C22 is (1,1).
        # So C coeffs: (0,0): 1, (1,1): 1.
        # A coeffs: (1,1): 1, (0,0): -1.
        # B coeffs: (0,0): 1, (1,1): -1.
        {
            'A': { (1,1): 1, (0,0): -1 },
            'B': { (0,0): 1, (1,1): -1 },
            'C': { (0,0): 1, (1,1): 1 }
        },
        # M2: A=(0,0)+(1,1), B=(1,1), C=(1,0)+(0,1)??
        # C21 = M2. C21 is (1,0).
        # C22 = -M2. C22 is (1,1).
        # So C coeffs: (1,0): 1, (1,1): -1.
        # A coeffs: (0,0): 1, (1,1): 1.
        # B coeffs: (1,1): 1.
        {
            'A': { (0,0): 1, (1,1): 1 },
            'B': { (1,1): 1 },
            'C': { (1,0): 1, (1,1): -1 }
        },
        # M3: A=(0,0), B=(0,1)-(1,1), C=(0,1)+(1,0)??
        # C12 = M3. C12 is (0,1).
        # C22 = M3. C22 is (1,1).
        # So C coeffs: (0,1): 1, (1,1): 1.
        # A coeffs: (0,0): 1.
        # B coeffs: (0,1): 1, (1,1): -1.
        {
            'A': { (0,0): 1 },
            'B': { (0,1): 1, (1,1): -1 },
            'C': { (0,1): 1, (1,1): 1 }
        },
        # M4: A=(1,1), B=(1,0)-(0,0), C=(0,0)+(1,0)??
        # C11 = M4. C11 is (0,0).
        # C21 = M4. C21 is (1,0).
        # So C coeffs: (0,0): 1, (1,0): 1.
        # A coeffs: (1,1): 1.
        # B coeffs: (1,0): 1, (0,0): -1.
        {
            'A': { (1,1): 1 },
            'B': { (1,0): 1, (0,0): -1 },
            'C': { (0,0): 1, (1,0): 1 }
        },
        # M5: A=(0,0)+(0,1), B=(0,0), C=(0,0)-(0,1)??
        # C11 = -M5. C11 is (0,0).
        # C12 = M5. C12 is (0,1).
        # So C coeffs: (0,0): -1, (0,1): 1.
        # A coeffs: (0,0): 1, (0,1): 1.
        # B coeffs: (0,0): 1.
        {
            'A': { (0,0): 1, (0,1): 1 },
            'B': { (0,0): 1 },
            'C': { (0,0): -1, (0,1): 1 }
        },
        # M6: A=(1,0)-(0,0), B=(0,0)+(0,1), C=(1,1)??
        # C22 = M6. C22 is (1,1).
        # So C coeffs: (1,1): 1.
        # A coeffs: (1,0): 1, (0,0): -1.
        # B coeffs: (0,0): 1, (0,1): 1.
        {
            'A': { (1,0): 1, (0,0): -1 },
            'B': { (0,0): 1, (0,1): 1 },
            'C': { (1,1): 1 }
        },
        # M7: A=(0,1)+(1,1), B=(1,0)-(1,1), C=(0,0)??
        # C11 = M7. C11 is (0,0).
        # So C coeffs: (0,0): 1.
        # A coeffs: (0,1): 1, (1,1): 1.
        # B coeffs: (1,0): 1, (1,1): -1.
        {
            'A': { (0,1): 1, (1,1): 1 },
            'B': { (1,0): 1, (1,1): -1 },
            'C': { (0,0): 1 }
        }
    ]
    
    # Define disjoint 2x2x2 blocks
    # i in {0,1}
    # j in {0,1}, {2,3}
    # k in {0,1}, {2,3}, {4,5}
    
    blocks = []
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            blocks.append((0, j_start, k_start)) # i_start=0
    
    used_indices = set()
    
    for i_start, j_start, k_start in blocks:
        for term in strassen_terms:
            # Add U
            for (li, lj), val in term['A'].items():
                gi = i_start + li
                gj = j_start + lj
                U[gi * m + gj][col] = val
            # Add V
            for (lj, lk), val in term['B'].items():
                gj = j_start + lj
                gk = k_start + lk
                V[gj * p + gk][col] = val
            # Add W
            for (lk, li), val in term['C'].items():
                gk = k_start + lk
                gi = i_start + li
                W[gk * n + gi][col] = val
            col += 1
        
        # Mark these 8 indices as used
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    used_indices.add((i_start + li, j_start + lj, k_start + lk))

    # Add schoolbook for remaining indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used_indices:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
