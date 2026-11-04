import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Try to build a lower rank decomposition using Strassen for 2x2x2 blocks
    # We split p=6 into two blocks of p=3.
    # For each 3x3x3 block, we try to use Strassen for the 2x2x2 corner (indices 0,1)
    # and schoolbook for the rest.
    
    def build_3x3x3_block():
        # Returns U, V, W for a 3x3x3 block
        # Indices i,j in 0..2, k in 0..2
        terms = []
        
        # Strassen terms for i,j,k in 0,1
        # u, v, w are dictionaries mapping (index1, index2) to coeff
        strassen_terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
        ]
        
        # Verify Strassen terms
        # Tensor T[i,j,k] should be 1 if i,j,k correspond to the term, else 0?
        # No, the tensor is sum of u*v*w.
        # For 2x2x2, T[i,j,k] = 1 for all i,j,k? No.
        # The multiplication tensor T has T[i,j,k] = 1 if the term is A_ij B_jk C_ik.
        # So for the 2x2x2 block, the tensor is 1 for all 8 indices (i,j,k).
        # Let's check if sum of Strassen terms gives 1 for each (i,j,k) in 0,1.
        
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sum_val = 0
                    for u, v, w in strassen_terms:
                        if (i,j) in u and (j,k) in v and (i,k) in w:
                            # Wait, the indices in u,v,w must match i,j,k
                            # u has (i,j), v has (j,k), w has (i,k)
                            # The term contributes u[(i,j)] * v[(j,k)] * w[(i,k)]
                            sum_val += u[(i,j)] * v[(j,k)] * w[(i,k)]
                    if abs(sum_val - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            for u, v, w in strassen_terms:
                terms.append((u, v, w))
            rank_block = 7
        else:
            # Fallback to schoolbook for 2x2x2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        terms.append(( {(i,j):1}, {(j,k):1}, {(i,k):1} ))
            rank_block = 8
            
        # Add schoolbook terms for the rest of 3x3x3
        # Indices where at least one of i,j,k is 2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i < 2 and j < 2 and k < 2:
                        continue # Already covered
                    terms.append(( {(i,j):1}, {(j,k):1}, {(i,k):1} ))
                    
        return terms

    block1_terms = build_3x3x3_block()
    block2_terms = build_3x3x3_block()
    
    total_rank = len(block1_terms) + len(block2_terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: k in 0,1,2
    for u, v, w in block1_terms:
        for (i, j), coeff_u in u.items():
            U[i * m + j][col] += coeff_u
        for (j, k), coeff_v in v.items():
            V[j * p + k][col] += coeff_v
        for (i, k), coeff_w in w.items():
            W[k * n + i][col] += coeff_w
        col += 1
        
    # Block 2: k in 3,4,5 -> map k' = k-3 to 0,1,2
    for u, v, w in block2_terms:
        for (i, j), coeff_u in u.items():
            U[i * m + j][col] += coeff_u
        for (j, k_prime), coeff_v in v.items():
            k = k_prime + 3
            V[j * p + k][col] += coeff_v
        for (i, k_prime), coeff_w in w.items():
            k = k_prime + 3
            W[k * n + i][col] += coeff_w
        col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
