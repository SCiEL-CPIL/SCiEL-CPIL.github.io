"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Strategy: Use Strassen's algorithm for the 4x4x4 block (rank 49) and schoolbook for the 4x4x3 block (rank 48).
    Total rank = 49 + 48 = 97.
    
    The tensor T for matrix multiplication M_{n,m,p} is defined by:
    T[i,j,k] = 1 if the indices align with the multiplication pattern.
    Specifically, the bilinear form is <C, A B> = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    The tensor is T = sum_{i,j,k} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    
    For the 4x4x7 case, we split the output dimension k into two parts:
    - k in [0, 4): This is the 4x4x4 multiplication tensor.
    - k in [4, 7): This is the 4x4x3 multiplication tensor.
    
    For the 4x4x4 block, we use Strassen's algorithm.
    For the 4x4x3 block, we use the schoolbook decomposition.
    
    The rank of the 4x4x4 tensor using Strassen is 49 (7^2).
    The rank of the 4x4x3 tensor using schoolbook is 48 (4*4*3).
    Total rank = 49 + 48 = 97.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Start with a deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds
    
    # Strassen's algorithm for 2x2x2 matrix multiplication
    # The 7 terms are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # The Strassen decomposition for 2x2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The rank-1 terms for Strassen 2x2x2:
    # Each term is (u, v, w) where:
    # u is a vector of length 4 (for the 2x2 A matrix)
    # v is a vector of length 4 (for the 2x2 B matrix)
    # w is a vector of length 4 (for the 2x2 C matrix)
    
    # The 7 Strassen terms:
    strassen_2x2_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]),
    ]
    
    # For 4x4x4, we can view it as a 2x2 block matrix where each block is 2x2.
    # The 4x4 matrix A is:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where each Aij is a 2x2 matrix.
    
    # The 4x4 matrix B is:
    # B = [[B00, B01],
    #      [B10, B11]]
    # where each Bjk is a 2x2 matrix.
    
    # The 4x4 matrix C is:
    # C = [[C00, C01],
    #      [C10, C11]]
    # where each Ckl is a 2x2 matrix.
    
    # The product C = A B is computed block-wise:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each block product (e.g., A00 B00) is a 2x2 matrix product, which can be computed using Strassen.
    # So we need to apply Strassen to each of the 4 block products.
    # But the blocks are not independent; they share the same A and B blocks.
    
    # Actually, the 4x4x4 tensor can be decomposed using the Kronecker product of the 2x2x2 tensor with itself.
    # The rank of the Kronecker product is the product of the ranks.
    # So the rank of the 4x4x4 tensor is at most 7 * 7 = 49.
    
    # To construct the rank-49 decomposition, we take all pairs of Strassen terms.
    # For each pair (s, t) where s and t are Strassen terms, we create a rank-1 term for the 4x4x4 tensor.
    
    # The indices for the 4x4 matrices:
    # A is 4x4, with indices (i, j) where i, j in [0, 4).
    # We can write i = 2*i1 + i2, j = 2*j1 + j2, where i1, j1, i2, j2 in [0, 2).
    # So A[i, j] = A[i1, i2, j1, j2].
    
    # Similarly for B and C.
    
    # The Strassen terms for the 2x2x2 tensor are (u, v, w) where u, v, w are length-4 vectors.
    # For the 4x4x4 tensor, the rank-1 terms are (u \otimes u', v \otimes v', w \otimes w')
    # where (u, v, w) and (u', v', w') are Strassen terms.
    
    # Let's construct the rank-49 decomposition for the 4x4x4 tensor.
    strassen_4x4_terms = []
    for s in strassen_2x2_terms:
        for t in strassen_2x2_terms:
            u_s, v_s, w_s = s
            u_t, v_t, w_t = t
            
            # The Kronecker product of u_s and u_t
            u_4x4 = []
            for a in u_s:
                for b in u_t:
                    u_4x4.append(a * b)
            
            v_4x4 = []
            for a in v_s:
                for b in v_t:
                    v_4x4.append(a * b)
            
            w_4x4 = []
            for a in w_s:
                for b in w_t:
                    w_4x4.append(a * b)
            
            strassen_4x4_terms.append((u_4x4, v_4x4, w_4x4))
    
    # Now, for the 4x4x3 tensor (k in [4, 7)), we use the schoolbook decomposition.
    # The schoolbook decomposition uses 4*4*3 = 48 terms.
    # For each (i, j, k) where i, j in [0, 4) and k in [4, 7), we create a rank-1 term.
    
    schoolbook_4x4x3_terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u = [0] * 16
                v = [0] * 28
                w = [0] * 28
                
                u[i * 4 + j] = 1
                v[j * 7 + k] = 1
                w[k * 4 + i] = 1
                
                schoolbook_4x4x3_terms.append((u, v, w))
    
    # Now, we need to combine the 4x4x4 and 4x4x3 decompositions.
    # The 4x4x4 decomposition has 49 terms, each with u, v, w of length 16, 28, 28.
    # The 4x4x3 decomposition has 48 terms, each with u, v, w of length 16, 28, 28.
    
    # Total rank = 49 + 48 = 97.
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Fill in the Strassen 4x4x4 terms
    for r, (u, v, w) in enumerate(strassen_4x4_terms):
        for i in range(16):
            if u[i] != 0:
                U[i][r] = u[i]
        for j in range(28):
            if v[j] != 0:
                V[j][r] = v[j]
        for k in range(28):
            if w[k] != 0:
                W[k][r] = w[k]
    
    # Fill in the schoolbook 4x4x3 terms
    for r, (u, v, w) in enumerate(schoolbook_4x4x3_terms):
        r_offset = 49
        for i in range(16):
            if u[i] != 0:
                U[i][r_offset + r] = u[i]
        for j in range(28):
            if v[j] != 0:
                V[j][r_offset + r] = v[j]
        for k in range(28):
            if w[k] != 0:
                W[k][r_offset + r] = w[k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
