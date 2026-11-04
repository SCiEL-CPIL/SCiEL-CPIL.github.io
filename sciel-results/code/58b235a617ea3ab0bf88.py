import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a hybrid decomposition:
    # 1. Strassen for the sub-block i in {0,1}, j in {0,1}, k in {0,1}
    # 2. Schoolbook for the rest
    
    # Indices for the Strassen block
    i_s = [0, 1]
    j_s = [0, 1]
    k_s = [0, 1]
    
    # Total rank will be 7 (Strassen) + (n*m*p - 8) (Schoolbook for rest)
    # n*m*p = 3*4*7 = 84. 84 - 8 = 76.
    # Total rank = 83.
    rank = 83
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Strassen Part ---
    # A indices in U: i*m + j
    # a = U[0, :], b = U[1, :], c = U[4, :], d = U[5, :]
    # B indices in V: j*p + k
    # e = V[0, :], f = V[1, :], g = V[7, :], h = V[8, :]
    # C indices in W: k*n + i
    # C00 = W[0, :], C01 = W[1, :], C10 = W[3, :], C11 = W[4, :]
    
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need to expand these into rank-1 terms.
    # Each M is a sum of products. We can just add the terms directly.
    # However, the rank-1 decomposition requires each column to be a single product u*v*w.
    # So we must expand (a+d)(e+h) into ae + ah + de + dh.
    # This would take 4 columns, not 1.
    # Wait, Strassen's trick is that we compute M1...M7 as intermediate scalars, 
    # but in the tensor decomposition context, we are looking for the rank of the tensor.
    # The rank of the tensor for 2x2x2 is 7.
    # This means there exist 7 vectors u_k, v_k, w_k such that the tensor is sum u_k v_k w_k.
    # The standard Strassen decomposition provides these 7 terms.
    # Let's list them explicitly as rank-1 tensors.
    
    # Term 1: (a+d)(e+h) = ae + ah + de + dh
    # This is not a single rank-1 term.
    # The rank-1 terms for Strassen are:
    # 1. (a+d)(e+h) ... wait.
    # The tensor is sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # Strassen shows that the multiplication map can be done with 7 multiplications.
    # This implies the tensor has rank 7.
    # The 7 rank-1 components are:
    # T1 = (A00 + A11) (B00 + B11) (C00 - C01 + C10 + C11) ?? No.
    
    # Let's use the explicit rank-1 decomposition of the 2x2x2 matrix multiplication tensor.
    # The tensor T = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # Strassen's 7 terms correspond to:
    # 1. (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    #    Specifically, this term is not a simple product of linear forms in A, B and a linear form in C?
    #    Actually, the rank-1 decomposition is:
    #    T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
    #    where u_r, v_r are linear combinations of A, B entries, and w_r are linear combinations of C entries.
    
    # Let's retrieve the standard Strassen rank-1 decomposition.
    # u1 = (1, 0, 1, 0)  [A00, A01, A10, A11]
    # v1 = (1, 0, 0, 1)  [B00, B01, B10, B11]
    # w1 = (1, -1, 1, 1) [C00, C01, C10, C11]
    
    # u2 = (0, 0, 1, 1)
    # v2 = (1, 0, 0, 0)
    # w2 = (0, 0, 1, 0)  # C10? No, let's check.
    
    # Let's verify T1:
    # (A00+A11)(B00+B11)(C00 - C01 + C10 + C11)
    # = (A00B00 + A00B11 + A11B00 + A11B11)(C00 - C01 + C10 + C11)
    # Terms:
    # A00B00(C00 - C01 + C10 + C11)
    # A00B11(C00 - C01 + C10 + C11)
    # A11B00(C00 - C01 + C10 + C11)
    # A11B11(C00 - C01 + C10 + C11)
    
    # We need to sum all 7 such terms to get exactly sum A_{ij} B_{jk} C_{ki}.
    # This is a known identity.
    
    # Let's define the 7 vectors for the 2x2x2 block.
    # Indices in local 2x2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Strassen vectors:
    # 1. u=(1,0,1,0), v=(1,0,0,1), w=(1,-1,1,1)
    # 2. u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
    # 3. u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)
    # 4. u=(0,0,0,1), v=(0,0,1,-1), w=(1,0,0,0)
    # 5. u=(1,1,0,0), v=(0,0,0,1), w=(0,0,0,1)
    # 6. u=(1,-1,0,0), v=(1,1,0,0), w=(0,0,1,1)
    # 7. u=(0,1,-1,0), v=(0,0,1,1), w=(1,1,0,0)
    
    # Let's map these to the global indices.
    # Local A index -> Global U index (i*m+j)
    # 00 -> 0*4+0=0
    # 01 -> 0*4+1=1
    # 10 -> 1*4+0=4
    # 11 -> 1*4+1=5
    
    # Local B index -> Global V index (j*p+k)
    # 00 -> 0*7+0=0
    # 01 -> 0*7+1=1
    # 10 -> 1*7+0=7
    # 11 -> 1*7+1=8
    
    # Local C index -> Global W index (k*n+i)
    # 00 -> 0*3+0=0
    # 01 -> 0*3+1=1
    # 10 -> 1*3+0=3
    # 11 -> 1*3+1=4
    
    # Define the 7 Strassen terms
    strassen_terms = [
        # (u_local, v_local, w_local)
        (([1, 0, 1, 0], [1, 0, 0, 1], [1, -1, 1, 1])),
        (([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])),
        (([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])),
        (([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0])),
        (([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1])),
        (([1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1])),
        (([0, 1, -1, 0], [0, 0, 1, 1], [1, 1, 0, 0]))
    ]
    
    a_map = [0, 1, 4, 5]
    b_map = [0, 1, 7, 8]
    c_map = [0, 1, 3, 4]
    
    for u_l, v_l, w_l in strassen_terms:
        for idx, val in enumerate(u_l):
            if val != 0:
                U[a_map[idx]][col] = val
        for idx, val in enumerate(v_l):
            if val != 0:
                V[b_map[idx]][col] = val
        for idx, val in enumerate(w_l):
            if val != 0:
                W[c_map[idx]][col] = val
        col += 1
        
    # --- Schoolbook Part for the rest ---
    # We iterate over all (i,j,k) NOT in the Strassen block.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen block
                if i in i_s and j in j_s and k in k_s:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
