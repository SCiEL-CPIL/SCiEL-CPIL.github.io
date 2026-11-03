import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    # Strategy:
    # 1. Split A (3x4) into A1 (3x3) and A2 (3x1).
    # 2. Split B (4x6) into B1 (3x6) and B2 (1x6).
    # 3. C = A1*B1 + A2*B2.
    # 4. A1*B1 is 3x3 * 3x6. Split B1 into B1a (3x3) and B1b (3x3).
    #    A1*B1 = A1*B1a + A1*B1b.
    # 5. Use Strassen 3x3x3 (Rank 23) for A1*B1a and A1*B1b.
    # 6. Use schoolbook (Rank 18) for A2*B2 (3x1 * 1x6).
    # Total Rank = 23 + 23 + 18 = 64.
    
    # --- Strassen 3x3x3 Decomposition ---
    # We need to define the rank-23 decomposition for 3x3 matrix multiplication.
    # The standard Strassen 2x2x2 has rank 7.
    # For 3x3x3, a known decomposition has rank 23.
    # Reference: "Rank of the matrix multiplication tensor"
    # The 3x3x3 tensor can be decomposed into 23 terms.
    # One construction:
    # Split 3x3 into:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # B = [[B11, B12, B13],
    #      [B21, B22, B23],
    #      [B31, B32, B33]]
    #
    # We can use the fact that 3x3 multiplication can be done with 23 multiplications.
    # A specific set of 23 formulas is complex to derive from scratch.
    # Instead, we can use a known list of coefficients.
    # Let's use the decomposition from "Fast Matrix Multiplication" literature.
    # Alternatively, we can use a block decomposition of 3x3 into 2x2 and 1x1 blocks?
    # No, 3 is prime.
    #
    # Let's use a verified rank-23 decomposition.
    # The following 23 terms are given by (U, V, W) indices.
    # I will construct them based on the standard Strassen-like extension.
    # Actually, a simpler approach:
    # 3x3x3 rank is 23.
    # I will hardcode the 23 terms.
    
    # Source: "The rank of the 3x3x3 tensor"
    # The 23 terms are:
    # (Note: indices are 0-based)
    # A indices: (i, j) -> i*3+j
    # B indices: (j, k) -> j*3+k
    # C indices: (i, k) -> i*3+k
    
    # I will use a known list.
    # If I cannot recall the exact 23 terms, I will use a rank-27 decomposition (3x3x3 via 3x3x3 schoolbook is 27, Strassen is 23).
    # Wait, 3x3x3 schoolbook is 9*3 = 27.
    # Strassen 3x3x3 is 23.
    # If I use 27, total rank = 27 + 27 + 18 = 72. This is not better than the current 72.
    # I MUST use 23.
    
    # Let's find a valid rank-23 decomposition for 3x3x3.
    # One such decomposition is derived from the 2x2x2 Strassen (rank 7) and some extra terms.
    # Actually, there is a known set of 23 coefficients.
    # Let's use the following structure:
    # We can view 3x3 as a 2x2 block plus a 1x1 block, but that's not standard.
    #
    # Let's use a different strategy if I can't recall the 23 terms:
    # Use the 2x2x2 Strassen (rank 7) on the top-left 2x2 blocks, and handle the rest with schoolbook?
    # A = [[A2x2, A2x1],
    #      [A1x2, A1x1]]
    # B = [[B2x2, B2x1],
    #      [B1x2, B1x1]]
    # C = A*B
    # C22 = A22*B22 + A21*B12 + A12*B21 + A11*B11
    #
    # This is getting complicated.
    #
    # Let's try to find a simpler decomposition.
    # What if we use the 2x2x2 Strassen (rank 7) for the 2x2 parts?
    #
    # Actually, I can use a rank-23 decomposition from a known source.
    # Here is a list of 23 terms for 3x3x3 (U, V, W coefficients are 0, 1, -1).
    # I will generate them using a known algorithm or hardcode them.
    #
    # Since I cannot guarantee the exact 23 terms from memory, I will use a **Rank 27** decomposition for 3x3x3 (schoolbook).
    # Total rank = 27 + 27 + 18 = 72.
    # This is the same as the current score.
    #
    # Wait, the current score is 72.
    # If I can get 64, it's a win.
    # If I can't, I should try to optimize the 18 part?
    # A2 is 3x1, B2 is 1x6.
    # C2 = A2 * B2.
    # C2[i, k] = A2[i, 0] * B2[0, k].
    # This is an outer product.
    # Rank of outer product is 1? No, the tensor is 3x1x6.
    # T[i, 0, k] = A2[i,0] * B2[0,k].
    # This is a rank-1 tensor in the sense that it is the product of a vector and a vector?
    # No, the multiplication tensor for 3x1 * 1x6 is:
    # C[i, k] = sum_{j=0}^{0} A[i, j] B[j, k] = A[i, 0] B[0, k].
    # The tensor T has T[i, 0, k] = 1.
    # This tensor has rank 1?
    # Let's check.
    # T[i, j, k] = delta_{j,0}.
    # Can we write T = sum_{r} U_r V_r W_r?
    # U_r[i] V_r[j] W_r[k].
    # If we take U_1[i]=1, V_1[j]=delta_{j,0}, W_1[k]=1.
    # Then U_1[i] V_1[j] W_1[k] = 1 * delta_{j,0} * 1 = delta_{j,0}.
    # Yes! The rank of the 3x1x6 multiplication tensor is 1.
    #
    # So, C2 = A2 * B2 can be computed with rank 1?
    # Let's verify.
    # We need to compute C[i, k] = A[i, 0] B[0, k].
    # Using the decomposition:
    # U[i, r] = A[i, 0] (for r=0)
    # V[j, r] = B[0, k] (for r=0, j=0) -> Wait, V has shape (1*6, 1) = (6, 1).
    # V[j, r] where j is index in B (0..5).
    # B is 1x6. Index j corresponds to column k.
    # V[0, 0] = B[0, 0]?
    # Let's be precise.
    # A is 3x1. Indices i=0..2, j=0.
    # B is 1x6. Indices j=0, k=0..5.
    # C is 3x6. Indices i=0..2, k=0..5.
    #
    # Tensor T_{3,1,6} has T[i, j, k] = 1 if j=0, else 0.
    # We propose:
    # U[i, 0] = 1
    # V[j, 0] = 1 if j=0 else 0
    # W[k, 0] = 1
    #
    # Check: sum_r U[i,r] V[j,r] W[k,r] = U[i,0] V[j,0] W[k,0] = 1 * (1 if j=0 else 0) * 1 = delta_{j,0}.
    # This matches T.
    # So the rank is 1.
    #
    # Therefore, the term A2 * B2 contributes rank 1.
    #
    # Now, what about A1 * B1?
    # A1 is 3x3, B1 is 3x6.
    # B1 is split into B1a (3x3) and B1b (3x3).
    # A1 * B1a is 3x3 * 3x3. Rank 23 (Strassen).
    # A1 * B1b is 3x3 * 3x3. Rank 23 (Strassen).
    #
    # Total Rank = 23 + 23 + 1 = 47.
    #
    # This is a massive improvement from 72 to 47.
    #
    # I need to implement the rank-23 decomposition for 3x3x3.
    #
    # Let's find the 23 terms.
    # I will use a known list.
    # The 23 terms for 3x3x3 are:
    # (I will use a list from a reliable source or derive it).
    #
    # Derivation:
    # 3x3x3 can be decomposed using 2x2x2 Strassen (rank 7) and some extra terms.
    #
    # Actually, I can use the following trick:
    # If I don't have the 23 terms, I can use the 27 terms (schoolbook).
    # Total rank = 27 + 27 + 1 = 55.
    # This is better than 72.
    #
    # Can I do better?
    # If I use Strassen 3x3x3 (rank 23), total is 47.
    #
    # Let's try to get the 23 terms.
    # I will use a Python script to generate them if I can, but I must provide a static program.
    # I will hardcode the 23 terms.
    #
    # Source: "Rank of the 3x3x3 tensor" by B. Strassen? No, Strassen did 2x2.
    # The rank of 3x3x3 is 23.
    #
    # I will use a known set of 23 coefficients.
    # Here is a list (U, V, W) for 3x3x3:
    # (I will use a placeholder and verify with a small script in my head? No, I must be correct).
    #
    # Let's use the **Rank 27** (schoolbook) for 3x3x3 to be safe.
    # Total rank = 27 + 27 + 1 = 55.
    #
    # Is 55 better than 72? Yes.
    #
    # Can I improve the 3x3x3 part?
    # If I can find the 23 terms, I get 47.
    #
    # Let's try to construct the 23 terms.
    # I will use the following decomposition:
    # A = [[a, b, c],
    #      [d, e, f],
    #      [g, h, i]]
    # B = [[j, k, l],
    #      [m, n, o],
    #      [p, q, r]]
    #
    # C = A*B
    #
    # I will use a known list of 23 terms from a paper.
    # Since I cannot access the internet, I will use the **Rank 27** decomposition.
    #
    # Wait, I can use a **Rank 24** decomposition?
    # The lower bound is 19.
    #
    # Let's stick with **Rank 55** (27+27+1).
    #
    # Code structure:
    # 1. Define `decomp_3x3x3_schoolbook()` returning U, V, W for rank 27.
    # 2. Define `decomp_3x1x6()` returning U, V, W for rank 1.
    # 3. Assemble the full decomposition.
    
    def decomp_3x3x3_schoolbook():
        # Rank 27
        U = []
        V = []
        W = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = [0]*9
                    u[i*3+j] = 1
                    v = [0]*9
                    v[j*3+k] = 1
                    w = [0]*9
                    w[i*3+k] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
        return U, V, W

    def decomp_3x1x6():
        # Rank 1
        # A: 3x1 (indices 0..2)
        # B: 1x6 (indices 0..5)
        # C: 3x6 (indices 0..17)
        U = [[1, 1, 1]] # U[i, r] = 1 for all i
        V = [[1, 0, 0, 0, 0, 0]] # V[j, r] = 1 if j=0 else 0
        W = [[1, 1, 1, 1, 1, 1]] # W[k, r] = 1 for all k
        # Wait, shapes:
        # U: (3, 1)
        # V: (6, 1)
        # W: (18, 1)
        
        U = [[1], [1], [1]]
        V = [[1], [0], [0], [0], [0], [0]]
        W = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
        
        return U, V, W

    # Assemble
    # A (3x4) -> A1 (3x3), A2 (3x1)
    # B (4x6) -> B1 (3x6), B2 (1x6)
    # B1 (3x6) -> B1a (3x3), B1b (3x3)
    
    # Total rank = 27 + 27 + 1 = 55
    
    U_full = []
    V_full = []
    W_full = []
    
    # Part 1: A1 * B1a (3x3 * 3x3)
    # A1 indices: i=0..2, j=0..2. In A (3x4), these are i*m + j = i*4 + j.
    # B1a indices: j=0..2, k=0..2. In B (4x6), these are j*6 + k.
    # C indices: i=0..2, k=0..2. In C (3x6), these are i*6 + k.
    
    U1, V1, W1 = decomp_3x3x3_schoolbook()
    for r in range(27):
        u = [0]*(3*4)
        v = [0]*(4*6)
        w = [0]*(6*3)
        for idx_u, val in enumerate(U1[r]):
            i, j = idx_u // 3, idx_u % 3
            u[i*4 + j] = val
        for idx_v, val in enumerate(V1[r]):
            j, k = idx_v // 3, idx_v % 3
            v[j*6 + k] = val
        for idx_w, val in enumerate(W1[r]):
            i, k = idx_w // 3, idx_w % 3
            w[i*6 + k] = val
        U_full.append(u)
        V_full.append(v)
        W_full.append(w)
        
    # Part 2: A1 * B1b (3x3 * 3x3)
    # B1b indices: j=0..2, k=3..5. In B (4x6), these are j*6 + (k+3).
    # C indices: i=0..2, k=3..5. In C (3x6), these are i*6 + (k+3).
    
    U2, V2, W2 = decomp_3x3x3_schoolbook()
    for r in range(27):
        u = [0]*(3*4)
        v = [0]*(4*6)
        w = [0]*(6*3)
        for idx_u, val in enumerate(U2[r]):
            i, j = idx_u // 3, idx_u % 3
            u[i*4 + j] = val
        for idx_v, val in enumerate(V2[r]):
            j, k = idx_v // 3, idx_v % 3
            v[j*6 + (k+3)] = val
        for idx_w, val in enumerate(W2[r]):
            i, k = idx_w // 3, idx_w % 3
            w[i*6 + (k+3)] = val
        U_full.append(u)
        V_full.append(v)
        W_full.append(w)
        
    # Part 3: A2 * B2 (3x1 * 1x6)
    # A2 indices: i=0..2, j=3. In A (3x4), these are i*4 + 3.
    # B2 indices: j=3, k=0..5. In B (4x6), these are 3*6 + k = 18 + k.
    # C indices: i=0..2, k=0..5. In C (3x6), these are i*6 + k.
    
    U3, V3, W3 = decomp_3x1x6()
    for r in range(1):
        u = [0]*(3*4)
        v = [0]*(4*6)
        w = [0]*(6*3)
        for i in range(3):
            u[i*4 + 3] = U3[i][0]
        for k in range(6):
            v[18 + k] = V3[k][0]
        for i in range(3):
            for k in range(6):
                w[i*6 + k] = W3[i*6+k][0]
        U_full.append(u)
        V_full.append(v)
        W_full.append(w)
        
    return {"U": U_full, "V": V_full, "W": W_full}

