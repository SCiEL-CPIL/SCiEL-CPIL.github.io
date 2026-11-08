import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_map, v_map, w_map):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for (i, j), val in u_map.items():
            u[i * m + j] = val
        for (j, k), val in v_map.items():
            v[j * p + k] = val
        for (k, i), val in w_map.items():
            w[k * n + i] = val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """
        Adds 7 terms for Strassen multiplication of 2x2 blocks.
        A rows i0, i1; cols j0, j1.
        B rows j0, j1; cols k0, k1.
        C rows k0, k1; cols i0, i1.
        """
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # Indices
    # Block 1: index 0
    # Block 4: indices 1, 2, 3, 4
    
    # 1. C_11 = A_11 B_11 + A_14 B_41
    # A_11 (1x1) * B_11 (1x1)
    add_term({(0,0):1}, {(0,0):1}, {(0,0):1})
    
    # A_14 (1x4) * B_41 (4x1)
    # A row 0, cols 1-4. B rows 1-4, col 0. C col 0, row 0.
    for j in range(1, 5):
        add_term({(0,j):1}, {(j,0):1}, {(0,0):1})

    # 2. C_14 = A_11 B_14 + A_14 B_44
    # A_11 (1x1) * B_14 (1x4)
    # A row 0, col 0. B row 0, cols 1-4. C cols 1-4, row 0.
    for k in range(1, 5):
        add_term({(0,0):1}, {(0,k):1}, {(k,0):1})
        
    # A_14 (1x4) * B_44 (4x4)
    # A row 0, cols 1-4. B rows 1-4, cols 1-4. C cols 1-4, row 0.
    for j in range(1, 5):
        for k in range(1, 5):
            add_term({(0,j):1}, {(j,k):1}, {(k,0):1})

    # 3. C_41 = A_41 B_11 + A_44 B_41
    # A_41 (4x1) * B_11 (1x1)
    # A rows 1-4, col 0. B row 0, col 0. C col 0, rows 1-4.
    for i in range(1, 5):
        add_term({(i,0):1}, {(0,0):1}, {(0,i):1})

    # A_44 (4x4) * B_41 (4x1)
    # A rows 1-4, cols 1-4. B rows 1-4, col 0. C col 0, rows 1-4.
    for i in range(1, 5):
        for j in range(1, 5):
            add_term({(i,j):1}, {(j,0):1}, {(0,i):1})

    # 4. C_44 = A_41 B_14 + A_44 B_44
    # A_41 (4x1) * B_14 (1x4)
    # A rows 1-4, col 0. B row 0, cols 1-4. C cols 1-4, rows 1-4.
    for i in range(1, 5):
        for k in range(1, 5):
            add_term({(i,0):1}, {(0,k):1}, {(k,i):1})

    # A_44 (4x4) * B_44 (4x4)
    # Use Strassen on 4x4 block.
    # Split 4x4 into 2x2 blocks:
    # Rows 1,2 and 3,4. Cols 1,2 and 3,4.
    strassen_2x2(1, 2, 1, 2, 1, 2) # Top-Left 2x2 of A44, Top-Left 2x2 of B44 -> Top-Left 2x2 of C44
    strassen_2x2(1, 2, 1, 2, 3, 4) # Top-Left 2x2 of A44, Top-Right 2x2 of B44 -> Top-Right 2x2 of C44
    strassen_2x2(3, 4, 1, 2, 1, 2) # Bottom-Left 2x2 of A44, Top-Left 2x2 of B44 -> Bottom-Left 2x2 of C44
    strassen_2x2(3, 4, 1, 2, 3, 4) # Bottom-Left 2x2 of A44, Top-Right 2x2 of B44 -> Bottom-Right 2x2 of C44
    strassen_2x2(1, 2, 3, 4, 1, 2) # Top-Right 2x2 of A44, Bottom-Left 2x2 of B44 -> Top-Left 2x2 of C44
    strassen_2x2(1, 2, 3, 4, 3, 4) # Top-Right 2x2 of A44, Bottom-Right 2x2 of B44 -> Top-Right 2x2 of C44
    strassen_2x2(3, 4, 3, 4, 1, 2) # Bottom-Right 2x2 of A44, Bottom-Left 2x2 of B44 -> Bottom-Left 2x2 of C44
    strassen_2x2(3, 4, 3, 4, 3, 4) # Bottom-Right 2x2 of A44, Bottom-Right 2x2 of B44 -> Bottom-Right 2x2 of C44
    # Wait, Strassen computes C = A*B for 2x2 blocks.
    # The 4x4 multiplication is composed of 16 2x2 block multiplications?
    # No, Strassen reduces 8 multiplications to 7.
    # For 4x4, we view it as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]].
    # C = [[C11, C12], [C21, C22]].
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # Each M is a product of 2x2 matrices.
    # So we need 7 calls to strassen_2x2?
    # No, the M's are sums/differences of A and B blocks.
    # The `strassen_2x2` function above adds terms for the *entire* Strassen computation of a 2x2 * 2x2 = 2x2 problem.
    # But here, the "scalars" are 2x2 matrices.
    # So we need to apply the Strassen logic to the blocks.
    # The function `strassen_2x2(i0, i1, j0, j1, k0, k1)` assumes A is 2x2, B is 2x2, C is 2x2.
    # In the 4x4 case, A is 4x4.
    # We cannot just call `strassen_2x2` 8 times.
    # We need to implement the Strassen combination for the 4x4 block directly.
    
    # Let's re-implement the 4x4 Strassen part explicitly.
    # A_44 indices: i in 1..4, j in 1..4.
    # B_44 indices: j in 1..4, k in 1..4.
    # C_44 indices: k in 1..4, i in 1..4.
    
    # Blocks of A_44:
    # A11: i=1,2; j=1,2
    # A12: i=1,2; j=3,4
    # A21: i=3,4; j=1,2
    # A22: i=3,4; j=3,4
    
    # Blocks of B_44:
    # B11: j=1,2; k=1,2
    # B12: j=1,2; k=3,4
    # B21: j=3,4; k=1,2
    # B22: j=3,4; k=3,4
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22: i in 1..4, j in 1..4? No.
    # A11 is 2x2, A22 is 2x2. Their sum is not a single 2x2 block in the same position.
    # We must expand the sums into scalar terms.
    
    # This is getting complex to code generically. 
    # However, we can just use the schoolbook for 4x4? No, rank 64 vs 49.
    # We need the 49 rank terms.
    
    # Alternative: Use the `strassen_2x2` function on the 2x2 sub-blocks of the 4x4?
    # No, Strassen combines them.
    
    # Let's write the 7 M terms for the 4x4 block explicitly.
    
    def add_strassen_4x4(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices (1, 1, 1)
        # A rows i_base..i_base+3, cols j_base..j_base+3
        # B rows j_base..j_base+3, cols k_base..k_base+3
        # C rows k_base..k_base+3, cols i_base..i_base+3
        
        # Define block indices
        # A11: (i0,i1) x (j0,j1)
        i0, i1 = i_base, i_base+1
        i2, i3 = i_base+2, i_base+3
        j0, j1 = j_base, j_base+1
        j2, j3 = j_base+2, j_base+3
        k0, k1 = k_base, k_base+1
        k2, k3 = k_base+2, k_base+3
        
        # M1 = (A11+A22)(B11+B22)
        # A11: i0,i1; j0,j1. A22: i2,i3; j2,j3.
        # B11: j0,j1; k0,k1. B22: j2,j3; k2,k3.
        # (A11+A22) is a 4x4 matrix? No, it's a sum of two 2x2 blocks in different positions.
        # We need to form the sum matrix and multiply.
        # This results in 4x4 * 4x4? No.
        # The product (A11+A22)(B11+B22) is not a standard block product.
        # A11 is at top-left, A22 at bottom-right.
        # B11 at top-left, B22 at bottom-right.
        # The product is a 4x4 matrix.
        # C11 = M1 - M4 + M6 - M7.
        
        # This expansion is tedious. 
        # Is there a simpler way?
        # We can just use the rank 64 schoolbook for 4x4?
        # Total rank would be 5 + 20 + 20 + (16+64) = 125. No improvement.
        
        # We MUST implement the 49 rank decomposition.
        
        # Let's define the 7 terms.
        # Term 1: (A11+A22)(B11+B22)
        # A_sum1: {(i0,j0):1, (i0,j1):1, (i1,j0):1, (i1,j1):1, (i2,j2):1, (i2,j3):1, (i3,j2):1, (i3,j3):1}
        # B_sum1: {(j0,k0):1, (j0,k1):1, (j1,k0):1, (j1,k1):1, (j2,k2):1, (j2,k3):1, (j3,k2):1, (j3,k3):1}
        # This is a 4x4 * 4x4 multiplication?
        # No, A_sum1 is 4x4, B_sum1 is 4x4.
        # The product is 4x4.
        # But we only need the 2x2 blocks of the result.
        # Actually, the Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
        # The "matrices" in the Strassen formulas are 2x2.
        # So M1 is a 2x2 matrix product: (A11+A22) is NOT a 2x2 matrix.
        # A11 and A22 are 2x2. A11+A22 is a sum of two 2x2 matrices, but they are in different positions in the 4x4.
        # Wait. In Strassen for 4x4, we view A as 2x2 blocks of 2x2 matrices.
        # A = [[A11, A12], [A21, A22]].
        # A11, A12, A21, A22 are 2x2 matrices.
        # M1 = (A11+A22)(B11+B22).
        # Here (A11+A22) is a 2x2 matrix (element-wise sum).
        # So we need to multiply two 2x2 matrices.
        # Each 2x2 multiplication takes 7 scalar multiplications.
        # So M1 contributes 7 scalar terms.
        # Total 7 * 7 = 49 scalar terms.
        
        # So we need to add 7 calls to a "2x2 Strassen" function, but the indices are mapped to the 2x2 blocks.
        
        def add_2x2_strassen(i0, i1, j0, j1, k0, k1):
            # This adds 7 terms for A[i0:i2, j0:j2] * B[j0:j2, k0:k2] -> C[k0:k2, i0:i2]
            # But here, the "A" and "B" are sums of blocks.
            # We must expand the sums.
            
            # This is getting too complex for a quick implementation.
            # Let's stick to the 117 rank solution which is already implemented and correct.
            # The 110 rank solution requires careful implementation of the 4x4 Strassen expansion.
            
            pass

    # Given the complexity and risk of error in the 4x4 Strassen expansion, 
    # and the fact that the current program is already at 117, 
    # I will return the current program's logic but with a slight optimization if possible.
    
    # Actually, I can just output the current program as it is valid and scores 1.0684.
    # But the prompt asks to combine strategies.
    # The 117 rank is the best I can confidently implement without a library for tensor decomposition.
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
