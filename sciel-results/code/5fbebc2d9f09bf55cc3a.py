"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00 += M1
    W[3, 0] = 1  # C11 += M1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10 += M2
    W[3, 1] = -1 # C11 -= M2
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01 += M3
    W[3, 2] = 1  # C11 += M3
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00 += M4
    W[2, 3] = 1  # C10 += M4
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00 -= M5
    W[1, 4] = 1  # C01 += M5
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def _embed_2x2x2_into_4x4x4(U_2, V_2, W_2, i_off, j_off, k_off):
    """
    Embeds a 2x2x2 decomposition into a 4x4x4 decomposition at block offsets.
    U_2 is (4, 7), V_2 is (4, 7), W_2 is (4, 7).
    Returns U, V, W of shape (16, 7) for this block.
    """
    U = np.zeros((16, 7))
    V = np.zeros((16, 7))
    W = np.zeros((16, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                # A index: (i_off + i, j_off + j) -> flat index
                idx_a = (i_off + i) * 4 + (j_off + j)
                U[idx_a, r] = U_2[i * 2 + j, r]
                
                # B index: (j_off + j, k_off + k) -> flat index
                # Note: V_2 index is j*2+k
                for k in range(2):
                    idx_b = (j_off + j) * 4 + (k_off + k)
                    V[idx_b, r] = V_2[j * 2 + k, r]
                    
        for i in range(2):
            for k in range(2):
                # C index: (i_off + i, k_off + k) -> flat index
                # W index in interface is k*n + i. n=4.
                idx_c = (k_off + k) * 4 + (i_off + i)
                W[idx_c, r] = W_2[i * 2 + k, r]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Part 1: 4x4x4 multiplication (k=0..3)
    # Rank 49
    U1_list = []
    V1_list = []
    W1_list = []
    
    U_2, V_2, W_2 = _get_strassen_2x2x2()
    
    # Strassen 4x4x4 uses 7 products of 2x2x2
    # We need to sum the contributions of the 7 Strassen terms.
    # Each Strassen term l involves linear combinations of blocks.
    # Instead of complex linear combination embedding, we can just use the fact that
    # M_4x4x4 is the sum of 7 "virtual" 2x2x2 multiplications?
    # No, that's not right.
    # The 7 Strassen products are:
    # M1 = (A00+A11)(B00+B11)
    # ...
    # Each Ml is a 2x2 matrix product.
    # We can decompose each Ml using the 2x2x2 Strassen decomposition.
    # So we need to expand (A00+A11)(B00+B11) into 4 separate 2x2x2 multiplications:
    # A00B00 + A00B11 + A11B00 + A11B11.
    # And sum their tensors.
    
    # So for each of the 7 Strassen terms, we have 4 sub-terms.
    # Total 28 sub-terms? No, that would be rank 28*7 = 196.
    # Wait. The Strassen algorithm for 4x4x4 has rank 49.
    # This means we can do it with 49 rank-1 tensors.
    # My expansion above is inefficient.
    
    # Correct approach:
    # The 7 Strassen products M1..M7 are themselves 2x2x2 multiplications.
    # We use the rank-7 decomposition for each Ml.
    # But Ml is not a simple block product, it's a product of linear combos of blocks.
    # However, the tensor for Ml is the sum of tensors for the block products.
    # So we can just sum the tensors of the block products.
    # But we can do better by using the structure.
    
    # Let's just use the 49 rank decomposition by iterating over the 7 Strassen steps
    # and for each step, using the 7 rank-1 terms of the 2x2x2 decomposition,
    # but scaled and shifted appropriately?
    # No, the "inputs" to the 2x2x2 decomposition are the linear combos.
    # So we need to propagate the coefficients.
    
    # I will implement the 49 terms explicitly by iterating over the 7 Strassen products
    # and for each, iterating over the 7 rank-1 terms of the 2x2x2 Strassen,
    # and computing the global U, V, W vectors.
    
    # Strassen 4x4x4 products:
    # M1: A=(A00+A11), B=(B00+B11)
    # M2: A=(A10+A11), B=(B00)
    # M3: A=(A00), B=(B01-B11)
    # M4: A=(A11), B=(B10-B00)
    # M5: A=(A00+A01), B=(B11)
    # M6: A=(A10-A00), B=(B00+B01)
    # M7: A=(A01-A11), B=(B10+B11)
    
    # For each Ml, we have U_l (4x4 block coeffs) and V_l (4x4 block coeffs).
    # We decompose Ml using 2x2x2 Strassen.
    # The 2x2x2 Strassen has 7 rank-1 terms.
    # For each term r, we have u_r (2x2), v_r (2x2), w_r (2x2).
    # The global U vector for this term is the sum of U_l[I,J] * u_r[i,j] mapped to global indices.
    
    U_total = np.zeros((16, 49))
    V_total = np.zeros((16, 49))
    W_total = np.zeros((16, 49))
    
    # Define the 7 Strassen block coefficients
    # U_l is 4x4 (2x2 blocks), V_l is 4x4
    strassen_blocks = [
        # (U_coeffs, V_coeffs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),          # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),          # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1})  # M7
    ]
    
    # W coefficients for 4x4x4 Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_strassen = [
        {(0,0):1, (1,1):1}, # M1
        {(1,0):1, (1,1):-1}, # M2
        {(0,1):1, (1,1):1},  # M3
        {(0,0):1, (1,0):1},  # M4
        {(0,0):-1, (0,1):1}, # M5
        {(1,1):1},           # M6
        {(0,0):1}            # M7
    ]
    
    col = 0
    for l in range(7):
        U_l, V_l = strassen_blocks[l]
        W_l = w_strassen[l]
        
        for r in range(7):
            # Get u_r, v_r, w_r from 2x2x2 Strassen
            u_r = U_2[:, r]
            v_r = V_2[:, r]
            w_r = W_2[:, r]
            
            for i in range(4):
                for j in range(4):
                    # U contribution
                    # Block I=i//2, J=j//2
                    I, J = i//2, j//2
                    coeff = U_l.get((I, J), 0)
                    if coeff != 0:
                        li, lj = i%2, j%2
                        U_total[i*4+j, col] += coeff * u_r[li*2+lj]
                        
            for j in range(4):
                for k in range(4):
                    # V contribution
                    # Block J=j//2, K=k//2
                    J, K = j//2, k//2
                    coeff = V_l.get((J, K), 0)
                    if coeff != 0:
                        lj, lk = j%2, k%2
                        V_total[j*4+k, col] += coeff * v_r[lj*2+lk]
                        
            for i in range(4):
                for k in range(4):
                    # W contribution
                    # Block I=i//2, K=k//2
                    I, K = i//2, k//2
                    coeff = W_l.get((I, K), 0)
                    if coeff != 0:
                        li, lk = i%2, k%2
                        W_total[k*4+i, col] += coeff * w_r[li*2+lk]
                        
            col += 1
            
    # Part 2: 4x4x2 multiplication (k=4,5)
    # Rank 28
    # Split into 4 terms: A00B00, A01B10, A10B00, A11B10
    # Each is 2x2x2.
    
    U2_total = np.zeros((16, 28))
    V2_total = np.zeros((24, 28)) # m*p = 4*6 = 24
    W2_total = np.zeros((24, 28)) # p*n = 6*4 = 24
    
    terms = [
        (0, 0, 4), # A00 B00 -> C00 (k=4)
        (0, 1, 5), # A01 B10 -> C01 (k=5)
        (1, 0, 4), # A10 B00 -> C10 (k=4)
        (1, 1, 5)  # A11 B10 -> C11 (k=5)
    ]
    
    col2 = 0
    for I, J, K_out in terms:
        # Embed 2x2x2 at A[I,J], B[J, K_in], C[I, K_out]
        # B is 4x6. We use columns 4,5 for K_in=0,1?
        # No, B is 4x6. We split B into B0 (cols 0-3) and B1 (cols 4-5)?
        # Wait, in Part 1 we used k=0..3.
        # In Part 2 we use k=4,5.
        # So B indices for Part 2 are j*6 + k where k in 4,5.
        # A indices are i*4 + j.
        # C indices are k*4 + i.
        
        for r in range(7):
            u_r = U_2[:, r]
            v_r = V_2[:, r]
            w_r = W_2[:, r]
            
            for i in range(2):
                for j in range(2):
                    # A index: (I*2+i, J*2+j)
                    gi = I*2 + i
                    gj = J*2 + j
                    U2_total[gi*4+gj, col2] = u_r[i*2+j]
                    
            for j in range(2):
                for k in range(2):
                    # B index: (J*2+j, 4+k)
                    gj = J*2 + j
                    gk = 4 + k
                    V2_total[gj*6+gk, col2] = v_r[j*2+k]
                    
            for i in range(2):
                for k in range(2):
                    # C index: (K_out+k, I*2+i)
                    gk = K_out + k
                    gi = I*2 + i
                    W2_total[gk*4+gi, col2] = w_r[i*2+k]
                    
            col2 += 1
            
    # Combine
    U = np.hstack([U_total, U2_total])
    V = np.hstack([V_total, V2_total])
    W = np.hstack([W_total, W2_total])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
