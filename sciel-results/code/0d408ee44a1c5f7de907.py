"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2_tensors():
    """
    Returns the 7 rank-1 tensors for 2x2 matrix multiplication.
    Each tensor is (U, V, W) where U, V, W are vectors of length 4.
    U corresponds to A[i,j], V to B[j,k], W to C[i,k].
    Indices: a=i*2+j, b=j*2+k, c=k*2+i.
    """
    tensors = []
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,j=0 -> a=0. A22: i=1,j=1 -> a=3.
    # B11: j=0,k=0 -> b=0. B22: j=1,k=1 -> b=3.
    # C11: i=0,k=0 -> c=0. C12: i=0,k=1 -> c=1.
    # C21: i=1,k=0 -> c=2. C22: i=1,k=1 -> c=3.
    
    # We need to express C[i,k] as sum of terms.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    # A parts: (0,0) and (1,1). Coeff 1.
    # B parts: (0,0) and (1,1). Coeff 1.
    # C parts: (0,0), (0,1), (1,0), (1,1). Coeff 1.
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 1, 1, 1] # C11, C12, C21, C22 all get +1 from M1?
    # Wait. M1 contributes to C11, C12, C21, C22?
    # C11 = M1...
    # C12 = ... (No M1)
    # C21 = ... (No M1)
    # C22 = M1...
    # So W1 should be [1, 0, 0, 1].
    W1 = [1, 0, 0, 1]
    tensors.append((U1, V1, W1))
    
    # Term 2: M2 = (A21+A22)B11
    # A21: i=1,j=0 -> a=2. A22: i=1,j=1 -> a=3.
    # B11: j=0,k=0 -> b=0.
    # C21: i=1,k=0 -> c=2.
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 0, 1, 0]
    tensors.append((U2, V2, W2))
    
    # Term 3: M3 = A11(B12-B22)
    # A11: i=0,j=0 -> a=0.
    # B12: j=0,k=1 -> b=1. B22: j=1,k=1 -> b=3.
    # C12: i=0,k=1 -> c=1. C22: i=1,k=1 -> c=3.
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 1, 0, 1]
    tensors.append((U3, V3, W3))
    
    # Term 4: M4 = A22(B21-B11)
    # A22: i=1,j=1 -> a=3.
    # B21: j=1,k=0 -> b=2. B11: j=0,k=0 -> b=0.
    # C11: i=0,k=0 -> c=0. C21: i=1,k=0 -> c=2.
    U4 = [0, 0, 0, 1]
    V4 = [0, 0, 1, 0] # B21 is b=2, B11 is b=0. Wait.
    # V4 should have 1 at b=2 and -1 at b=0.
    V4 = [-1, 0, 1, 0]
    W4 = [1, 0, 1, 0]
    tensors.append((U4, V4, W4))
    
    # Term 5: M5 = (A11+A12)B22
    # A11: a=0. A12: i=0,j=1 -> a=1.
    # B22: j=1,k=1 -> b=3.
    # C11: i=0,k=0 -> c=0 (coeff -1). C12: i=0,k=1 -> c=1 (coeff 1).
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 1, 0, 0]
    tensors.append((U5, V5, W5))
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # A21: a=2. A11: a=0.
    # B11: b=0. B12: b=1.
    # C22: i=1,k=1 -> c=3 (coeff 1).
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    tensors.append((U6, V6, W6))
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # A12: a=1. A22: a=3.
    # B21: b=2. B22: b=3.
    # C11: i=0,k=0 -> c=0 (coeff 1).
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 0]
    tensors.append((U7, V7, W7))
    
    return tensors

def _construct_4x4_core_tensors():
    """
    Generates the 49 rank-1 tensors for 4x4 matrix multiplication
    using nested Strassen.
    """
    strassen_2x2 = _strassen_2x2_tensors()
    core_tensors = []
    
    # The 4x4 matrix is composed of 2x2 blocks.
    # We apply Strassen at the block level.
    # The 7 "multiplications" M1..M7 are themselves 2x2 multiplications.
    # So for each block-level term (coeffs U_block, V_block, W_block),
    # we expand it into 7 rank-1 terms using the 2x2 Strassen tensors.
    
    # Block level indices:
    # I, J, K in {0, 1}
    # A[I, J], B[J, K], C[I, K]
    
    # We need to map the block-level Strassen coefficients to the 2x2 level.
    # This is tricky because the block-level "multiplication" is not just a scalar product,
    # it's a 2x2 matrix product.
    # So M1_block = (A11_block + A22_block) * (B11_block + B22_block).
    # This is a sum of 4 scalar products? No, it's a matrix product.
    # (A11 + A22) is a 2x2 matrix. (B11 + B22) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # This product can be decomposed into 7 rank-1 tensors (Strassen for 2x2).
    
    # So, for each of the 7 block-level terms, we have a 2x2 matrix product.
    # We decompose that 2x2 product into 7 rank-1 tensors.
    # Total 49 terms.
    
    # We need to define the block-level Strassen coefficients for A, B, C.
    # Let S_A[I, J] be the coefficient of A[I, J] in the block-level term.
    # Let S_B[J, K] be the coefficient of B[J, K].
    # Let S_C[I, K] be the coefficient of C[I, K].
    
    # We reuse the 2x2 Strassen tensors, but now the "elements" are 2x2 blocks.
    # The 7 tensors from _strassen_2x2_tensors() give us S_A, S_B, S_C for the blocks.
    
    for block_idx in range(7):
        U_block, V_block, W_block = strassen_2x2[block_idx]
        # U_block is a vector of 4 coeffs for A[I, J] where I,J in 0,1.
        # Index: I*2+J.
        
        # We need to expand the product of the block matrices.
        # The block product is P = (sum S_A[I,J] A[I,J]) * (sum S_B[J,K] B[J,K]).
        # This is a 2x2 matrix.
        # We decompose this 2x2 matrix product into 7 rank-1 tensors.
        
        # For each of the 7 inner Strassen terms:
        for inner_idx in range(7):
            U_inner, V_inner, W_inner = strassen_2x2[inner_idx]
            # U_inner is coeffs for local A[i', j'] (2x2).
            # V_inner is coeffs for local B[j', k'] (2x2).
            # W_inner is coeffs for local C[i', k'] (2x2).
            
            # The full term is:
            # sum_{I,J,i',j'} S_A[I,J] U_inner[i'2+j'] A[I*2+i', J*2+j']
            # * sum_{J,K,j',k'} S_B[J,K] V_inner[j'2+k'] B[J*2+j', K*2+k']
            # * sum_{I,K,i',k'} S_C[I,K] W_inner[i'2+k'] C[I*2+i', K*2+k']
            
            # This is a rank-1 tensor in the 4x4x4 space.
            U_full = [0.0] * 16
            V_full = [0.0] * 16
            W_full = [0.0] * 16
            
            for I in range(2):
                for J in range(2):
                    coeff_A = U_block[I*2+J]
                    if coeff_A == 0: continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            coeff_inner_A = U_inner[i_prime*2+j_prime]
                            if coeff_inner_A == 0: continue
                            i = I*2 + i_prime
                            j = J*2 + j_prime
                            # Global index in 4x4 core: i*4+j
                            idx_A = i*4 + j
                            U_full[idx_A] += coeff_A * coeff_inner_A
            
            for J in range(2):
                for K in range(2):
                    coeff_B = V_block[J*2+K]
                    if coeff_B == 0: continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            coeff_inner_B = V_inner[j_prime*2+k_prime]
                            if coeff_inner_B == 0: continue
                            j = J*2 + j_prime
                            k = K*2 + k_prime
                            # Global index in 4x4 core: j*4+k
                            idx_B = j*4 + k
                            V_full[idx_B] += coeff_B * coeff_inner_B
            
            for I in range(2):
                for K in range(2):
                    coeff_C = W_block[I*2+K]
                    if coeff_C == 0: continue
                    for i_prime in range(2):
                        for k_prime in range(2):
                            coeff_inner_C = W_inner[i_prime*2+k_prime]
                            if coeff_inner_C == 0: continue
                            i = I*2 + i_prime
                            k = K*2 + k_prime
                            # Global index in 4x4 core: k*4+i (Note: W index is k*n+i)
                            idx_C = k*4 + i
                            W_full[idx_C] += coeff_C * coeff_inner_C
            
            core_tensors.append((U_full, V_full, W_full))
            
    return core_tensors

def construct():
    """
    Returns the best matrix multiplication tensor decomposition found.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Core: 4x4x4 (j<4, k<4)
    # Strip: 4x1x6 (j=4, k<6)
    
    core_tensors = _construct_4x4_core_tensors()
    core_rank = len(core_tensors) # 49
    
    strip_rank = n * p # 24
    
    total_rank = core_rank + strip_rank
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Fill Core
    for r, (U_c, V_c, W_c) in enumerate(core_tensors):
        # U_c is 16 entries for i,j in 0..3.
        # Global index a = i*5 + j.
        for i in range(4):
            for j in range(4):
                a = i*5 + j
                U[a, r] = U_c[i*4+j]
        
        # V_c is 16 entries for j,k in 0..3.
        # Global index b = j*6 + k.
        for j in range(4):
            for k in range(4):
                b = j*6 + k
                V[b, r] = V_c[j*4+k]
        
        # W_c is 16 entries for k,i in 0..3.
        # Global index c = k*4 + i.
        for k in range(4):
            for i in range(4):
                c = k*4 + i
                W[c, r] = W_c[k*4+i]
                
    # Fill Strip
    # j=4
    # i in 0..3, k in 0..5
    # a = i*5 + 4
    # b = 4*6 + k = 24 + k
    # c = k*4 + i
    
    for i in range(n):
        for k in range(p):
            r = core_rank + i*p + k
            a = i*m + 4
            b = 4*p + k
            c = k*n + i
            U[a, r] = 1
            V[b, r] = 1
            W[c, r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
