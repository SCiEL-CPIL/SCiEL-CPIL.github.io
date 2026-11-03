"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # A is 2x3, B is 3x10.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x10) and B2 (1x10).
    # C = A1 B1 + A2 B2.
    
    # Rank of M_2,2,10 using Strassen blocks:
    # B1 is 2x10, split into 5 blocks of 2x2.
    # Each 2x2 by 2x2 uses rank 7.
    # Total rank for A1 B1 is 5 * 7 = 35.
    
    # Rank of M_2,1,10:
    # A2 is 2x1, B2 is 1x10.
    # C_ik = A2_i1 B2_1k.
    # This is rank 2 * 10 = 20.
    
    total_rank = 35 + 20
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Part 1: A1 B1 (2x2 by 2x10)
    # Strassen for 2x2 by 2x2
    # A1 indices: i in 0..1, j in 0..1
    # B1 indices: j in 0..1, k in 0..9
    
    # Strassen coefficients for 2x2 by 2x2
    # U, V, W for one block
    # U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries)
    # We need to map these to the global indices.
    
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 terms for one 2x2 block.
    # Each term c has U_c (2x2), V_c (2x2), W_c (2x2).
    # U_c[i,j] is coeff of A_ij.
    # V_c[j,k] is coeff of B_jk.
    # W_c[k,i] is coeff of C_ki? No, W is indexed by k*n+i.
    # In the tensor sum, W_{k,i,c} multiplies the product.
    
    # Let's define the 7 U, V, W matrices for 2x2.
    # Note: W in the tensor decomposition is p*n x rank.
    # For 2x2, W is 4 x 7.
    # W_{k,i,c} corresponds to the coefficient of C_{i,k} in the reconstruction.
    # Wait, the sum is sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
    # So W_{k,i,c} is the coefficient that multiplies the product to contribute to T_{i,j,k}=1.
    # Actually, the product U_{i,j,c} V_{j,k,c} is a scalar for fixed i,j,k,c.
    # So W_{k,i,c} must be such that the sum is 1.
    
    # Let's construct the 7 terms.
    # Term 1: M1
    # U1: A00=1, A11=1. Others 0.
    # V1: B00=1, B11=1. Others 0.
    # W1: C00=1, C11=1.
    # Check: U1_{0,0} V1_{0,0} W1_{0,0} = 1*1*1 = 1.
    # U1_{0,0} V1_{0,1} W1_{1,0} = 1*0*0 = 0.
    # This seems correct.
    
    def get_strassen_block():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2 by 2x2
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        
        # Indices: 0:00, 1:01, 2:10, 3:11
        # U
        U_b[0, 0] = 1; U_b[3, 0] = 1 # M1
        U_b[2, 1] = 1; U_b[3, 1] = 1 # M2
        U_b[0, 2] = 1 # M3
        U_b[3, 3] = 1 # M4
        U_b[0, 4] = 1; U_b[1, 4] = 1 # M5
        U_b[2, 5] = 1; U_b[0, 5] = -1 # M6
        U_b[1, 6] = 1; U_b[3, 6] = -1 # M7
        
        # V
        V_b[0, 0] = 1; V_b[3, 0] = 1 # M1
        V_b[0, 1] = 1 # M2
        V_b[3, 2] = 1; V_b[1, 2] = -1 # M3
        V_b[2, 3] = 1; V_b[0, 3] = -1 # M4
        V_b[3, 4] = 1 # M5
        V_b[0, 5] = 1; V_b[1, 5] = 1 # M6
        V_b[2, 6] = 1; V_b[3, 6] = 1 # M7
        
        # W
        # C00 = M1 + M4 - M5 + M7
        W_b[0, 0] = 1 # C00 from M1
        W_b[0, 3] = 1 # C00 from M4
        W_b[0, 4] = -1 # C00 from M5
        W_b[0, 6] = 1 # C00 from M7
        
        # C01 = M3 + M5
        W_b[1, 2] = 1 # C01 from M3
        W_b[1, 4] = 1 # C01 from M5
        
        # C10 = M2 + M4
        W_b[2, 1] = 1 # C10 from M2
        W_b[2, 3] = 1 # C10 from M4
        
        # C11 = M1 - M2 + M3 + M6
        W_b[3, 0] = 1 # C11 from M1
        W_b[3, 1] = -1 # C11 from M2
        W_b[3, 2] = 1 # C11 from M3
        W_b[3, 5] = 1 # C11 from M6
        
        return U_b, V_b, W_b

    U_b, V_b, W_b = get_strassen_block()
    
    for b in range(5):
        # Block b corresponds to k in 2b, 2b+1
        # U indices: i in 0..1, j in 0..1 (A1)
        # V indices: j in 0..1, k in 2b..2b+1 (B1)
        # W indices: k in 2b..2b+1, i in 0..1 (C)
        
        for c in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = U_b[i*2+j, c]
                    if val != 0:
                        U[i*m + j, col] = val
                        
            # V
            for j in range(2):
                for dk in range(2):
                    k = 2*b + dk
                    val = V_b[j*2+dk, c]
                    if val != 0:
                        V[j*p + k, col] = val
                        
            # W
            for i in range(2):
                for dk in range(2):
                    k = 2*b + dk
                    val = W_b[dk*2+i, c] # W_b index is k_local * 2 + i_local
                    if val != 0:
                        W[k*n + i, col] = val
                        
            col += 1
            
    # Part 2: A2 B2 (2x1 by 1x10)
    # A2: i in 0..1, j=2
    # B2: j=2, k in 0..9
    # C: k in 0..9, i in 0..1
    
    for i in range(2):
        for k in range(10):
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
