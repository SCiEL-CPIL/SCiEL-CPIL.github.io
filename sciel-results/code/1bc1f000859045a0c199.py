"""Tensor decomposition for 4x4 * 4x8 matrix multiplication using recursive Strassen."""

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 with Strassen's 7-term decomposition.
    U: shape (4, 7), indexed [i*2+j, t]
    V: shape (4, 7), indexed [j*2+k, t]
    W: shape (4, 7), indexed [k*2+i, t]  (trace convention: C[i,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t])
    """
    # Strassen's 7 products:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B10 - B11)
    # P4 = A11 (B01 - B10)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U[i*2+j, t] coefficients for A_ij
    # V[j*2+k, t] coefficients for B_jk
    # W[k*2+i, t] coefficients for C_ik (trace convention)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # t=0: P1
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # t=1: P2
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1; V[1][1] = 1  # B00, B10... wait B is 2x2, B00 is [0,0], B01 is [0,1]
    # B indices: row j, col k. B00 = j=0,k=0 -> idx 0. B01 = j=0,k=1 -> idx 1.
    # B10 = j=1,k=0 -> idx 2. B11 = j=1,k=1 -> idx 3.
    # P2 = (A10+A11)*B00. So V should have B00 = idx 0.
    V[0][1] = 1  # B00
    # C = P2 contributes to C10. C10 = i=1,k=0 -> idx k*2+i = 0*2+1 = 1.
    W[1][1] = 1
    
    # t=2: P3
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10 - B11
    # C = P3 contributes to C01 and C11. C01 = i=0,k=1 -> idx 1*2+0=2. C11 = i=1,k=1 -> idx 1*2+1=3.
    W[2][2] = 1; W[3][2] = 1
    
    # t=3: P4
    U[3][3] = 1  # A11
    V[1][3] = 1; V[2][3] = -1  # B01 - B10
    # C = P4 contributes to C00 and C10. C00 = idx 0, C10 = idx 1.
    W[0][3] = 1; W[1][3] = 1
    
    # t=4: P5
    U[0][4] = 1; U[1][4] = 1  # A00 + A01
    V[3][4] = 1  # B11
    # C = P5 contributes to C01 (positive) and C00 (negative).
    W[2][4] = 1; W[0][4] = -1
    
    # t=5: P6
    U[2][5] = 1; U[0][5] = -1  # A10 - A00
    V[0][5] = 1; V[1][5] = 1  # B00 + B01
    # C = P6 contributes to C11.
    W[3][5] = 1
    
    # t=6: P7
    U[1][6] = 1; U[3][6] = -1  # A01 - A11
    V[2][6] = 1; V[3][6] = 1  # B10 + B11
    # C = P7 contributes to C00.
    W[0][6] = 1
    
    return U, V, W


def _strassen_2x2_2x4():
    """Return (U, V, W) for 2x2 * 2x4 -> 2x4 using Strassen-like decomposition.
    Split B into two 2x2 blocks: B = [B_L, B_R] where B_L, B_R are 2x2.
    Use 7 products, each computing a 2x2 * 2x2 product.
    U: shape (4, 14), V: shape (8, 14), W: shape (8, 14)
    Trace convention: C[i,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    A is 2x2: idx i*2+j
    B is 2x4: idx j*4+k, where k=0..1 is left block, k=2..3 is right block.
    C is 2x4: idx k*2+i, where k=0..1 is left, k=2..3 is right.
    """
    # We'll use two independent Strassen decompositions for the left and right halves.
    # For each half, we have 7 terms. Total rank = 14.
    # For left half (k=0,1): C_L[i, k_L] where k_L=0,1. Global k = k_L.
    # For right half (k=2,3): C_R[i, k_R] where k_R=0,1. Global k = 2 + k_R.
    
    U_base, V_base, W_base = _strassen_2x2_2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    # Left half: terms 0-6
    # B left half: j*4 + k_L, where k_L in {0,1}
    # C left half: k_L*2 + i
    for t in range(7):
        for ij in range(4):
            U[ij][t] = U_base[ij][t]
        for jk in range(4):
            j = jk // 2
            k_L = jk % 2
            # V_base index is j*2 + k_L = jk
            V[j*4 + k_L][t] = V_base[jk][t]
        for ki in range(4):
            k_L = ki // 2
            i = ki % 2
            # W_base index is k_L*2 + i = ki
            W[k_L*2 + i][t] = W_base[ki][t]
    
    # Right half: terms 7-13
    # B right half: j*4 + 2 + k_R, where k_R in {0,1}
    # C right half: (2+k_R)*2 + i = 4 + 2*k_R + i
    for t in range(7):
        tt = t + 7
        for ij in range(4):
            U[ij][tt] = U_base[ij][t]
        for jk in range(4):
            j = jk // 2
            k_R = jk % 2
            V[j*4 + 2 + k_R][tt] = V_base[jk][t]
        for ki in range(4):
            k_R = ki // 2
            i = ki % 2
            W[(2+k_R)*2 + i][tt] = W_base[ki][t]
    
    return U, V, W


def construct():
    n, m, p = 4, 4, 8
    
    # We decompose 4x4 * 4x8 as follows:
    # Split A into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B into two 4x4 blocks horizontally: B = [B_L, B_R]
    # For each half, we have 4x4 * 4x4.
    # Use recursive Strassen: 4x4 * 4x4 has rank 7^2 = 49.
    # Total rank = 49 * 2 = 98.
    
    # Actually, let's think more carefully. We want to multiply 4x4 * 4x8.
    # We can view this as two independent 4x4 * 4x4 multiplications (left and right halves of B).
    # Each 4x4 * 4x4 can be done with rank 49 (Strassen squared).
    # So total rank = 98.
    
    # For 4x4 * 4x4 Strassen:
    # A is 4x4, split into 2x2 blocks. B is 4x4, split into 2x2 blocks.
    # 7 products, each 2x2 * 2x2, each of rank 7. Total rank 49.
    
    # Let's construct the 4x4 * 4x4 decomposition first.
    
    def _strassen_4x4_4x4():
        """4x4 * 4x4 -> 4x4, rank 49.
        U: (16, 49), V: (16, 49), W: (16, 49)
        A idx: i*4+j, B idx: j*4+k, C idx: k*4+i
        """
        U_outer, V_outer, W_outer = _strassen_2x2_2x2()
        U_inner, V_inner, W_inner = _strassen_2x2_2x2()
        
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for s in range(7):
            for r in range(7):
                t = s * 7 + r
                # Outer term s gives coefficients for the 2x2 block multipliers.
                # Inner term r gives coefficients for the 2x2 within each block.
                # A block (I,J): global index (2I+i)*4 + (2J+j) = 8I + 4J + 2i + j
                # U_inner[r, i*2+j] * U_outer[s, I*2+J]
                for I in range(2):
                    for J in range(2):
                        outer_idx = I * 2 + J
                        for i in range(2):
                            for j in range(2):
                                inner_idx = i * 2 + j
                                global_idx = (2*I + i) * 4 + (2*J + j)
                                U[global_idx][t] = U_inner[inner_idx][r] * U_outer[outer_idx][s]
                
                # B block (J,K): global index (2J+j)*4 + (2K+k) = 8J + 4K + 2j + k
                for J in range(2):
                    for K in range(2):
                        outer_idx = J * 2 + K
                        for j in range(2):
                            for k in range(2):
                                inner_idx = j * 2 + k
                                global_idx = (2*J + j) * 4 + (2*K + k)
                                V[global_idx][t] = V_inner[inner_idx][r] * V_outer[outer_idx][s]
                
                # C block (I,K): global index (2K+k)*4 + (2I+i) = 8K + 4I + 2k + i
                for I in range(2):
                    for K in range(2):
                        outer_idx = I * 2 + K
                        for k in range(2):
                            for i in range(2):
                                inner_idx = k * 2 + i
                                global_idx = (2*K + k) * 4 + (2*I + i)
                                W[global_idx][t] = W_inner[inner_idx][r] * W_outer[outer_idx][s]
        
        return U, V, W
    
    U_44, V_44, W_44 = _strassen_4x4_4x4()
    
    # Now for 4x4 * 4x8:
    # Split B into B_L (4x4) and B_R (4x4).
    # C_L = A * B_L, C_R = A * B_R.
    # C = [C_L, C_R] is 4x8.
    # C idx: k*4+i where k=0..7.
    # For C_L: k=0..3, C_L[k, i] = C[k, i] (global C idx = k*4+i)
    # For C_R: k=4..7, C_R[k-4, i] = C[k, i] (global C idx = k*4+i)
    
    # B_L: j*4 + k_L, k_L=0..3 -> global B idx = j*4 + k_L
    # B_R: j*4 + 4 + k_R, k_R=0..3 -> global B idx = j*4 + 4 + k_R
    
    # A is the same for both: 4x4, idx i*4+j
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # Left half: terms 0-48
    for t in range(49):
        for ij in range(16):
            U[ij][t] = U_44[ij][t]
        for jk in range(16):
            j = jk // 4
            k_L = jk % 4
            V[j*8 + k_L][t] = V_44[jk][t]
        for ki in range(16):
            k_L = ki // 4
            i = ki % 4
            W[k_L*4 + i][t] = W_44[ki][t]
    
    # Right half: terms 49-97
    for t in range(49):
        tt = t + 49
        for ij in range(16):
            U[ij][tt] = U_44[ij][t]
        for jk in range(16):
            j = jk // 4
            k_R = jk % 4
            V[j*8 + 4 + k_R][tt] = V_44[jk][t]
        for ki in range(16):
            k_R = ki // 4
            i = ki % 4
            W[(4+k_R)*4 + i][tt] = W_44[ki][t]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
