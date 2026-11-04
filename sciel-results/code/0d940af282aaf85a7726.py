import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _build_tensor():
    n, m, p = 4, 4, 8
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, T):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _to_half_integer(mat):
    mat2 = mat * 2
    rounded = np.round(mat2)
    if np.allclose(mat2, rounded, atol=1e-6):
        return (rounded / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()

def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _laderman_3x3x3():
    # Laderman's rank-23 algorithm for 3x3 matrix multiplication
    # U[i,j], V[k,l], W[m,n] with indices 0..2 for 3x3
    U = np.zeros((9, 23), dtype=np.float64)
    V = np.zeros((9, 23), dtype=np.float64)
    W = np.zeros((9, 23), dtype=np.float64)
    
    # Laderman's 23 products (standard formulation)
    # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    
    # Product 1: (a00+a11+a22)(b00+b11+b22)
    U[0,0]=1; U[4,0]=1; U[8,0]=1
    V[0,0]=1; V[4,0]=1; V[8,0]=1
    W[0,0]=1; W[4,0]=1; W[8,0]=1; W[1,0]=-1; W[2,0]=-1; W[3,0]=-1; W[5,0]=-1; W[6,0]=-1; W[7,0]=-1
    
    # Product 2: (a00+a11)(b00+b11)
    U[0,1]=1; U[4,1]=1
    V[0,1]=1; V[4,1]=1
    W[0,1]=1; W[4,1]=1; W[1,1]=-1; W[3,1]=-1; W[5,1]=-1; W[7,1]=-1
    
    # Product 3: (a00+a22)(b00+b22)
    U[0,2]=1; U[8,2]=1
    V[0,2]=1; V[8,2]=1
    W[0,2]=1; W[8,2]=1; W[2,2]=-1; W[6,2]=-1
    
    # Product 4: (a11+a22)(b11+b22)
    U[4,3]=1; U[8,3]=1
    V[4,3]=1; V[8,3]=1
    W[4,3]=1; W[8,3]=1; W[5,3]=-1; W[7,3]=-1
    
    # Product 5: (a00+a01+a10+a11)(b00+b01+b10+b11)
    U[0,4]=1; U[1,4]=1; U[3,4]=1; U[4,4]=1
    V[0,4]=1; V[1,4]=1; V[3,4]=1; V[4,4]=1
    W[0,4]=1; W[1,4]=1; W[3,4]=1; W[4,4]=1; W[2,4]=-1; W[5,4]=-1; W[6,4]=-1; W[7,4]=-1
    
    # Product 6: (a01+a11+a12+a22)(b02+b11+b12+b22)
    U[1,5]=1; U[4,5]=1; U[5,5]=1; U[8,5]=1
    V[1,5]=1; V[4,5]=1; V[5,5]=1; V[8,5]=1
    W[1,5]=1; W[4,5]=1; W[5,5]=1; W[8,5]=1; W[2,5]=-1; W[6,5]=-1; W[7,5]=-1
    
    # Product 7: (a00+a02+a20+a22)(b00+b02+b20+b22)
    U[0,6]=1; U[2,6]=1; U[6,6]=1; U[8,6]=1
    V[0,6]=1; V[2,6]=1; V[6,6]=1; V[8,6]=1
    W[0,6]=1; W[2,6]=1; W[6,6]=1; W[8,6]=1; W[1,6]=-1; W[3,6]=-1; W[5,6]=-1; W[7,6]=-1
    
    # Product 8: (a02+a12)(b12+b22)
    U[2,7]=1; U[5,7]=1
    V[5,7]=1; V[8,7]=1
    W[2,7]=1; W[5,7]=1; W[6,7]=-1
    
    # Product 9: (a20+a22)(b02+b22)
    U[6,8]=1; U[8,8]=1
    V[2,8]=1; V[8,8]=1
    W[6,8]=1; W[8,8]=1; W[7,8]=-1
    
    # Product 10: (a01+a21)(b10+b20)
    U[1,9]=1; U[7,9]=1
    V[3,9]=1; V[6,9]=1
    W[3,9]=1; W[7,9]=1; W[2,9]=-1
    
    # Product 11: (a10+a20)(b00+b20)
    U[3,10]=1; U[6,10]=1
    V[0,10]=1; V[6,10]=1
    W[3,10]=1; W[6,10]=1; W[5,10]=-1
    
    # Product 12: (a00+a11+a22)(b01+b12+b20)
    U[0,11]=1; U[4,11]=1; U[8,11]=1
    V[1,11]=1; V[5,11]=1; V[6,11]=1
    W[1,11]=1; W[5,11]=1; W[6,11]=1; W[0,11]=-1; W[2,11]=-1; W[3,11]=-1; W[4,11]=-1; W[7,11]=-1; W[8,11]=-1
    
    # Product 13: (a01+a12)(b00+b11)
    U[1,12]=1; U[5,12]=1
    V[0,12]=1; V[4,12]=1
    W[1,12]=1; W[5,12]=1; W[0,12]=-1; W[4,12]=-1; W[2,12]=1; W[6,12]=-1
    
    # Product 14: (a02+a11)(b10+b22)
    U[2,13]=1; U[4,13]=1
    V[3,13]=1; V[8,13]=1
    W[2,13]=1; W[4,13]=1; W[3,13]=-1; W[8,13]=-1; W[6,12]=0  # placeholder
    
    # Product 15: (a10+a22)(b01+b20)
    U[3,14]=1; U[8,14]=1
    V[1,14]=1; V[6,14]=1
    W[3,14]=1; W[8,14]=1; W[1,14]=-1; W[6,14]=-1; W[5,14]=1; W[7,14]=-1
    
    # Product 16: (a01+a21)(b02+b22)
    U[1,15]=1; U[7,15]=1
    V[2,15]=1; V[8,15]=1
    W[2,15]=1; W[7,15]=1; W[1,15]=-1; W[8,15]=-1
    
    # Product 17: (a00+a22)(b12+b20)
    U[0,16]=1; U[8,16]=1
    V[5,16]=1; V[6,16]=1
    W[0,16]=1; W[8,16]=1; W[5,16]=-1; W[6,16]=-1
    
    # Product 18: (a10+a12)(b01+b22)
    U[3,17]=1; U[5,17]=1
    V[1,17]=1; V[8,17]=1
    W[3,17]=1; W[5,17]=1; W[1,17]=-1; W[8,17]=-1
    
    # Product 19: (a02+a20)(b10+b12)
    U[2,18]=1; U[6,18]=1
    V[3,18]=1; V[5,18]=1
    W[2,18]=1; W[6,18]=1; W[3,18]=-1; W[5,18]=-1
    
    # Product 20: (a01+a12)(b10+b21)
    U[1,19]=1; U[5,19]=1
    V[3,19]=1; V[7,19]=1
    W[3,19]=1; W[7,19]=1; W[1,19]=-1; W[5,19]=-1
    
    # Product 21: (a02+a11)(b01+b20)
    U[2,20]=1; U[4,20]=1
    V[1,20]=1; V[6,20]=1
    W[2,20]=1; W[4,20]=1; W[1,20]=-1; W[6,20]=-1
    
    # Product 22: (a10+a22)(b02+b11)
    U[3,21]=1; U[8,21]=1
    V[2,21]=1; V[4,21]=1
    W[3,21]=1; W[8,21]=1; W[2,21]=-1; W[4,21]=-1
    
    # Product 23: (a00+a11)(b12+b20)
    U[0,22]=1; U[4,22]=1
    V[5,22]=1; V[6,22]=1
    W[0,22]=1; W[4,22]=1; W[5,22]=-1; W[6,22]=-1
    
    return U, V, W

def _verify_laderman():
    """Verify Laderman decomposition is correct."""
    U, V, W = _laderman_3x3x3()
    T = np.zeros((9,9,9), dtype=np.int64)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                T[i*3+k, k*3+j, j*3+i] = 1
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _build_4x4x8_kron():
    """
    Build 4x4x8 decomposition using Kronecker product structure.
    
    Strategy: View 4x4x8 as a 2x2 block structure where:
    - A is 4x4 = 2x2 blocks of 2x2
    - B is 4x8 = 2x4 blocks of 2x4  (or 4x2 blocks)
    - C is 8x4 = 4x2 blocks of 2x2
    
    Actually, let's use a cleaner approach:
    4x4x8 can be decomposed as:
    - Split the 4x4 input into 2x2 blocks (2x2 grid)
    - Split the 4x8 input into 2x4 blocks (2x2 grid of 2x4)
    - Output 8x4 into 4x2 blocks (4x2 grid of 2x2)
    
    This gives a 2x2x2 block structure with 2x2x4 sub-problems.
    
    Alternative cleaner approach: Use the fact that 4x4x8 = (2x2x4) x (2x2x2)
    in a tensor product sense.
    
    Let me use a direct structured approach:
    Rank-161 = 23 * 7 from Laderman(3x3x3) x Strassen(2x2x2) won't work directly
    for 4x4x8.
    
    Better: Use 2x2x2 Strassen at the top level with 2x2x4 sub-problems.
    Each 2x2x4 sub-problem has rank 8 (schoolbook) or can be done better.
    
    Actually, the cleanest structured approach for 4x4x8:
    - Top level: 2x2x2 Strassen (rank 7)
    - Each sub-problem: 2x2x4 matrix multiplication
    - 2x2x4 rank = 8 (schoolbook: 2*2*4=8)
    - Total rank = 7 * 8 = 56
    
    Wait, that's not quite right either. Let me think more carefully.
    
    For 4x4x8: A is 4x4, B is 4x8, C = A*B is 4x8.
    
    Block structure: Split into 2x2 blocks where each block is 2x2 (for A),
    2x4 (for B), and 2x4 (for C).
    
    So we have a 2x2x2 block tensor where:
    - A blocks are 2x2
    - B blocks are 2x4  
    - C blocks are 2x4
    
    Using Strassen at the top: 7 products of (2x2 x 2x4) = 2x4 results.
    Each 2x2x4 multiplication has rank 8.
    Total rank = 7 * 8 = 56.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # 2x2x4 schoolbook decomposition
    # A is 2x2, B is 2x4, C is 2x4
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # U has shape (4, 8), V has shape (8, 8), W has shape (8, 8)
    U_224 = np.zeros((4, 8), dtype=np.float64)
    V_224 = np.zeros((8, 8), dtype=np.float64)
    W_224 = np.zeros((8, 8), dtype=np.float64)
    
    col = 0
    for i in range(2):
        for k in range(2):
            for j in range(4):
                U_224[i*2+k, col] = 1.0
                V_224[k*4+j, col] = 1.0
                W_224[j*2+i, col] = 1.0
                col += 1
    
    # Now build the full 4x4x8 using Strassen recursion
    # Top level: 2x2x2 with Strassen
    # Each sub-block: A is 2x2, B is 2x4, C is 2x4
    
    U_full = np.zeros((16, 56), dtype=np.float64)
    V_full = np.zeros((32, 56), dtype=np.float64)
    W_full = np.zeros((32, 56), dtype=np.float64)
    
    # Strassen block offsets for 2x2 blocks of size 2x2 (A), 2x4 (B), 2x4 (C)
    # A: 4x4 = 2x2 blocks of 2x2. Block (i,j) starts at row i*2, col j*2
    # B: 4x8 = 2x2 blocks of 2x4. Block (j,k) starts at row j*2, col k*4
    # C: 4x8 = 2x2 blocks of 2x4. Block (k,i) starts at row k*2, col i*4
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Strassen product t, we need to compute a 2x2 x 2x4 multiplication
    # The A part is a 2x2 matrix (linear combination of A blocks)
    # The B part is a 2x4 matrix (linear combination of B blocks)
    # The C part is a 2x4 matrix (linear combination of C blocks)
    
    # Strassen coefficients for A blocks (2x2 grid):
    # t: A00, A01, A10, A11
    A_coeffs = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    # Strassen coefficients for B blocks (2x2 grid, but B is 2x4 per block):
    # t: B00, B01, B10, B11
    B_coeffs = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10-B11
        [1, -1, 0, 0],  # M4: B01-B00  (note: B01-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # Strassen coefficients for C blocks (2x2 grid, C is 2x4 per block):
    # t: C00, C01, C10, C11
    C_coeffs = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [-1, 0, 0, 1],  # M3: C10-C00  (actually C00-C10? Let me recheck)
        # Actually for C: C = M1 - M3 + M4 - M5 + M6 + M7 (standard Strassen)
        # Let me use the standard Strassen C reconstruction:
        # C00 = M1 - M3 + M4 - M5 + M6
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # So C_coeffs should be:
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [-1, 0, 0, 1],  # M3: C10-C00 (negative for C00, positive for C10)
        [1, 0, 1, -1],  # M4: C00+C10-C11  (hmm, this is getting complicated)
        [-1, 0, 0, 1],  # M5: C10-C00
        [1, 0, 0, 0],   # M6: C00
        [0, 1, 0, 1],   # M7: C01+C11
    ]
    
    # Let me redo this more carefully using the standard Strassen formulation.
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M3 + M4 - M5 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So for each C block position (ci, cj), the coefficient of Mt is:
    # C00: M1=1, M2=0, M3=-1, M4=1, M5=-1, M6=1, M7=0
    # C01: M1=0, M2=0, M3=0, M4=0, M5=1, M6=0, M7=1
    # C10: M1=0, M2=1, M3=0, M4=1, M5=0, M6=0, M7=0
    # C11: M1=1, M2=-1, M3=1, M4=0, M5=0, M6=0, M7=1
    
    C_block_coeffs = [
        [1, 0, -1, 1, -1, 1, 0],   # C00
        [0, 0, 0, 0, 1, 0, 1],     # C01
        [0, 1, 0, 1, 0, 0, 0],     # C10
        [1, -1, 1, 0, 0, 0, 1],    # C11
    ]
    
    # Now build the full decomposition
    for t in range(7):
        # For each Strassen product t, we have a 2x2 x 2x4 sub-problem
        # The sub-problem uses the 2x2x4 schoolbook decomposition
        for sub_r in range(8):
            col = t * 8 + sub_r
            # U contribution: from A coefficients and U_224
            for i_local in range(2):
                for k_local in range(2):
                    u_val = U_224[i_local * 2 + k_local, sub_r]
                    if u_val != 0:
                        for a_block in range(4):
                            coeff = A_coeffs[t][a_block]
                            if coeff != 0:
                                # A block a_block is at position (a_block//2, a_block%2)
                                # Row offset: (a_block//2) * 2, Col offset: (a_block%2) * 2
                                row_off = (a_block // 2) * 2
                                col_off = (a_block % 2) * 2
                                U_full[row_off + i_local, col] += coeff * u_val
                                # Wait, U_224 index is i*2+k where i is row, k is col of the 2x2 A block
                                # So U_full row = block_row_offset + i_local, but we need to map
                                # Actually U_224[i*2+k] corresponds to A[i,k] in the 2x2 block
                                # So for A block at (br, bc), the element A[br*2+i, bc*2+k] 
                                # maps to U_224[i*2+k]
                                pass
            
            # Let me redo this more carefully
            # U_224 has shape (4, 8) where row index = i*2+k for 2x2 A block
            # For the full 4x4 A matrix, block (br, bc) has elements at rows br*2..br*2+1, cols bc*2..bc*2+1
            # Element A[br*2+i, bc*2+k] corresponds to U_224[i*2+k]
            
            # So for U_full:
            for a_block in range(4):
                coeff = A_coeffs[t][a_block]
                if coeff == 0:
                    continue
                br = a_block // 2
                bc = a_block % 2
                for i_local in range(2):
                    for k_local in range(2):
                        u_val = U_224[i_local * 2 + k_local, sub_r]
                        if u_val != 0:
                            row = br * 2 + i_local
                            col_idx = bc * 2 + k_local
                            # But U_full only has 16 rows (4x4 flattened as i*4+j)
                            # Wait, U has shape (n*m, rank) = (16, rank)
                            # Row index in U = i*m + j = i*4 + j for A[i,j]
                            # So for A block (br, bc), element (i_local, k_local):
                            # i = br*2 + i_local, j = bc*2 + k_local
                            # U row = i*4 + j = (br*2+i_local)*4 + (bc*2+k_local)
                            u_row = (br * 2 + i_local) * 4 + (bc * 2 + k_local)
                            U_full[u_row, col] += coeff * u_val
            
            # V contribution: from B coefficients and V_224
            # V_224 has shape (8, 8) where row index = k*4+j for 2x4 B block
            # For the full 4x8 B matrix, block (br, bc) has elements at rows br*2..br*2+1, cols bc*4..bc*4+3
            # Element B[br*2+k, bc*4+j] corresponds to V_224[k*4+j]
            for b_block in range(4):
                coeff = B_coeffs[t][b_block]
                if coeff == 0:
                    continue
                br = b_block // 2
                bc = b_block % 2
                for k_local in range(2):
                    for j_local in range(4):
                        v_val = V_224[k_local * 4 + j_local, sub_r]
                        if v_val != 0:
                            # B row = br*2 + k_local, B col = bc*4 + j_local
                            # V row = j*p + k = (bc*4+j_local)*4 + (br*2+k_local)
                            # Wait, V has shape (m*p, rank) = (32, rank)
                            # Row index in V = j*p + k where j is col of B, k is row of B
                            # So V row = (bc*4+j_local)*4 + (br*2+k_local)
                            v_row = (bc * 4 + j_local) * 4 + (br * 2 + k_local)
                            V_full[v_row, col] += coeff * v_val
            
            # W contribution: from C coefficients and W_224
            # W_224 has shape (8, 8) where row index = j*2+i for 2x4 C block
            # For the full 4x8 C matrix, block (br, bc) has elements at rows br*2..br*2+1, cols bc*4..bc*4+3
            # Element C[br*2+i, bc*4+j] corresponds to W_224[j*2+i]
            for c_block in range(4):
                coeff = C_block_coeffs[c_block][t]
                if coeff == 0:
                    continue
                br = c_block // 2
                bc = c_block % 2
                for j_local in range(4):
                    for i_local in range(2):
                        w_val = W_224[j_local * 2 + i_local, sub_r]
                        if w_val != 0:
                            # C row = br*2 + i_local, C col = bc*4 + j_local
                            # W row = k*n + i where k is col of C, i is row of C
                            # W has shape (p*n, rank) = (32, rank)
                            # Row index in W = k*n + i = (bc*4+j_local)*4 + (br*2+i_local)
                            w_row = (bc * 4 + j_local) * 4 + (br * 2 + i_local)
                            W_full[w_row, col] += coeff * w_val
    
    return U_full, V_full, W_full

def _prune_decomposition(U, V, W, T, deadline):
    """
    Structured pruning: try removing rank-1 terms one at a time.
    For each term r, check if T can be reconstructed without it.
    If yes, remove it. Repeat until no more terms can be removed.
    Also try merging pairs of terms.
    """
    rank = U.shape[1]
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5:
            break
        
        current_rank = current_U.shape[1]
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            # Try removing term r
            mask = [i for i in range(current_rank) if i != r]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            T_recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            if np.allclose(T_recon, T, atol=1e-9):
                # Successfully removed term r
                current_U = U_test
                current_V = V_test
                current_W = W_test
                improved = True
                break  # Restart the loop with the new rank
    
    # Try merging pairs of terms
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5:
            break
        
        current_rank = current_U.shape[1]
        for r1 in range(current_rank):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 5:
                    break
                
                # Try to see if term r2 is a linear combination of other terms
                # Simple check: if U[:,r2] is in the span of other U columns, etc.
                # This is expensive, so only do it for a few pairs
                if (r1 + r2) % 10 != 0:  # Only check 1 in 10 pairs
                    continue
                
                # Check if removing r2 still works (already handled above)
                # Try: can we replace terms r1 and r2 with a single term?
                # This is a rank-reduction via linear algebra
                # Skip for now, just do single-term removal
    
    return current_U, current_V, current_W

def _build_4x4x8_98():
    """Build 4x4x8 rank-98 decomposition via two 4x4x4 blocks (from current program)."""
    # Reuse the existing 4x4x4 rank-49 construction
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    def offset(b):
        if b == 0: return 0
        if b == 1: return 2
        if b == 2: return 8
        if b == 3: return 10
        return 0

    A_blocks = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    B_blocks = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    C_blocks = [
        [(0, 1), (3, 1), (1, -1), (2, 1)],
        [(2, 1), (3, -1)],
        [(0, -1), (1, 1)],
        [(1, -1), (3, 1)],
        [(0, 1), (1, -1), (3, 1)],
        [(2, -1), (3, 1)],
        [(0, 1), (2, -1)]
    ]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            u_vals = U2[:, s]
            v_vals = V2[:, s]
            w_vals = W2[:, s]
            
            for i_local in range(4):
                val = u_vals[i_local]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        g_idx = offset(bA) + i_local
                        U4[g_idx, col] += val * sA
            
            for j_local in range(4):
                val = v_vals[j_local]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        g_idx = offset(bB) + j_local
                        V4[g_idx, col] += val * sB
            
            for k_local in range(4):
                val = w_vals[k_local]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        g_idx = offset(bC) + k_local
                        W4[g_idx, col] += val * sC
                        
    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)
    W = np.zeros((32, 98), dtype=np.float64)
    
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            v_idx = j * 4 + k_local
            v_target = j * 8 + k_local + 4
            V[v_target, 49:98] = V4[v_idx, 0:49]
            
    for k_local in range(4):
        for i in range(4):
            w_idx = k_local * 4 + i
            w_target = (k_local + 4) * 4 + i
            W[w_target, 49:98] = W4[w_idx, 0:49]
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build the tensor
    T = _build_tensor()
    
    # Best known fallback
    U98, V98, W98 = _build_4x4x8_98()
    best_U, best_V, best_W = U98, V98, W98
    best_rank = 98
    
    if not _verify(U98, V98, W98, T):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try the Kronecker/structured 56-rank decomposition
    if time.time() < deadline - 20:
        try:
            U56, V56, W56 = _build_4x4x8_kron()
            if _verify(U56, V56, W56, T):
                best_U, best_V, best_W = U56, V56, W56
                best_rank = 56
            else:
                # Try pruning the 98-rank solution
                if time.time() < deadline - 10:
                    U_p, V_p, W_p = _prune_decomposition(U98, V98, W98, T, deadline)
                    if _verify(U_p, V_p, W_p, T) and U_p.shape[1] < best_rank:
                        best_U, best_V, best_W = U_p, V_p, W_p
                        best_rank = U_p.shape[1]
        except Exception:
            pass
    
    # Convert to half-integer representation
    U_out = _to_half_integer(best_U)
    V_out = _to_half_integer(best_V)
    W_out = _to_half_integer(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
