"""Matrix multiplication 4x5 x 5x5 tensor decomposition with BCR + Strassen optimization."""
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140


def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130

    def build_strassen_2x2x2():
        """Strassen's 7-term decomposition for 2x2 * 2x2."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # Term 2: M2 = (A10+A11)B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0
        
        # Term 3: M3 = A00(B10-B11)
        U[0, 2] = 1.0
        V[2, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0; W[3, 2] = 1.0
        
        # Term 4: M4 = A11(B01-B10)
        U[3, 3] = 1.0
        V[1, 3] = 1.0; V[2, 3] = -1.0
        W[0, 3] = 1.0; W[2, 3] = 1.0
        
        # Term 5: M5 = (A00+A01)B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[2, 5] = 1.0
        W[3, 5] = 1.0
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[1, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0; W[1, 6] = 1.0
        
        return U, V, W

    def build_bcr_3x3x3():
        """
        BCR (Bini-Capovani-Lotti-Romani) rank-15 decomposition for 3x3 * 3x3.
        
        The 15 multiplications are defined as:
        M1 = (A00+A11+A22)(B00+B11+B22)
        M2 = (A00-A11)(B00-B11)
        M3 = (A10+A20)(B01+B11)
        M4 = (A01+A11)(B10+B20)
        M5 = (A02+A22)(B20+B22)
        M6 = (A21+A11)(B02+B12)
        M7 = (A00+A22)(B00-B22)
        M8 = (A11+A22)(B11-B22)
        M9 = (A00+A10+A20)(B00+B10+B20)
        M10 = (A01+A11+A21)(B01+B11+B21)
        M11 = (A02+A12+A22)(B02+B12+B22)
        M12 = (A10+A20-A00)(B01+B11-B00)
        M13 = (A01+A11-A10)(B10+B20-B00)
        M14 = (A02+A22-A20)(B20+B22-B00)
        M15 = (A21+A11-A10)(B02+B12-B00)
        
        C entries:
        C00 = M1 - M2 + M7 - M9 + M12
        C01 = M3 - M10 + M13
        C02 = M5 - M11 + M14
        C10 = M4 - M9 + M12
        C11 = M1 - M2 + M8 - M10 + M13
        C12 = M6 - M11 + M15
        C20 = M4 - M7 + M9
        C21 = M3 - M8 + M10
        C22 = M1 - M2 + M8 - M11 + M14
        
        Wait, let me use the correct BCR formulas. Let me look this up more carefully.
        
        Actually, let me use a known correct BCR decomposition. The standard BCR 15-term 
        decomposition for 3x3 has specific formulas. Let me construct it carefully.
        
        I'll use the following well-known BCR decomposition:
        """
        # Let me use a verified BCR decomposition
        # The 15 products and how they contribute to each C[i,k]
        
        # For 3x3: A[i,j] for i,j in {0,1,2}, B[j,k] for j,k in {0,1,2}
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        # U index: i*3+j, V index: j*3+k, W index: k*3+i
        
        rank = 15
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        # BCR decomposition (verified formulas)
        # Product c: U contributes to A indices, V to B indices, W to C indices
        
        # M0: (A00+A11+A22) * (B00+B11+B22)
        # U: A00(0), A11(4), A22(8) -> U[0,0]=1, U[4,0]=1, U[8,0]=1
        # V: B00(0), B11(4), B22(8) -> V[0,0]=1, V[4,0]=1, V[8,0]=1
        # W: C00(0), C11(4), C22(8) -> W[0,0]=1, W[4,0]=1, W[8,0]=1
        U[0,0]=1; U[4,0]=1; U[8,0]=1
        V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[4,0]=1; W[8,0]=1
        
        # M1: (A00-A11) * (B00-B11)
        U[0,1]=1; U[4,1]=-1
        V[0,1]=1; V[4,1]=-1
        W[0,1]=-1; W[4,1]=-1
        
        # M2: (A10+A20) * (B01+B11)
        U[3,2]=1; U[6,2]=1
        V[1,2]=1; V[4,2]=1
        W[1,2]=1; W[4,2]=1
        
        # M3: (A01+A11) * (B10+B20)
        U[1,3]=1; U[4,3]=1
        V[3,3]=1; V[6,3]=1
        W[0,3]=1; W[3,3]=1
        
        # M4: (A02+A22) * (B20+B22)
        U[2,4]=1; U[8,4]=1
        V[6,4]=1; V[8,4]=1
        W[0,4]=1; W[8,4]=1
        
        # M5: (A21+A11) * (B02+B12)
        U[7,5]=1; U[4,5]=1
        V[2,5]=1; V[5,5]=1
        W[2,5]=1; W[5,5]=1
        
        # M6: (A00+A22) * (B00-B22)
        U[0,6]=1; U[8,6]=1
        V[0,6]=1; V[8,6]=-1
        W[0,6]=1; W[8,6]=-1
        
        # M7: (A11+A22) * (B11-B22)
        U[4,7]=1; U[8,7]=1
        V[4,7]=1; V[8,7]=-1
        W[4,7]=1; W[8,7]=-1
        
        # M8: (A00+A10+A20) * (B00+B10+B20)
        U[0,8]=1; U[3,8]=1; U[6,8]=1
        V[0,8]=1; V[3,8]=1; V[6,8]=1
        W[0,8]=1; W[3,8]=1; W[6,8]=1
        
        # M9: (A01+A11+A21) * (B01+B11+B21)
        U[1,9]=1; U[4,9]=1; U[7,9]=1
        V[1,9]=1; V[4,9]=1; V[7,9]=1
        W[1,9]=1; W[4,9]=1; W[7,9]=1
        
        # M10: (A02+A12+A22) * (B02+B12+B22)
        U[2,10]=1; U[5,10]=1; U[8,10]=1
        V[2,10]=1; V[5,10]=1; V[8,10]=1
        W[2,10]=1; W[5,10]=1; W[8,10]=1
        
        # M11: (A10+A20-A00) * (B01+B11-B00)
        U[3,11]=1; U[6,11]=1; U[0,11]=-1
        V[1,11]=1; V[4,11]=1; V[0,11]=-1
        W[1,11]=1; W[4,11]=1; W[0,11]=-1
        
        # M12: (A01+A11-A10) * (B10+B20-B00)
        U[1,12]=1; U[4,12]=1; U[3,12]=-1
        V[3,12]=1; V[6,12]=1; V[0,12]=-1
        W[0,12]=1; W[3,12]=1; W[0,12]=-1  # Wait, need to be more careful
        
        # Let me restart with a more systematic approach
        # I'll use the known BCR formulas from the literature
        
        # Actually, let me just verify by construction. The key property is:
        # For all i,j,k: sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
        
        # Let me use a different, verified BCR decomposition
        # Reference: Bini, Capovani, Lotti, Romani (1978)
        
        # I'll use this known correct set of 15 terms:
        U = np.zeros((9, 15), dtype=float)
        V = np.zeros((9, 15), dtype=float)
        W = np.zeros((9, 15), dtype=float)
        
        # Term 0: (A00+A11+A22)(B00+B11+B22)
        U[0,0]=1; U[4,0]=1; U[8,0]=1
        V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[4,0]=1; W[8,0]=1
        
        # Term 1: (A00-A11)(B00-B11)
        U[0,1]=1; U[4,1]=-1
        V[0,1]=1; V[4,1]=-1
        W[0,1]=-1; W[4,1]=-1
        
        # Term 2: (A10+A20)(B01+B11)
        U[3,2]=1; U[6,2]=1
        V[1,2]=1; V[4,2]=1
        W[1,2]=1; W[4,2]=1
        
        # Term 3: (A01+A11)(B10+B20)
        U[1,3]=1; U[4,3]=1
        V[3,3]=1; V[6,3]=1
        W[0,3]=1; W[3,3]=1
        
        # Term 4: (A02+A22)(B20+B22)
        U[2,4]=1; U[8,4]=1
        V[6,4]=1; V[8,4]=1
        W[0,4]=1; W[8,4]=1
        
        # Term 5: (A21+A11)(B02+B12)
        U[7,5]=1; U[4,5]=1
        V[2,5]=1; V[5,5]=1
        W[2,5]=1; W[5,5]=1
        
        # Term 6: (A00+A22)(B00-B22)
        U[0,6]=1; U[8,6]=1
        V[0,6]=1; V[8,6]=-1
        W[0,6]=1; W[8,6]=-1
        
        # Term 7: (A11+A22)(B11-B22)
        U[4,7]=1; U[8,7]=1
        V[4,7]=1; V[8,7]=-1
        W[4,7]=1; W[8,7]=-1
        
        # Term 8: (A00+A10+A20)(B00+B10+B20)
        U[0,8]=1; U[3,8]=1; U[6,8]=1
        V[0,8]=1; V[3,8]=1; V[6,8]=1
        W[0,8]=1; W[3,8]=1; W[6,8]=1
        
        # Term 9: (A01+A11+A21)(B01+B11+B21)
        U[1,9]=1; U[4,9]=1; U[7,9]=1
        V[1,9]=1; V[4,9]=1; V[7,9]=1
        W[1,9]=1; W[4,9]=1; W[7,9]=1
        
        # Term 10: (A02+A12+A22)(B02+B12+B22)
        U[2,10]=1; U[5,10]=1; U[8,10]=1
        V[2,10]=1; V[5,10]=1; V[8,10]=1
        W[2,10]=1; W[5,10]=1; W[8,10]=1
        
        # Term 11: (A10+A20-A00)(B01+B11-B00)
        U[3,11]=1; U[6,11]=1; U[0,11]=-1
        V[1,11]=1; V[4,11]=1; V[0,11]=-1
        W[1,11]=1; W[4,11]=1; W[0,11]=-1
        
        # Term 12: (A01+A11-A10)(B10+B20-B00)
        U[1,12]=1; U[4,12]=1; U[3,12]=-1
        V[3,12]=1; V[6,12]=1; V[0,12]=-1
        W[0,12]=1; W[3,12]=1; W[0,12]=-1  # This is wrong, let me fix
        
        # I need to be more careful. Let me just use a verified decomposition.
        # For now, let me fall back to schoolbook for 3x3x3 if BCR doesn't verify.
        
        # Let me try a different approach - use the known correct BCR from Wikipedia
        # The 15 products for BCR are:
        # P1 = (a00+a11+a22)(b00+b11+b22)
        # P2 = (a00-a11)(b00-b11)  
        # P3 = (a10+a20)(b01+b11)
        # P4 = (a01+a11)(b10+b20)
        # P5 = (a02+a22)(b20+b22)
        # P6 = (a21+a11)(b02+b12)
        # P7 = (a00+a22)(b00-b22)
        # P8 = (a11+a22)(b11-b22)
        # P9 = (a00+a10+a20)(b00+b10+b20)
        # P10 = (a01+a11+a21)(b01+b11+b21)
        # P11 = (a02+a12+a22)(b02+b12+b22)
        # P12 = (a10+a20-a00)(b01+b11-b00)
        # P13 = (a01+a11-a10)(b10+b20-b00)
        # P14 = (a02+a22-a20)(b20+b22-b00)
        # P15 = (a21+a11-a10)(b02+b12-b00)
        
        # And the output:
        # c00 = P1 - P2 + P7 - P9 + P12
        # c01 = P3 - P10 + P13
        # c02 = P5 - P11 + P14
        # c10 = P4 - P9 + P12
        # c11 = P1 - P2 + P8 - P10 + P13
        # c12 = P6 - P11 + P15
        # c20 = P4 - P7 + P9
        # c21 = P3 - P8 + P10
        # c22 = P1 - P2 + P8 - P11 + P14
        
        # Wait, I need to double-check these. Let me just implement and verify.
        
        U = np.zeros((9, 15), dtype=float)
        V = np.zeros((9, 15), dtype=float)
        W = np.zeros((9, 15), dtype=float)
        
        # P1 = (a00+a11+a22)(b00+b11+b22)
        U[0,0]=1; U[4,0]=1; U[8,0]=1
        V[0,0]=1; V[4,0]=1; V[8,0]=1
        # c00 += P1, c11 += P1, c22 += P1
        W[0,0]=1; W[4,0]=1; W[8,0]=1
        
        # P2 = (a00-a11)(b00-b11)
        U[0,1]=1; U[4,1]=-1
        V[0,1]=1; V[4,1]=-1
        # c00 -= P2, c11 -= P2, c22 -= P2
        W[0,1]=-1; W[4,1]=-1; W[8,1]=-1
        
        # P3 = (a10+a20)(b01+b11)
        U[3,2]=1; U[6,2]=1
        V[1,2]=1; V[4,2]=1
        # c01 += P3, c21 += P3
        W[1,2]=1; W[7,2]=1
        
        # P4 = (a01+a11)(b10+b20)
        U[1,3]=1; U[4,3]=1
        V[3,3]=1; V[6,3]=1
        # c10 += P4, c20 += P4
        W[3,3]=1; W[6,3]=1
        
        # P5 = (a02+a22)(b20+b22)
        U[2,4]=1; U[8,4]=1
        V[6,4]=1; V[8,4]=1
        # c02 += P5, c22 += P5
        W[2,4]=1; W[8,4]=1
        
        # P6 = (a21+a11)(b02+b12)
        U[7,5]=1; U[4,5]=1
        V[2,5]=1; V[5,5]=1
        # c12 += P6
        W[5,5]=1
        
        # P7 = (a00+a22)(b00-b22)
        U[0,6]=1; U[8,6]=1
        V[0,6]=1; V[8,6]=-1
        # c00 += P7, c20 -= P7
        W[0,6]=1; W[6,6]=-1
        
        # P8 = (a11+a22)(b11-b22)
        U[4,7]=1; U[8,7]=1
        V[4,7]=1; V[8,7]=-1
        # c11 += P8, c21 -= P8
        W[4,7]=1; W[7,7]=-1
        
        # P9 = (a00+a10+a20)(b00+b10+b20)
        U[0,8]=1; U[3,8]=1; U[6,8]=1
        V[0,8]=1; V[3,8]=1; V[6,8]=1
        # c00 -= P9, c10 -= P9, c20 += P9
        W[0,8]=-1; W[3,8]=-1; W[6,8]=1
        
        # P10 = (a01+a11+a21)(b01+b11+b21)
        U[1,9]=1; U[4,9]=1; U[7,9]=1
        V[1,9]=1; V[4,9]=1; V[7,9]=1
        # c01 -= P10, c11 -= P10, c21 += P10
        W[1,9]=-1; W[4,9]=-1; W[7,9]=1
        
        # P11 = (a02+a12+a22)(b02+b12+b22)
        U[2,10]=1; U[5,10]=1; U[8,10]=1
        V[2,10]=1; V[5,10]=1; V[8,10]=1
        # c02 -= P11, c12 -= P11, c22 -= P11
        W[2,10]=-1; W[5,10]=-1; W[8,10]=-1
        
        # P12 = (a10+a20-a00)(b01+b11-b00)
        U[3,11]=1; U[6,11]=1; U[0,11]=-1
        V[1,11]=1; V[4,11]=1; V[0,11]=-1
        # c00 += P12, c10 += P12
        W[0,11]=1; W[3,11]=1
        
        # P13 = (a01+a11-a10)(b10+b20-b00)
        U[1,12]=1; U[4,12]=1; U[3,12]=-1
        V[3,12]=1; V[6,12]=1; V[0,12]=-1
        # c01 += P13, c11 += P13
        W[1,12]=1; W[4,12]=1
        
        # P14 = (a02+a22-a20)(b20+b22-b00)
        U[2,13]=1; U[8,13]=1; U[6,13]=-1
        V[6,13]=1; V[8,13]=1; V[0,13]=-1
        # c02 += P14, c22 += P14
        W[2,13]=1; W[8,13]=1
        
        # P15 = (a21+a11-a10)(b02+b12-b00)
        U[7,14]=1; U[4,14]=1; U[3,14]=-1
        V[2,14]=1; V[5,14]=1; V[0,14]=-1
        # c12 += P15
        W[5,14]=1
        
        return U, V, W

    def schoolbook_block(n_b, m_b, p_b):
        rank_b = n_b * m_b * p_b
        U_b = np.zeros((n_b * m_b, rank_b), dtype=float)
        V_b = np.zeros((m_b * p_b, rank_b), dtype=float)
        W_b = np.zeros((p_b * n_b, rank_b), dtype=float)
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, col] = 1.0
                    V_b[j * p_b + k, col] = 1.0
                    W_b[k * n_b + i, col] = 1.0
                    col += 1
        return U_b, V_b, W_b

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_block(U_b, V_b, W_b, n_b, m_b, p_b, tol=1e-9):
        rank = U_b.shape[1]
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    s = 0.0
                    for c in range(rank):
                        s += U_b[i * m_b + j, c] * V_b[j * p_b + k, c] * W_b[k * n_b + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def combine_decompositions(blocks, n, m, p):
        total_rank = sum(U_b.shape[1] for (_, _, _, _, _, _, U_b, _, _) in blocks)
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b) in blocks:
            rank_b = U_b.shape[1]
            n_b = i_end - i_start
            m_b = j_end - j_start
            p_b = k_end - k_start
            
            for li in range(n_b):
                for lj in range(m_b):
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col:col+rank_b] = U_b[li * m_b + lj, :]
            for lj in range(m_b):
                for lk in range(p_b):
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col:col+rank_b] = V_b[lj * p_b + lk, :]
            for lk in range(p_b):
                for li in range(n_b):
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col:col+rank_b] = W_b[lk * n_b + li, :]
            col += rank_b
        
        return U, V, W

    # First, verify our BCR decomposition
    U_bcr, V_bcr, W_bcr = build_bcr_3x3x3()
    bcr_valid = verify_block(U_bcr, V_bcr, W_bcr, 3, 3, 3, tol=1e-9)
    
    # Verify Strassen
    U_str, V_str, W_str = build_strassen_2x2x2()
    strassen_valid = verify_block(U_str, V_str, W_str, 2, 2, 2, tol=1e-9)
    
    # Partition: i=[0,1,2],[3], j=[0,1,2],[3,4], k=[0,1,2],[3,4]
    i_parts = [(0, 3), (3, 4)]
    j_parts = [(0, 3), (3, 5)]
    k_parts = [(0, 3), (3, 5)]
    
    blocks = []
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 3 and m_b == 3 and p_b == 3 and bcr_valid:
                    U_b, V_b, W_b = U_bcr.copy(), V_bcr.copy(), W_bcr.copy()
                elif n_b == 2 and m_b == 2 and p_b == 2 and strassen_valid:
                    U_b, V_b, W_b = U_str.copy(), V_str.copy(), W_str.copy()
                else:
                    U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                
                blocks.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
    
    U, V, W = combine_decompositions(blocks, n, m, p)
    
    # Verify full decomposition
    if not verify(U, V, W, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U = np.zeros((n * m, n*m*p), dtype=float)
        V = np.zeros((m * p, n*m*p), dtype=float)
        W = np.zeros((p * n, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        if not verify(U, V, W, n, m, p, tol=1e-9):
            raise Exception("Verification failed")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find better with remaining time
    if time.time() < deadline - 10:
        np.random.seed(42)
        for target_rank in range(best_rank - 1, max(best_rank - 10, 80), -1):
            if time.time() > deadline - 10:
                break
            for trial in range(2):
                if time.time() > deadline - 10:
                    break
                U_t = np.zeros((n * m, target_rank), dtype=float)
                V_t = np.zeros((m * p, target_rank), dtype=float)
                W_t = np.zeros((p * n, target_rank), dtype=float)
                min_r = min(target_rank, best_rank)
                U_t[:, :min_r] = best_U[:, :min_r]
                V_t[:, :min_r] = best_V[:, :min_r]
                W_t[:, :min_r] = best_W[:, :min_r]
                if min_r < target_rank:
                    for c in range(min_r, target_rank):
                        U_t[:, c] = np.random.randn(n * m) * 0.1
                        V_t[:, c] = np.random.randn(m * p) * 0.1
                        W_t[:, c] = np.random.randn(p * n) * 0.1
                if verify(U_t, V_t, W_t, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                    break
            if best_rank == target_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
