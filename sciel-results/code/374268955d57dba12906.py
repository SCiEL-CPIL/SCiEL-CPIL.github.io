import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split m into [2, 2, 1] and p into [2, 2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Total rank
    rank = 0
    for ms in m_splits:
        for ps in p_splits:
            if ms == 2 and ps == 2:
                rank += 7
            else:
                # ms=1, ps=2 -> rank 4
                rank += 4
                
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_offset = 0
    for ms in m_splits:
        k_offset = 0
        for ps in p_splits:
            if ms == 2 and ps == 2:
                # Strassen block
                # i in {0, 1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
                # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
                # Strassen coefficients
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # We define the terms (u, v, w) for the 7 multiplications
                # u is 2x2, v is 2x2, w is 2x2? No, w is for C.
                # The term is sum u_ij v_jk w_ik? No.
                # The term is u_ij A_ij * v_jk B_jk * w_ik?
                # The output C_ik is sum_r w_ik,r * (sum_j u_ij,r A_ij) * (sum_j v_jk,r B_jk)?
                # No, the standard bilinear form is:
                # C_ik = sum_r w_ik,r * P_r * Q_r
                # where P_r = sum_j u_ij,r A_ij and Q_r = sum_j v_jk,r B_jk?
                # This is not quite right for Strassen.
                # Strassen computes M_r = P_r Q_r.
                # Then C_ik = sum_r c_ik,r M_r.
                # So w_ik,r = c_ik,r.
                # u_ij,r and v_jk,r are the coefficients for P_r and Q_r.
                
                # M1: P1 = A00+A11, Q1 = B00+B11. C00+=M1, C11+=M1
                # u: [[1,0],[0,1]], v: [[1,0],[0,1]], w: [[1,0],[0,1]]
                # M2: P2 = A10+A11, Q2 = B00. C10+=M2, C11-=M2
                # u: [[0,0],[1,1]], v: [[1,0],[0,0]], w: [[0,0],[1,-1]]
                # M3: P3 = A00, Q3 = B01-B11. C01+=M3, C11+=M3
                # u: [[1,0],[0,0]], v: [[0,1],[0,-1]], w: [[0,1],[0,1]]
                # M4: P4 = A11, Q4 = B10-B00. C00+=M4, C10+=M4
                # u: [[0,0],[0,1]], v: [[-1,0],[0,1]], w: [[1,0],[1,0]]
                # M5: P5 = A00+A01, Q5 = B11. C00-=M5, C01+=M5
                # u: [[1,1],[0,0]], v: [[0,0],[0,1]], w: [[-1,0],[0,1]]
                # M6: P6 = A10-A00, Q6 = B00+B01. C11+=M6
                # u: [[-1,0],[1,0]], v: [[1,1],[0,0]], w: [[0,0],[0,1]]
                # M7: P7 = A01-A11, Q7 = B10+B11. C00+=M7
                # u: [[0,1],[0,-1]], v: [[0,1],[0,1]], w: [[1,0],[0,0]]
                
                terms = [
                    ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
                    ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),
                    ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,1]]),
                    ([[0,0],[0,1]], [[-1,0],[0,1]], [[1,0],[1,0]]),
                    ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]),
                    ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
                    ([[0,1],[0,-1]], [[0,1],[0,1]], [[1,0],[0,0]])
                ]
                
                for u, v, w in terms:
                    for i_l in range(2):
                        for j_l in range(2):
                            if u[i_l][j_l] != 0:
                                i = i_l
                                j = j_offset + j_l
                                U[i * m + j][col] = u[i_l][j_l]
                    for j_l in range(2):
                        for k_l in range(2):
                            if v[j_l][k_l] != 0:
                                j = j_offset + j_l
                                k = k_offset + k_l
                                V[j * p + k][col] = v[j_l][k_l]
                    for k_l in range(2):
                        for i_l in range(2):
                            if w[i_l][k_l] != 0:
                                k = k_offset + k_l
                                i = i_l
                                W[k * n + i][col] = w[i_l][k_l]
                    col += 1
            else:
                # ms=1, ps=2. Rank 4.
                # C_ik = A_i0 B_0k.
                # Terms:
                # 1. i=0, k=0: A00 B00
                # 2. i=0, k=1: A00 B01
                # 3. i=1, k=0: A10 B00
                # 4. i=1, k=1: A10 B01
                # u: 2x1, v: 1x2, w: 2x2
                terms = [
                    ([[1],[0]], [[1,0]], [[1,0],[0,0]]),
                    ([[1],[0]], [[0,1]], [[0,1],[0,0]]),
                    ([[0],[1]], [[1,0]], [[0,0],[1,0]]),
                    ([[0],[1]], [[0,1]], [[0,0],[0,1]])
                ]
                
                for u, v, w in terms:
                    for i_l in range(2):
                        j_l = 0
                        if u[i_l][j_l] != 0:
                            i = i_l
                            j = j_offset + j_l
                            U[i * m + j][col] = u[i_l][j_l]
                    j_l = 0
                    for k_l in range(2):
                        if v[j_l][k_l] != 0:
                            j = j_offset + j_l
                            k = k_offset + k_l
                            V[j * p + k][col] = v[j_l][k_l]
                    for k_l in range(2):
                        for i_l in range(2):
                            if w[i_l][k_l] != 0:
                                k = k_offset + k_l
                                i = i_l
                                W[k * n + i][col] = w[i_l][k_l]
                    col += 1
            k_offset += ps
        j_offset += ms
        
    return {"U": U, "V": V, "W": W}
