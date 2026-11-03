import numpy as np
import time

DEADLINE = 140.0

def _strassen_m222():
    """Return rank-7 decomposition of M_{2,2,2}.
    U (4x7), V (4x7), W (4x7) with W[k*2+i] convention.
    """
    # Strassen's 7 products for 2x2 * 2x2
    # A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U: index i*2+j for A[i,j]
    # V: index j*2+k for B[j,k]
    # W: index k*2+i for C[i,k]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1: (a11+a22)(b11+b22)
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    W[0, 0] = 1  # C[0,0] k=0,i=0
    W[3, 0] = 1  # C[1,1] k=1,i=1
    
    # M2: (a21+a22)b11
    U[2, 1] = 1  # a21
    U[3, 1] = 1  # a22
    V[0, 1] = 1  # b11
    W[2, 1] = 1  # C[1,0] k=0,i=1
    
    # M3: a11(b12-b22)
    U[0, 2] = 1  # a11
    V[1, 2] = 1  # b12
    V[3, 2] = -1 # b22
    W[1, 2] = 1  # C[0,1] k=1,i=0
    W[3, 2] = 1  # C[1,1] k=1,i=1
    
    # M4: a22(b21-b11)
    U[3, 3] = 1  # a22
    V[2, 3] = 1  # b21
    V[0, 3] = -1 # b11
    W[0, 3] = 1  # C[0,0] k=0,i=0
    W[2, 3] = 1  # C[1,0] k=0,i=1
    
    # M5: (a11+a12)b22
    U[0, 4] = 1  # a11
    U[1, 4] = 1  # a12
    V[3, 4] = 1  # b22
    W[3, 4] = -1 # C[1,1] k=1,i=1
    W[0, 4] = -1 # C[0,0] k=0,i=0
    
    # M6: (a21-a11)(b11+b12)
    U[2, 5] = 1  # a21
    U[0, 5] = -1 # a11
    V[0, 5] = 1  # b11
    V[1, 5] = 1  # b12
    W[3, 5] = 1  # C[1,1] k=1,i=1
    
    # M7: (a12-a22)(b21+b22)
    U[1, 6] = 1  # a12
    U[3, 6] = -1 # a22
    V[2, 6] = 1  # b21
    V[3, 6] = 1  # b22
    W[0, 6] = 1  # C[0,0] k=0,i=0
    W[1, 6] = 1  # C[0,1] k=1,i=0
    
    return U, V, W


def _verify(n, m, p, U, V, W):
    """Verify decomposition."""
    rank = U.shape[1]
    total = np.zeros((n, m, p), dtype=int)
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                uv = U[i * m + j, t]
                if uv == 0:
                    continue
                for k in range(p):
                    total[i, j, k] += uv * V[j * p + k, t] * W[k * n + i, t]
    return np.all(total == 1)


def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7
    
    # Build schoolbook as fallback
    schoolbook_rank = n * m * p  # 84
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=int)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=int)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build M_{3,2,2} from Strassen M_{2,2,2} + schoolbook M_{1,2,2}
    # M_{3,2,2}: A is 3x2, B is 2x2, C is 3x2
    # Split A into A_top (2x2) and A_bot (1x2)
    # C = A_top @ B + A_bot @ B
    # First part: M_{2,2,2} (Strassen, rank 7)
    # Second part: M_{1,2,2} (schoolbook, rank 4)
    
    U_str, V_str, W_str = _strassen_m222()
    
    # M_{1,2,2} schoolbook: A is 1x2, B is 2x2, C is 1x2
    # U (2x4), V (4x4), W (2x4)
    # C[k] = sum_j A[j] B[j,k]
    U_122 = np.zeros((2, 4), dtype=int)
    V_122 = np.zeros((4, 4), dtype=int)
    W_122 = np.zeros((2, 4), dtype=int)
    col = 0
    for j in range(2):
        for k in range(2):
            U_122[j, col] = 1
            V_122[j * 2 + k, col] = 1
            W_122[k * 1 + 0, col] = 1  # k*1+0, C has only i=0
            col += 1
    
    # Combine into M_{3,2,2}: rank 11
    # U (6x11): rows 0-3 for A_top (2x2), rows 4-5 for A_bot (1x2)
    # V (4x11)
    # W (6x11): W[k*3+i] for C[i,k], i=0..2, k=0..1
    r_322 = 11
    U_322 = np.zeros((6, r_322), dtype=int)
    V_322 = np.zeros((4, r_322), dtype=int)
    W_322 = np.zeros((6, r_322), dtype=int)
    
    # Part 1: Strassen for A_top (rows 0-3 of U_322)
    # U_str is 4x7, maps A[i,j] for i=0,1 j=0,1 -> index i*2+j = 0..3
    # In U_322, A_top is rows 0..3 (i=0,1; j=0,1)
    U_322[0:4, 0:7] = U_str
    V_322[0:4, 0:7] = V_str
    # W_str is 4x7, W[k*2+i] for C[i,k] with i=0,1 k=0,1
    # In W_322, C[i,k] -> k*3+i for i=0,1,2 k=0,1
    # So W_str[k*2+i] goes to W_322[k*3+i]
    for k in range(2):
        for i in range(2):
            W_322[k * 3 + i, 0:7] = W_str[k * 2 + i, 0:7]
    
    # Part 2: M_{1,2,2} for A_bot (rows 4-5 of U_322)
    # U_122 is 2x4, maps A[j] for j=0,1 -> index j (since n=1, i*m+j = 0*2+j = j)
    # In U_322, A_bot is rows 4,5 (i=2; j=0,1) -> index 2*2+j = 4+j
    U_322[4:6, 7:11] = U_122
    V_322[0:4, 7:11] = V_122
    # W_122 is 2x4, W[k*1+0] = W[k] for C[0,k]
    # In W_322, C[2,k] -> k*3+2
    for k in range(2):
        W_322[k * 3 + 2, 7:11] = W_122[k * 1 + 0, 7:11]
    
    # Verify M_{3,2,2}
    if not _verify(3, 2, 2, U_322, V_322, W_322):
        # Fallback to schoolbook
        return best
    
    # Build M_{3,2,7} from 3 copies of M_{3,2,2} + 1 copy of M_{3,2,1}
    # M_{3,2,1}: A is 3x2, B is 2x1, C is 3x1
    # Schoolbook rank = 3*2*1 = 6
    U_321 = np.zeros((6, 6), dtype=int)
    V_321 = np.zeros((2, 6), dtype=int)
    W_321 = np.zeros((3, 6), dtype=int)
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(1):
                U_321[i * 2 + j, col] = 1
                V_321[j * 1 + k, col] = 1
                W_321[k * 3 + i, col] = 1
                col += 1
    
    r_327 = 3 * r_322 + 6  # 3*11 + 6 = 39
    U_327 = np.zeros((6, r_327), dtype=int)
    V_327 = np.zeros((14, r_327), dtype=int)
    W_327 = np.zeros((21, r_327), dtype=int)
    
    offset = 0
    # Block 1: k=0,1
    for t in range(r_322):
        U_327[:, offset] = U_322[:, t]
        for j in range(2):
            for k in range(2):
                V_327[j * 7 + k, offset] = V_322[j * 2 + k, t]
        for k in range(2):
            for i in range(3):
                W_327[k * 3 + i, offset] = W_322[k * 3 + i, t]
        offset += 1
    
    # Block 2: k=2,3
    for t in range(r_322):
        U_327[:, offset] = U_322[:, t]
        for j in range(2):
            for k in range(2):
                V_327[j * 7 + (k + 2), offset] = V_322[j * 2 + k, t]
        for k in range(2):
            for i in range(3):
                W_327[(k + 2) * 3 + i, offset] = W_322[k * 3 + i, t]
        offset += 1
    
    # Block 3: k=4,5
    for t in range(r_322):
        U_327[:, offset] = U_322[:, t]
        for j in range(2):
            for k in range(2):
                V_327[j * 7 + (k + 4), offset] = V_322[j * 2 + k, t]
        for k in range(2):
            for i in range(3):
                W_327[(k + 4) * 3 + i, offset] = W_322[k * 3 + i, t]
        offset += 1
    
    # Block 4: k=6 (M_{3,2,1})
    for t in range(6):
        U_327[:, offset] = U_321[:, t]
        for j in range(2):
            V_327[j * 7 + 6, offset] = V_321[j * 1 + 0, t]
        for i in range(3):
            W_327[6 * 3 + i, offset] = W_321[0 * 3 + i, t]
        offset += 1
    
    # Verify M_{3,2,7}
    if not _verify(3, 2, 7, U_327, V_327, W_327):
        return best
    
    # Build M_{3,4,7} from two M_{3,2,7}
    # A is 3x4, split into A[:,0:2] and A[:,2:4]
    # B is 4x7, split into B[0:2,:] and B[2:4,:]
    R_total = 2 * r_327  # 78
    U_final = np.zeros((12, R_total), dtype=int)
    V_final = np.zeros((28, R_total), dtype=int)
    W_final = np.zeros((21, R_total), dtype=int)
    
    # Part 1: A[:,0:2] @ B[0:2,:]
    for t in range(r_327):
        # U_327[:, t] maps A[i,j] for i=0..2, j=0..1 -> i*2+j = 0..5
        # In U_final, A[:,0:2] is rows 0..5 (i*4+j for j=0,1)
        # A[i,0] -> i*4+0, A[i,1] -> i*4+1
        for i in range(3):
            for j in range(2):
                U_final[i * 4 + j, t] = U_327[i * 2 + j, t]
        # V_327[:, t] maps B[j,k] for j=0,1 k=0..6 -> j*7+k = 0..13
        # In V_final, B[0:2,:] is rows 0..13 (j*7+k for j=0,1)
        for j in range(2):
            for k in range(7):
                V_final[j * 7 + k, t] = V_327[j * 7 + k, t]
        # W_327[:, t] maps C[i,k] -> k*3+i = 0..20
        # In W_final, same
        for k in range(7):
            for i in range(3):
                W_final[k * 3 + i, t] = W_327[k * 3 + i, t]
    
    # Part 2: A[:,2:4] @ B[2:4,:]
    offset2 = r_327
    for t in range(r_327):
        for i in range(3):
            for j in range(2):
                U_final[i * 4 + (j + 2), offset2 + t] = U_327[i * 2 + j, t]
        for j in range(2):
            for k in range(7):
                V_final[(j + 2) * 7 + k, offset2 + t] = V_327[j * 7 + k, t]
        for k in range(7):
            for i in range(3):
                W_final[k * 3 + i, offset2 + t] = W_327[k * 3 + i, t]
    
    # Verify final
    if _verify(3, 4, 7, U_final, V_final, W_final):
        best_rank = R_total
        best = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    if time.time() > DEADLINE:
        pass
    
    return best
