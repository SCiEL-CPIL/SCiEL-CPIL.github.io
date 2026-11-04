import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    def is_valid(U, V, W, tol=1e-9):
        """Check if (U,V,W) exactly decomposes the multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+a, c] * V[a*p+j, c] * W[j*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook fallback
    def schoolbook():
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    U[i*m+a, col] = 1
                    V[a*p+j, col] = 1
                    W[j*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Strassen for 2x2x2, rank 7
    def strassen_222():
        # U rows: A11, A12, A21, A22 (i*m+a)
        # V rows: B11, B12, B21, B22 (a*p+j)
        # W rows: C11, C12, C21, C22 (j*n+i)
        
        U = np.array([
            [1, 1, 1, 0, 1, 1, 1],   # A11
            [0, 0, 1, 0, 1, 1, -1],  # A12
            [0, 0, 0, 0, 1, 0, 0],   # A21
            [-1, 1, 0, 1, 0, -1, 0], # A22
        ], dtype=float)
        
        V = np.array([
            [0, 1, -1, 0, 1, 0, 1],  # B11
            [0, 1, 0, 0, 0, 0, 1],   # B12
            [1, 0, 0, 1, 0, 1, 0],   # B21
            [-1, 1, 1, 0, 0, 1, 0],  # B22
        ], dtype=float)
        
        W = np.array([
            [0, 0, 0, 1, 1, -1, -1], # C11
            [1, 0, 0, 0, 0, 0, 1],   # C12
            [-1, 1, 1, 0, -1, 0, 0], # C21
            [1, -1, -1, -1, 0, 1, 0] # C22
        ], dtype=float)
        
        # Verify
        n_s, m_s, p_s = 2, 2, 2
        for i in range(n_s):
            for a in range(m_s):
                for j in range(p_s):
                    s = sum(U[i*m_s+a, c] * V[a*p_s+j, c] * W[j*n_s+i, c] for c in range(7))
                    if abs(s - 1) > 1e-9:
                        # Fall back to schoolbook for 2x2x2
                        U_sb = np.zeros((4, 8))
                        V_sb = np.zeros((4, 8))
                        W_sb = np.zeros((4, 8))
                        col = 0
                        for ii in range(2):
                            for aa in range(2):
                                for jj in range(2):
                                    U_sb[ii*2+aa, col] = 1
                                    V_sb[aa*2+jj, col] = 1
                                    W_sb[jj*2+ii, col] = 1
                                    col += 1
                        return U_sb, V_sb, W_sb, 8
        return U, V, W, 7
    
    # Build 2x3x7 via block decomposition
    # C[i,j] = sum_{a=0}^{2} A[i,a]*B[a,j]
    # = sum_{a=0}^{1} A[i,a]*B[a,j] + A[i,2]*B[2,j]
    
    # Part 1: 2x2x7 (A is 2x2, B is 2x7)
    # Split j into blocks of 2: (0,1), (2,3), (4,5), and single (6)
    # Three 2x2x2 blocks (Strassen, rank 7 each) + one 2x2x1 block (rank 4)
    # Total: 3*7 + 4 = 25
    
    # Part 2: 2x1x7 (A is 2x1, B is 1x7)
    # Outer product: rank 14
    
    # Total rank: 25 + 14 = 39
    
    U_s, V_s, W_s, r_s = strassen_222()
    
    # 2x2x1 schoolbook
    U_o = np.zeros((4, 4))
    V_o = np.zeros((2, 4))
    W_o = np.zeros((2, 4))
    for i in range(2):
        for a in range(2):
            for j in range(1):
                col = i*2 + a*1 + j
                U_o[i*2+a, col] = 1
                V_o[a*1+j, col] = 1
                W_o[j*2+i, col] = 1
    
    # Build 2x2x7 decomposition
    r_227 = 3 * r_s + 4
    U_227 = np.zeros((4, r_227))
    V_227 = np.zeros((14, r_227))  # 2*7
    W_227 = np.zeros((14, r_227))  # 7*2
    
    col_offset = 0
    # Three 2x2x2 blocks for j pairs (0,1), (2,3), (4,5)
    for block in range(3):
        j_start = block * 2
        for c in range(r_s):
            # U: same
            U_227[:, col_offset+c] = U_s[:, c]
            # V: rows a*7+j
            for a in range(2):
                for j_local in range(2):
                    j_global = j_start + j_local
                    V_227[a*7+j_global, col_offset+c] = V_s[a*2+j_local, c]
            # W: rows j*2+i
            for j_local in range(2):
                for i in range(2):
                    j_global = j_start + j_local
                    W_227[j_global*2+i, col_offset+c] = W_s[j_local*2+i, c]
        col_offset += r_s
    
    # One 2x2x1 block for j=6
    for c in range(4):
        U_227[:, col_offset+c] = U_o[:, c]
        for a in range(2):
            V_227[a*7+6, col_offset+c] = V_o[a*1+0, c]
        for i in range(2):
            W_227[6*2+i, col_offset+c] = W_o[0*2+i, c]
    col_offset += 4
    
    # Build 2x1x7 decomposition (outer product)
    r_217 = 14
    U_217 = np.zeros((2, r_217))
    V_217 = np.zeros((7, r_217))
    W_217 = np.zeros((14, r_217))
    
    col = 0
    for i in range(2):
        for j in range(7):
            U_217[i, col] = 1
            V_217[j, col] = 1
            W_217[j*2+i, col] = 1
            col += 1
    
    # Combine into 2x3x7
    total_rank = r_227 + r_217
    U_final = np.zeros((6, total_rank))
    V_final = np.zeros((21, total_rank))
    W_final = np.zeros((14, total_rank))
    
    # Part 1: 2x2x7 -> uses A[i,0:2] and B[0:2, j]
    # U rows 0-3 (i*m+a for i=0,1; a=0,1)
    U_final[0:4, 0:r_227] = U_227
    # V rows 0-13 (a*p+j for a=0,1; j=0..6)
    V_final[0:14, 0:r_227] = V_227
    # W rows 0-13 (j*n+i for j=0..6; i=0,1)
    W_final[0:14, 0:r_227] = W_227
    
    # Part 2: 2x1x7 -> uses A[i,2] and B[2, j]
    # U rows 4-5 (i*m+2 for i=0,1)
    U_final[4:6, r_227:r_227+r_217] = U_217
    # V rows 14-20 (2*p+j for j=0..6)
    V_final[14:21, r_227:r_227+r_217] = V_217
    # W rows 0-13 (j*n+i) - same indexing
    W_final[0:14, r_227:r_227+r_217] = W_217
    
    # Verify
    if is_valid(U_final, V_final, W_final):
        U, V, W = U_final, V_final, W_final
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
