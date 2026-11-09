"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen decomposition.
    U rows: A00, A01, A10, A11
    V rows: B00, B01, B10, B11
    W rows: C00, C01, C10, C11
    """
    U = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    V = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using block Strassen (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    U row i*4+j = A[i,j]
    V row j*4+k = B[j,k]
    W row k*4+i = C[k,i]  (trace convention: W[k*n+i, r] for n=4)
    """
    u2, v2, w2 = _strassen_2x2()
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    def a_row(i, j):
        return i * 4 + j
    
    def b_row(j, k):
        return j * 4 + k
    
    def w_row(k, i):
        return k * 4 + i
    
    # 7 Strassen multiplications, each of rank 7
    # M1: (A00+A11)(B00+B11) -> C00
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B01-B11) -> C01
    # M4: A11(B10-B00) -> C00
    # M5: (A00+A01)B11 -> C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # M1: A00+A11 times B00+B11, result in C00
    col = 0
    for t in range(7):
        # A_sub[0,0] = A[0,0]+A[2,2], A_sub[0,1] = A[0,1]+A[2,3]
        # A_sub[1,0] = A[1,0]+A[3,2], A_sub[1,1] = A[1,1]+A[3,3]
        U[a_row(0,0), col+t] += u2[0, t]
        U[a_row(2,2), col+t] += u2[0, t]
        U[a_row(0,1), col+t] += u2[1, t]
        U[a_row(2,3), col+t] += u2[1, t]
        U[a_row(1,0), col+t] += u2[2, t]
        U[a_row(3,2), col+t] += u2[2, t]
        U[a_row(1,1), col+t] += u2[3, t]
        U[a_row(3,3), col+t] += u2[3, t]
        
        # B_sub[0,0] = B[0,0]+B[2,2], B_sub[0,1] = B[0,1]+B[2,3]
        # B_sub[1,0] = B[1,0]+B[3,2], B_sub[1,1] = B[1,1]+B[3,3]
        V[b_row(0,0), col+t] += v2[0, t]
        V[b_row(2,2), col+t] += v2[0, t]
        V[b_row(0,1), col+t] += v2[1, t]
        V[b_row(2,3), col+t] += v2[1, t]
        V[b_row(1,0), col+t] += v2[2, t]
        V[b_row(3,2), col+t] += v2[2, t]
        V[b_row(1,1), col+t] += v2[3, t]
        V[b_row(3,3), col+t] += v2[3, t]
        
        # C_sub[0,0] -> C[0,0], C_sub[0,1] -> C[0,1]
        # C_sub[1,0] -> C[1,0], C_sub[1,1] -> C[1,1]
        W[w_row(0,0), col+t] += w2[0, t]
        W[w_row(1,0), col+t] += w2[1, t]
        W[w_row(0,1), col+t] += w2[2, t]
        W[w_row(1,1), col+t] += w2[3, t]
    
    # M2: (A10+A11)B00, result in C10
    col = 7
    for t in range(7):
        # A_sub[0,0] = A[2,0]+A[2,2], A_sub[0,1] = A[2,1]+A[2,3]
        # A_sub[1,0] = A[3,0]+A[3,2], A_sub[1,1] = A[3,1]+A[3,3]
        U[a_row(2,0), col+t] += u2[0, t]
        U[a_row(2,2), col+t] += u2[0, t]
        U[a_row(2,1), col+t] += u2[1, t]
        U[a_row(2,3), col+t] += u2[1, t]
        U[a_row(3,0), col+t] += u2[2, t]
        U[a_row(3,2), col+t] += u2[2, t]
        U[a_row(3,1), col+t] += u2[3, t]
        U[a_row(3,3), col+t] += u2[3, t]
        
        # B_sub = B00: B[0,0], B[0,1], B[1,0], B[1,1]
        V[b_row(0,0), col+t] += v2[0, t]
        V[b_row(0,1), col+t] += v2[1, t]
        V[b_row(1,0), col+t] += v2[2, t]
        V[b_row(1,1), col+t] += v2[3, t]
        
        # C_sub[0,0] -> C[2,0], C_sub[0,1] -> C[2,1]
        # C_sub[1,0] -> C[3,0], C_sub[1,1] -> C[3,1]
        W[w_row(0,2), col+t] += w2[0, t]
        W[w_row(1,2), col+t] += w2[1, t]
        W[w_row(0,3), col+t] += w2[2, t]
        W[w_row(1,3), col+t] += w2[3, t]
    
    # M3: A00(B01-B11), result in C01
    col = 14
    for t in range(7):
        # A_sub = A00: A[0,0], A[0,1], A[1,0], A[1,1]
        U[a_row(0,0), col+t] += u2[0, t]
        U[a_row(0,1), col+t] += u2[1, t]
        U[a_row(1,0), col+t] += u2[2, t]
        U[a_row(1,1), col+t] += u2[3, t]
        
        # B_sub[0,0] = B[0,2]-B[2,2], B_sub[0,1] = B[0,3]-B[2,3]
        # B_sub[1,0] = B[1,2]-B[3,2], B_sub[1,1] = B[1,3]-B[3,3]
        V[b_row(0,2), col+t] += v2[0, t]
        V[b_row(2,2), col+t] += -v2[0, t]
        V[b_row(0,3), col+t] += v2[1, t]
        V[b_row(2,3), col+t] += -v2[1, t]
        V[b_row(1,2), col+t] += v2[2, t]
        V[b_row(3,2), col+t] += -v2[2, t]
        V[b_row(1,3), col+t] += v2[3, t]
        V[b_row(3,3), col+t] += -v2[3, t]
        
        # C_sub[0,0] -> C[0,2], C_sub[0,1] -> C[0,3]
        # C_sub[1,0] -> C[1,2], C_sub[1,1] -> C[1,3]
        W[w_row(2,0), col+t] += w2[0, t]
        W[w_row(3,0), col+t] += w2[1, t]
        W[w_row(2,1), col+t] += w2[2, t]
        W[w_row(3,1), col+t] += w2[3, t]
    
    # M4: A11(B10-B00), result in C00
    col = 21
    for t in range(7):
        # A_sub = A11: A[2,2], A[2,3], A[3,2], A[3,3]
        U[a_row(2,2), col+t] += u2[0, t]
        U[a_row(2,3), col+t] += u2[1, t]
        U[a_row(3,2), col+t] += u2[2, t]
        U[a_row(3,3), col+t] += u2[3, t]
        
        # B_sub[0,0] = B[2,0]-B[0,0], B_sub[0,1] = B[2,1]-B[0,1]
        # B_sub[1,0] = B[3,0]-B[1,0], B_sub[1,1] = B[3,1]-B[1,1]
        V[b_row(2,0), col+t] += v2[0, t]
        V[b_row(0,0), col+t] += -v2[0, t]
        V[b_row(2,1), col+t] += v2[1, t]
        V[b_row(0,1), col+t] += -v2[1, t]
        V[b_row(3,0), col+t] += v2[2, t]
        V[b_row(1,0), col+t] += -v2[2, t]
        V[b_row(3,1), col+t] += v2[3, t]
        V[b_row(1,1), col+t] += -v2[3, t]
        
        # C_sub[0,0] -> C[0,0], C_sub[0,1] -> C[0,1]
        # C_sub[1,0] -> C[1,0], C_sub[1,1] -> C[1,1]
        W[w_row(0,0), col+t] += w2[0, t]
        W[w_row(1,0), col+t] += w2[1, t]
        W[w_row(0,1), col+t] += w2[2, t]
        W[w_row(1,1), col+t] += w2[3, t]
    
    # M5: (A00+A01)B11, result in C01
    col = 28
    for t in range(7):
        # A_sub[0,0] = A[0,0]+A[0,2], A_sub[0,1] = A[0,1]+A[0,3]
        # A_sub[1,0] = A[1,0]+A[1,2], A_sub[1,1] = A[1,1]+A[1,3]
        U[a_row(0,0), col+t] += u2[0, t]
        U[a_row(0,2), col+t] += u2[0, t]
        U[a_row(0,1), col+t] += u2[1, t]
        U[a_row(0,3), col+t] += u2[1, t]
        U[a_row(1,0), col+t] += u2[2, t]
        U[a_row(1,2), col+t] += u2[2, t]
        U[a_row(1,1), col+t] += u2[3, t]
        U[a_row(1,3), col+t] += u2[3, t]
        
        # B_sub = B11: B[2,2], B[2,3], B[3,2], B[3,3]
        V[b_row(2,2), col+t] += v2[0, t]
        V[b_row(2,3), col+t] += v2[1, t]
        V[b_row(3,2), col+t] += v2[2, t]
        V[b_row(3,3), col+t] += v2[3, t]
        
        # C_sub[0,0] -> C[0,2], C_sub[0,1] -> C[0,3]
        # C_sub[1,0] -> C[1,2], C_sub[1,1] -> C[1,3]
        W[w_row(2,0), col+t] += w2[0, t]
        W[w_row(3,0), col+t] += w2[1, t]
        W[w_row(2,1), col+t] += w2[2, t]
        W[w_row(3,1), col+t] += w2[3, t]
    
    # M6: (A10-A00)(B00+B01), result in C11
    col = 35
    for t in range(7):
        # A_sub[0,0] = A[2,0]-A[0,0], A_sub[0,1] = A[2,1]-A[0,1]
        # A_sub[1,0] = A[3,0]-A[1,0], A_sub[1,1] = A[3,1]-A[1,1]
        U[a_row(2,0), col+t] += u2[0, t]
        U[a_row(0,0), col+t] += -u2[0, t]
        U[a_row(2,1), col+t] += u2[1, t]
        U[a_row(0,1), col+t] += -u2[1, t]
        U[a_row(3,0), col+t] += u2[2, t]
        U[a_row(1,0), col+t] += -u2[2, t]
        U[a_row(3,1), col+t] += u2[3, t]
        U[a_row(1,1), col+t] += -u2[3, t]
        
        # B_sub[0,0] = B[0,0]+B[0,2], B_sub[0,1] = B[0,1]+B[0,3]
        # B_sub[1,0] = B[1,0]+B[1,2], B_sub[1,1] = B[1,1]+B[1,3]
        V[b_row(0,0), col+t] += v2[0, t]
        V[b_row(0,2), col+t] += v2[0, t]
        V[b_row(0,1), col+t] += v2[1, t]
        V[b_row(0,3), col+t] += v2[1, t]
        V[b_row(1,0), col+t] += v2[2, t]
        V[b_row(1,2), col+t] += v2[2, t]
        V[b_row(1,1), col+t] += v2[3, t]
        V[b_row(1,3), col+t] += v2[3, t]
        
        # C_sub[0,0] -> C[2,2], C_sub[0,1] -> C[2,3]
        # C_sub[1,0] -> C[3,2], C_sub[1,1] -> C[3,3]
        W[w_row(2,2), col+t] += w2[0, t]
        W[w_row(3,2), col+t] += w2[1, t]
        W[w_row(2,3), col+t] += w2[2, t]
        W[w_row(3,3), col+t] += w2[3, t]
    
    # M7: (A01-A11)(B10+B11), result in C00
    col = 42
    for t in range(7):
        # A_sub[0,0] = A[0,2]-A[2,2], A_sub[0,1] = A[0,3]-A[2,3]
        # A_sub[1,0] = A[1,2]-A[3,2], A_sub[1,1] = A[1,3]-A[3,3]
        U[a_row(0,2), col+t] += u2[0, t]
        U[a_row(2,2), col+t] += -u2[0, t]
        U[a_row(0,3), col+t] += u2[1, t]
        U[a_row(2,3), col+t] += -u2[1, t]
        U[a_row(1,2), col+t] += u2[2, t]
        U[a_row(3,2), col+t] += -u2[2, t]
        U[a_row(1,3), col+t] += u2[3, t]
        U[a_row(3,3), col+t] += -u2[3, t]
        
        # B_sub[0,0] = B[2,0]+B[2,2], B_sub[0,1] = B[2,1]+B[2,3]
        # B_sub[1,0] = B[3,0]+B[3,2], B_sub[1,1] = B[3,1]+B[3,3]
        V[b_row(2,0), col+t] += v2[0, t]
        V[b_row(2,2), col+t] += v2[0, t]
        V[b_row(2,1), col+t] += v2[1, t]
        V[b_row(2,3), col+t] += v2[1, t]
        V[b_row(3,0), col+t] += v2[2, t]
        V[b_row(3,2), col+t] += v2[2, t]
        V[b_row(3,1), col+t] += v2[3, t]
        V[b_row(3,3), col+t] += v2[3, t]
        
        # C_sub[0,0] -> C[0,0], C_sub[0,1] -> C[0,1]
        # C_sub[1,0] -> C[1,0], C_sub[1,1] -> C[1,1]
        W[w_row(0,0), col+t] += w2[0, t]
        W[w_row(1,0), col+t] += w2[1, t]
        W[w_row(0,1), col+t] += w2[2, t]
        W[w_row(1,1), col+t] += w2[3, t]
    
    return U, V, W


def construct():
    """Construct a decomposition for 4x4x5 matrix multiplication.
    Returns {'U': (16, rank), 'V': (20, rank), 'W': (20, rank)}
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x1 schoolbook decomposition (rank 16)
    U1 = np.zeros((16, 16), dtype=np.int64)
    V1 = np.zeros((20, 16), dtype=np.int64)
    W1 = np.zeros((20, 16), dtype=np.int64)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                col = i * 4 + j + k * 0  # Actually, let's use a proper column index
                col = (i * 4 + j) * 1 + k * 0  # This is wrong, let me fix
    
    # Proper schoolbook for 4x4x1 (A is 4x4, B is 4x1, C is 4x1)
    # C[i] = sum_j A[i,j] * B[j,0]
    # U row i*4+j, V row j*1+0 = j, W row 0*4+i = i
    U1 = np.zeros((16, 16), dtype=np.int64)
    V1 = np.zeros((20, 16), dtype=np.int64)
    W1 = np.zeros((20, 16), dtype=np.int64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i * 4 + j, col] = 1
            V1[j, col] = 1  # B[j,0] is at row j*1+0 = j
            W1[i, col] = 1  # C[i,0] is at row 0*4+i = i
            col += 1
    
    # Combine: 4x4x4 (rank 49) + 4x4x1 (rank 16) = rank 65
    # But wait, the 5th column of B is B[:,4], and C[:,4] = A * B[:,4]
    # So we need 4x4x1 for the 5th column too
    
    # Actually, let me reconsider. For 4x4x5:
    # A is 4x4, B is 4x5, C is 4x5
    # C = A * B
    # We can split B into B1 (4x4) and B2 (4x1)
    # C1 = A * B1 (4x4), C2 = A * B2 (4x1)
    # C1 uses rank 49, C2 uses rank 16
    # Total rank: 49 + 16 = 65
    
    # But the V matrix has 20 rows (4*5), and W has 20 rows (5*4)
    # For the 4x4x4 part: V has 16 rows (4*4), W has 16 rows (4*4)
    # For the 4x4x1 part: V has 4 rows (4*1), W has 4 rows (1*4)
    # Total V rows: 16 + 4 = 20, W rows: 16 + 4 = 20
    
    # Let's build the full decomposition
    rank = 49 + 16
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((20, rank), dtype=np.int64)
    W = np.zeros((20, rank), dtype=np.int64)
    
    # 4x4x4 part (columns 0-48)
    U[:, :49] = U4
    V[:16, :49] = V4  # B[0:4, 0:4]
    W[:16, :49] = W4  # C[0:4, 0:4] in trace convention
    
    # 4x4x1 part (columns 49-64)
    U[:, 49:65] = U1
    V[16:20, 49:65] = V1  # B[0:4, 4]
    W[16:20, 49:65] = W1  # C[0:4, 4] in trace convention
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
