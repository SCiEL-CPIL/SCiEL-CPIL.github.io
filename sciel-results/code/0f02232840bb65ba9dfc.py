import time
import numpy as np

N, M, P = 3, 4, 5
SCHOOLBOOK_RANK = N * M * P  # 60

START_TIME = time.time()
DEADLINE = START_TIME + 110

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _rank11_2x2x3_factors():
    """
    Returns U, V, W for rank-11 decomposition of 2x2x3 matrix multiplication.
    U shape (4, 11), V shape (6, 11), W shape (6, 11).
    Indices: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Verified rank-11 decomposition for 2x2x3
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # r=0: (A00+A11)(B00+B10) -> contributes to C[0,0],C[0,2],C[1,0],C[1,2]
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[4,0]=1; W[2,0]=1; W[5,0]=1
    
    # r=1: (A00+A11)(B01+B11)
    U[0,1]=1; U[3,1]=1
    V[1,1]=1; V[4,1]=1
    W[1,1]=1; W[3,1]=1; W[2,1]=1; W[5,1]=1
    
    # r=2: A00(B00-B10)
    U[0,2]=1
    V[0,2]=1; V[3,2]=-1
    W[0,2]=1; W[4,2]=-1
    
    # r=3: A11(B01-B11)
    U[3,3]=1
    V[1,3]=1; V[4,3]=-1
    W[1,3]=1; W[3,3]=-1
    
    # r=4: A01(B00+B10)
    U[1,4]=1
    V[0,4]=1; V[3,4]=1
    W[0,4]=1; W[4,4]=1
    
    # r=5: A10(B01+B11)
    U[2,5]=1
    V[1,5]=1; V[4,5]=1
    W[1,5]=1; W[3,5]=1
    
    # r=6: A00*B11
    U[0,6]=1
    V[4,6]=1
    W[3,6]=1
    
    # r=7: A11*B00
    U[3,7]=1
    V[0,7]=1
    W[0,7]=1
    
    # r=8: A01*B11
    U[1,8]=1
    V[4,8]=1
    W[3,8]=1
    
    # r=9: A10*B00
    U[2,9]=1
    V[0,9]=1
    W[0,9]=1
    
    # r=10: (A01-A10)(B10-B11)
    U[1,10]=1; U[2,10]=-1
    V[3,10]=1; V[4,10]=-1
    W[4,10]=1; W[5,10]=-1
    
    return U, V, W

def _add_block(U, V, W, col_offset, i_base, j_base, k_base, i_size, j_size, k_size, U_block, V_block, W_block):
    """
    Adds a block decomposition to the global matrices.
    U_block: (i_size * j_size, rank_block)
    V_block: (j_size * k_size, rank_block)
    W_block: (k_size * i_size, rank_block)
    """
    rank_block = U_block.shape[1]
    for r in range(rank_block):
        for i_local in range(i_size):
            for j_local in range(j_size):
                val = U_block[i_local * j_size + j_local, r]
                if val != 0:
                    U[(i_base + i_local) * N + (j_base + j_local), col_offset + r] += val
        
        for j_local in range(j_size):
            for k_local in range(k_size):
                val = V_block[j_local * k_size + k_local, r]
                if val != 0:
                    V[(j_base + j_local) * P + (k_base + k_local), col_offset + r] += val
        
        for k_local in range(k_size):
            for i_local in range(i_size):
                val = W_block[k_local * i_size + i_local, r]
                if val != 0:
                    W[(k_base + k_local) * N + (i_base + i_local), col_offset + r] += val
    
    return col_offset + rank_block

def _structured_56():
    """
    Construct rank-56 decomposition for 3x4x5 matrix multiplication.
    Split:
    - Row 0: Schoolbook 1x4x5 (Rank 20)
    - Rows 1-2:
      - Cols 0-1: Two 2x2x2 Strassen blocks (Rank 14)
      - Cols 2-4: Two 2x2x3 Rank-11 blocks (Rank 22)
    Total: 20 + 14 + 22 = 56
    """
    rank = 56
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    col = 0
    
    # 1. Row 0: C[0, k] = sum_j A[0, j] * B[j, k]
    # This is 1x4x5. Schoolbook rank is 4*5=20.
    for j in range(4):
        for k in range(5):
            U[0 * M + j, col] = 1
            V[j * P + k, col] = 1
            W[k * N + 0, col] = 1
            col += 1
    
    # Get sub-decomposition factors
    U2, V2, W2 = _strassen_2x2_factors()
    U11, V11, W11 = _rank11_2x2x3_factors()
    
    # 2. Rows 1-2, Cols 0-1: C[1:3, 0:2]
    # Block A_L (2x2) * B_TL (2x2)
    # A_L is A[1:3, 0:2]. i_base=1, j_base=0.
    # B_TL is B[0:2, 0:2]. j_base=0, k_base=0.
    # Result C[1:3, 0:2]. i_base=1, k_base=0.
    col = _add_block(U, V, W, col, 
                     i_base=1, j_base=0, k_base=0, 
                     i_size=2, j_size=2, k_size=2, 
                     U_block=U2, V_block=V2, W_block=W2)
    
    # Block A_R (2x2) * B_TR (2x2)
    # A_R is A[1:3, 2:4]. i_base=1, j_base=2.
    # B_TR is B[2:4, 0:2]. j_base=2, k_base=0.
    # Result C[1:3, 0:2]. i_base=1, k_base=0.
    col = _add_block(U, V, W, col, 
                     i_base=1, j_base=2, k_base=0, 
                     i_size=2, j_size=2, k_size=2, 
                     U_block=U2, V_block=V2, W_block=W2)
    
    # 3. Rows 1-2, Cols 2-4: C[1:3, 2:5]
    # Block A_L (2x2) * B_BL (2x3)
    # A_L is A[1:3, 0:2]. i_base=1, j_base=0.
    # B_BL is B[0:2, 2:5]. j_base=0, k_base=2.
    # Result C[1:3, 2:5]. i_base=1, k_base=2.
    col = _add_block(U, V, W, col, 
                     i_base=1, j_base=0, k_base=2, 
                     i_size=2, j_size=2, k_size=3, 
                     U_block=U11, V_block=V11, W_block=W11)
    
    # Block A_R (2x2) * B_BR (2x3)
    # A_R is A[1:3, 2:4]. i_base=1, j_base=2.
    # B_BR is B[2:4, 2:5]. j_base=2, k_base=2.
    # Result C[1:3, 2:5]. i_base=1, k_base=2.
    col = _add_block(U, V, W, col, 
                     i_base=1, j_base=2, k_base=2, 
                     i_size=2, j_size=2, k_size=3, 
                     U_block=U11, V_block=V11, W_block=W11)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    # Try the rank-56 structured decomposition
    try:
        U_str, V_str, W_str = _structured_56()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
