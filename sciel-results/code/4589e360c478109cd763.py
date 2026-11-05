"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using einsum."""
    rank = len(U[0])
    U = np.array(U, dtype=float).reshape(n, m, rank)
    V = np.array(V, dtype=float).reshape(m, p, rank)
    W = np.array(W, dtype=float).reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return np.allclose(result, np.ones((n, m, p)), atol=tol)

def _strassen_2x2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    
    For A (2x2) times B (2x2), with entries:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    Strassen factors:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    c11 = M1+M4-M5+M7
    c12 = M3+M5
    c21 = M2+M4
    c22 = M1-M2+M3+M6
    
    We need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] such that
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 when (i,j,k) corresponds to
    the right product entry, 0 otherwise.
    
    For 2x2x2: n=2, m=2, p=2
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]
    W[k*2+i, c] corresponds to C[i,k] (output at row i, col k)
    """
    # Strassen factors as (U_factor, V_factor, W_factor) for each of the 7 products
    # Each factor is a tuple of 4 values (for the 4 entries)
    
    # M1 = (a11+a22)(b11+b22)
    # U: [1, 0, 0, 1] (a11 + a22)
    # V: [1, 0, 0, 1] (b11 + b22)
    # W: contributes to c11 (+1) and c22 (+1)
    # W[k*2+i]: c11 is k=0,i=0 -> W[0]; c22 is k=1,i=1 -> W[3]
    m1_U = [1, 0, 0, 1]
    m1_V = [1, 0, 0, 1]
    m1_W = [1, 0, 0, 1]  # W[0]=1 (c11), W[3]=1 (c22)
    
    # M2 = (a21+a22)b11
    # U: [0, 0, 1, 1] (a21 + a22)
    # V: [1, 0, 0, 0] (b11)
    # W: contributes to c21 (+1) and c22 (-1)
    # c21: k=0, i=1 -> W[1]; c22: k=1, i=1 -> W[3]
    m2_U = [0, 0, 1, 1]
    m2_V = [1, 0, 0, 0]
    m2_W = [0, 1, 0, -1]  # W[1]=1 (c21), W[3]=-1 (c22)
    
    # M3 = a11(b12-b22)
    # U: [1, 0, 0, 0] (a11)
    # V: [0, 1, 0, -1] (b12 - b22)
    # W: contributes to c12 (+1) and c22 (+1)
    # c12: k=1, i=0 -> W[2]; c22: k=1, i=1 -> W[3]
    m3_U = [1, 0, 0, 0]
    m3_V = [0, 1, 0, -1]
    m3_W = [0, 0, 1, 1]  # W[2]=1 (c12), W[3]=1 (c22)
    
    # M4 = a22(b21-b11)
    # U: [0, 0, 0, 1] (a22)
    # V: [0, 0, 1, 0] - wait, b21-b11. B[j,k]: b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    # V[j*2+k]: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # b21-b11 = B[1,0] - B[0,0] = V[2] - V[0]
    # V: [-1, 0, 1, 0]
    # W: contributes to c11 (+1) and c21 (+1)
    # c11: k=0, i=0 -> W[0]; c21: k=0, i=1 -> W[1]
    m4_U = [0, 0, 0, 1]
    m4_V = [-1, 0, 1, 0]
    m4_W = [1, 1, 0, 0]  # W[0]=1 (c11), W[1]=1 (c21)
    
    # M5 = (a11+a12)b22
    # U: [1, 1, 0, 0] (a11 + a12)
    # V: [0, 0, 0, 1] (b22)
    # W: contributes to c11 (-1) and c12 (+1)
    # c11: k=0, i=0 -> W[0]; c12: k=1, i=0 -> W[2]
    m5_U = [1, 1, 0, 0]
    m5_V = [0, 0, 0, 1]
    m5_W = [-1, 0, 1, 0]  # W[0]=-1 (c11), W[2]=1 (c12)
    
    # M6 = (a21-a11)(b11+b12)
    # U: [-1, 0, 1, 0] (a21 - a11)
    # V: [1, 1, 0, 0] (b11 + b12)
    # W: contributes to c22 (+1)
    # c22: k=1, i=1 -> W[3]
    m6_U = [-1, 0, 1, 0]
    m6_V = [1, 1, 0, 0]
    m6_W = [0, 0, 0, 1]  # W[3]=1 (c22)
    
    # M7 = (a12-a22)(b21+b22)
    # U: [0, 1, 0, -1] (a12 - a22)
    # V: [0, 0, 1, 1] (b21 + b22)
    # W: contributes to c11 (+1)
    # c11: k=0, i=0 -> W[0]
    m7_U = [0, 1, 0, -1]
    m7_V = [0, 0, 1, 1]
    m7_W = [1, 0, 0, 0]  # W[0]=1 (c11)
    
    factors = [
        (m1_U, m1_V, m1_W),
        (m2_U, m2_V, m2_W),
        (m3_U, m3_V, m3_W),
        (m4_U, m4_V, m4_W),
        (m5_U, m5_V, m5_W),
        (m6_U, m6_V, m6_W),
        (m7_U, m7_V, m7_W),
    ]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for c, (u_f, v_f, w_f) in enumerate(factors):
        for idx in range(4):
            U[idx][c] = u_f[idx]
            V[idx][c] = v_f[idx]
            W[idx][c] = w_f[idx]
    
    return U, V, W

def _build_3x4x2_block():
    """
    Build a rank-18 decomposition for 3x4 x 4x2 matrix multiplication.
    
    Split n=3 into n=2 (top) and n=1 (bottom).
    
    Top part: 2x4 x 4x2
    Split m=4 into m=2+2:
      A_top = [A_left | A_right] where A_left, A_right are 2x2
      B = [B_top; B_bot] where B_top, B_bot are 2x2
      C_top = A_left * B_top + A_right * B_bot
    
    Each 2x2x2 uses Strassen (rank 7), so 2*7 = 14 for the top part.
    
    Bottom part: 1x4 x 4x2
    A_bot is 1x4, B is 4x2. C_bot[i,k] = sum_j A_bot[j] * B[j,k]
    This is rank 4 (schoolbook for 1x4x2).
    
    Total: 14 + 4 = 18.
    
    Indexing:
    n=3, m=4, p=2
    U[i*m+j, c] = U[i*4+j, c] for i in 0..2, j in 0..3
    V[j*p+k, c] = V[j*2+k, c] for j in 0..3, k in 0..1
    W[k*n+i, c] = W[k*3+i, c] for k in 0..1, i in 0..2
    """
    U = [[0]*18 for _ in range(12)]  # 3*4 = 12 rows
    V = [[0]*18 for _ in range(8)]   # 4*2 = 8 rows
    W = [[0]*18 for _ in range(6)]   # 2*3 = 6 rows
    
    # --- Top part: 2x4 x 4x2, using two 2x2x2 Strassen blocks ---
    # A_top has rows i=0,1 and columns j=0,1,2,3
    # Split into A_left (j=0,1) and A_right (j=2,3)
    # B has rows j=0,1,2,3 and cols k=0,1
    # Split into B_top (j=0,1) and B_bot (j=2,3)
    
    # Strassen block 1: A_left (2x2) * B_top (2x2) -> C_top_left (2x2)
    # A_left[i, j] = A[i, j] for i in 0,1; j in 0,1
    # B_top[j, k] = B[j, k] for j in 0,1; k in 0,1
    # C_top_left[i, k] = C[i, k] for i in 0,1; k in 0,1
    
    U1, V1, W1 = _strassen_2x2x2()
    for c in range(7):
        for idx in range(4):
            # U1[idx] corresponds to A_left row idx//2, col idx%2
            i = idx // 2
            j = idx % 2
            U[i*4 + j][c] += U1[idx][c]
            
            # V1[idx] corresponds to B_top row idx//2, col idx%2
            j_v = idx // 2
            k = idx % 2
            V[j_v*2 + k][c] += V1[idx][c]
            
            # W1[idx] corresponds to C output at row idx//2, col idx%2
            # In our W[k*n+i] indexing: k is col, i is row
            i_w = idx // 2
            k_w = idx % 2
            W[k_w*3 + i_w][c] += W1[idx][c]
    
    # Strassen block 2: A_right (2x2) * B_bot (2x2) -> C_top_right (2x2)
    # A_right[i, j'] = A[i, j'+2] for i in 0,1; j' in 0,1
    # B_bot[j', k] = B[j'+2, k] for j' in 0,1; k in 0,1
    # C_top_right[i, k] = C[i, k] for i in 0,1; k in 0,1 (adds to same output)
    
    U2, V2, W2 = _strassen_2x2x2()
    base_c = 7
    for c in range(7):
        for idx in range(4):
            # U2[idx] corresponds to A_right row idx//2, col idx%2
            i = idx // 2
            j = idx % 2 + 2  # offset by 2 for right block
            U[i*4 + j][base_c + c] += U2[idx][c]
            
            # V2[idx] corresponds to B_bot row idx//2, col idx%2
            j_v = idx // 2 + 2  # offset by 2 for bottom block
            k = idx % 2
            V[j_v*2 + k][base_c + c] += V2[idx][c]
            
            # W2[idx] corresponds to C output at row idx//2, col idx%2
            i_w = idx // 2
            k_w = idx % 2
            W[k_w*3 + i_w][base_c + c] += W2[idx][c]
    
    # --- Bottom part: 1x4 x 4x2, schoolbook rank 4 ---
    # A_bot[j] = A[2, j] for j in 0..3
    # B[j, k] for j in 0..3, k in 0..1
    # C_bot[k] = C[2, k] for k in 0..1
    base_c = 14
    for j in range(4):
        for k in range(2):
            c = base_c + j*2 + k
            U[2*4 + j][c] = 1   # A[2, j]
            V[j*2 + k][c] = 1   # B[j, k]
            W[k*3 + 2][c] = 1   # C[2, k]
    
    return U, V, W

def _construct_3x4x6():
    """
    Construct the full 3x4x6 decomposition by splitting p=6 into three 2-blocks.
    
    p=6: split into k=0,1; k=2,3; k=4,5 (three blocks of size 2)
    Each block uses the 3x4x2 construction (rank 18).
    Total rank: 3 * 18 = 54.
    
    Indexing:
    n=3, m=4, p=6
    U[i*m+j, c] = U[i*4+j, c]
    V[j*p+k, c] = V[j*6+k, c]
    W[k*n+i, c] = W[k*3+i, c]
    """
    n, m, p = 3, 4, 6
    rank = 54
    U = [[0]*rank for _ in range(n*m)]  # 12 x 54
    V = [[0]*rank for _ in range(m*p)]  # 24 x 54
    W = [[0]*rank for _ in range(p*n)]  # 18 x 54
    
    U_block, V_block, W_block = _build_3x4x2_block()
    
    for block in range(3):
        base_c = block * 18
        # k offset for this block
        k_off = block * 2
        
        for c in range(18):
            # U: same for all blocks (A doesn't change)
            for i in range(n):
                for j in range(m):
                    U[i*m + j][base_c + c] = U_block[i*m + j][c]
            
            # V: V_block[j*2+k, c] -> V[j*6 + (k_off+k), base_c+c]
            for j in range(m):
                for k in range(2):
                    V[j*p + (k_off + k)][base_c + c] = V_block[j*2 + k][c]
            
            # W: W_block[k*3+i, c] -> W[(k_off+k)*3 + i, base_c+c]
            for k in range(2):
                for i in range(n):
                    W[(k_off + k)*n + i][base_c + c] = W_block[k*n + i][c]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    try:
        U, V, W = _construct_3x4x6()
        
        # Verify
        if _verify_fast(U, V, W, n, m, p, tol=1e-9):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback: schoolbook
            rank = n * m * p
            U_sb = [[0] * rank for _ in range(n * m)]
            V_sb = [[0] * rank for _ in range(m * p)]
            W_sb = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j][col] = 1
                        V_sb[j*p+k][col] = 1
                        W_sb[k*n+i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    except Exception:
        # Fallback: schoolbook
        rank = n * m * p
        U_sb = [[0] * rank for _ in range(n * m)]
        V_sb = [[0] * rank for _ in range(m * p)]
        W_sb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
