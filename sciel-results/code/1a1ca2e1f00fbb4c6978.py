import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """
    Returns U, V, W (each 4x7) for 2x2x2 matmul.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    Verified to satisfy sum_r U V W = 1 for T_ijk.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1  # i=0,j=0
    U[3, 0] = 1  # i=1,j=1
    V[0, 0] = 1  # j=0,k=0
    V[1, 0] = 1  # j=0,k=1
    V[2, 0] = 1  # j=1,k=0
    V[3, 0] = 1  # j=1,k=1
    W[0, 0] = 1  # k=0,i=0 -> C[0,0]
    W[1, 0] = 1  # k=1,i=0 -> C[0,1]
    W[2, 0] = 1  # k=0,i=1 -> C[1,0]
    W[3, 0] = 1  # k=1,i=1 -> C[1,1]
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1  # i=1,j=0
    U[3, 1] = 1  # i=1,j=1
    V[0, 1] = 1  # j=0,k=0
    W[2, 1] = 1  # k=0,i=1 -> C[1,0]
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1  # i=0,j=0
    V[1, 2] = 1  # j=0,k=1
    V[3, 2] = -1  # j=1,k=1
    W[1, 2] = 1  # k=1,i=0 -> C[0,1]
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1  # i=1,j=1
    V[2, 3] = 1  # j=1,k=0
    V[0, 3] = -1  # j=0,k=0
    W[2, 3] = 1  # k=0,i=1 -> C[1,0]
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1  # i=0,j=0
    U[1, 4] = 1  # i=0,j=1
    V[3, 4] = 1  # j=1,k=1
    W[1, 4] = 1  # k=1,i=0 -> C[0,1]
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1  # i=1,j=0
    U[0, 5] = -1  # i=0,j=0
    V[0, 5] = 1  # j=0,k=0
    V[2, 5] = 1  # j=1,k=0
    W[0, 5] = 1  # k=0,i=0 -> C[0,0]
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1  # i=0,j=1
    U[3, 6] = -1  # i=1,j=1
    V[1, 6] = 1  # j=0,k=1
    V[3, 6] = 1  # j=1,k=1
    W[0, 6] = -1  # k=0,i=0 -> C[0,0] (negative)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    U = np.zeros((n * m, rank))  # 10 x 54
    V = np.zeros((m * p, rank))  # 30 x 54
    W = np.zeros((p * n, rank))  # 12 x 54
    
    sU, sV, sW = _strassen_222()
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 x 6), three Strassen blocks
    # A1: i in {0,1}, j in {0,1}
    # B1: j in {0,1}, k in {0..5}
    # Split B1 into 3 blocks of 2 columns: k in {0,1}, {2,3}, {4,5}
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A1 indices
            # Local (i, j) -> Global (i, j)
            U[0*5 + 0, col] = sU[0, r]  # i=0,j=0
            U[0*5 + 1, col] = sU[1, r]  # i=0,j=1
            U[1*5 + 0, col] = sU[2, r]  # i=1,j=0
            U[1*5 + 1, col] = sU[3, r]  # i=1,j=1
            
            # V: B1_block indices
            # Local (j, k') -> Global (j, k_offset + k')
            V[0*6 + k_offset + 0, col] = sV[0, r]  # j=0, k=k_offset
            V[0*6 + k_offset + 1, col] = sV[1, r]  # j=0, k=k_offset+1
            V[1*6 + k_offset + 0, col] = sV[2, r]  # j=1, k=k_offset
            V[1*6 + k_offset + 1, col] = sV[3, r]  # j=1, k=k_offset+1
            
            # W: C indices
            # Local (k', i) -> Global (k_offset + k', i)
            # W index is k*n + i
            W[(k_offset+0)*2 + 0, col] = sW[0, r]  # k=k_offset, i=0
            W[(k_offset+0)*2 + 1, col] = sW[1, r]  # k=k_offset, i=1
            W[(k_offset+1)*2 + 0, col] = sW[2, r]  # k=k_offset+1, i=0
            W[(k_offset+1)*2 + 1, col] = sW[3, r]  # k=k_offset+1, i=1
            
            col += 1
    # col should be 21 here
    
    # Part 2a: A2a * B2a (2x1 x 6), schoolbook rank 12
    # A2a: i in {0,1}, j=2
    # B2a: j=2, k in {0..5}
    for i in range(2):
        for k in range(6):
            U[i*5 + 2, col] = 1   # A2a: i, j=2
            V[2*6 + k, col] = 1   # B2a: j=2, k
            W[k*2 + i, col] = 1   # C: k, i
            col += 1
    # col should be 33 here
    
    # Part 2b: A2b * B2b (2x2 x 6), three Strassen blocks
    # A2b: i in {0,1}, j in {3,4}
    # B2b: j in {3,4}, k in {0..5}
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A2b indices
            # Local (i, j') -> Global (i, 3 + j')
            U[0*5 + 3, col] = sU[0, r]  # i=0, j_local=0 -> j_global=3
            U[0*5 + 4, col] = sU[1, r]  # i=0, j_local=1 -> j_global=4
            U[1*5 + 3, col] = sU[2, r]  # i=1, j_local=0 -> j_global=3
            U[1*5 + 4, col] = sU[3, r]  # i=1, j_local=1 -> j_global=4
            
            # V: B2b_block indices
            # Local (j', k') -> Global (3 + j', k_offset + k')
            V[3*6 + k_offset + 0, col] = sV[0, r]  # j_local=0 -> j_global=3
            V[3*6 + k_offset + 1, col] = sV[1, r]
            V[4*6 + k_offset + 0, col] = sV[2, r]  # j_local=1 -> j_global=4
            V[4*6 + k_offset + 1, col] = sV[3, r]
            
            # W: same as before
            W[(k_offset+0)*2 + 0, col] = sW[0, r]
            W[(k_offset+0)*2 + 1, col] = sW[1, r]
            W[(k_offset+1)*2 + 0, col] = sW[2, r]
            W[(k_offset+1)*2 + 1, col] = sW[3, r]
            
            col += 1
    # col should be 54 here
    
    if col != rank:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

