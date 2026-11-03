import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_block(n, m, p):
    """Returns U, V, W for a schoolbook n x m x p block."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, n * m * p

def _strassen_222_block():
    """Returns U, V, W for Strassen 2x2x2 block (Rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1 # Wait, W indices are k*n+i.
    # C[1,0] = M2+M4. C[1,0] index in W is k=0, i=1 -> 0*2+1=1.
    # C[0,0] index in W is k=0, i=0 -> 0.
    # Let's re-verify W indices for Strassen.
    # W[k*n+i, r]. n=2.
    # C[0,0] -> W[0, r]. C[0,1] -> W[2, r]. C[1,0] -> W[1, r]. C[1,1] -> W[3, r].
    
    # Correct Strassen Coefficients for W:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M2
    
    # Let's fill W based on this.
    # M1: C00(+1), C01(+1), C10(+1), C11(+1) -> W[0]=1, W[2]=1, W[1]=1, W[3]=1
    # M2: C10(+1), C11(-1) -> W[1]=1, W[3]=-1
    # M3: C01(+1), C11(-1) -> W[2]=1, W[3]=-1
    # M4: C00(-1), C10(+1) -> W[0]=-1, W[1]=1
    # M5: C01(+1), C11(+1) -> W[2]=1, W[3]=1
    # M6: C00(+1) -> W[0]=1
    # M7: C00(-1) -> W[0]=-1
    
    # Re-initialize and fill correctly
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[2,0]=1; W[1,0]=1; W[3,0]=1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=-1
    
    # M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=-1; W[1,3]=1
    
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[2,4]=1; W[3,4]=1
    
    # M6
    U[2,5]=1; U[1,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1
    
    # M7
    U[0,6]=1; U[3,6]=1
    V[2,6]=1; V[3,6]=1
    W[0,6]=-1
    
    return U, V, W, 7

def _add_block(global_U, global_V, global_W, 
               U_block, V_block, W_block, 
               i_off, j_off, k_off, 
               n, m, p, col_start):
    """
    Adds a block decomposition to global matrices.
    i_off: offset in i (rows of A)
    j_off: offset in j (cols of A / rows of B)
    k_off: offset in k (cols of B)
    n, m, p: dimensions of the block
    col_start: starting column index in global matrices
    """
    rank_block = U_block.shape[1]
    
    for r in range(rank_block):
        for i in range(n):
            for j in range(m):
                global_U[(i + i_off) * 5 + (j + j_off), col_start + r] += U_block[i * m + j, r]
                
        for j in range(m):
            for k in range(p):
                global_V[(j + j_off) * 5 + (k + k_off), col_start + r] += V_block[j * p + k, r]
                
        for k in range(p):
            for i in range(n):
                global_W[(k + k_off) * 3 + (i + i_off), col_start + r] += W_block[k * n + i, r]
                
    return col_start + rank_block

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 5, 5
    total_rank = 0
    
    # Allocate global matrices with max possible rank (75) to be safe, then slice
    # Or just build lists of rows/cols? No, numpy is fine.
    # Max rank 75.
    global_U = np.zeros((15, 75))
    global_V = np.zeros((25, 75))
    global_W = np.zeros((15, 75))
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # n=2, m=2, p=2. i_off=0, j_off=0, k_off=0
    U1, V1, W1, r1 = _strassen_222_block()
    col = _add_block(global_U, global_V, global_W, U1, V1, W1, 0, 0, 0, 2, 2, 2, col)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # n=2, m=2, p=2. i_off=0, j_off=2, k_off=0
    U2, V2, W2, r2 = _strassen_222_block()
    col = _add_block(global_U, global_V, global_W, U2, V2, W2, 0, 2, 0, 2, 2, 2, col)
    
    # Block 3: A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]
    # n=2, m=1, p=2. i_off=0, j_off=4, k_off=0
    U3, V3, W3, r3 = _schoolbook_block(2, 1, 2)
    col = _add_block(global_U, global_V, global_W, U3, V3, W3, 0, 4, 0, 2, 1, 2, col)
    
    # Block 4: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # n=2, m=2, p=3. i_off=0, j_off=0, k_off=2
    U4, V4, W4, r4 = _schoolbook_block(2, 2, 3)
    col = _add_block(global_U, global_V, global_W, U4, V4, W4, 0, 0, 2, 2, 2, 3, col)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    # n=2, m=2, p=3. i_off=0, j_off=2, k_off=2
    U5, V5, W5, r5 = _schoolbook_block(2, 2, 3)
    col = _add_block(global_U, global_V, global_W, U5, V5, W5, 0, 2, 2, 2, 2, 3, col)
    
    # Block 6: A[0:2, 4:5] * B[4:5, 2:5] -> C[0:2, 2:5]
    # n=2, m=1, p=3. i_off=0, j_off=4, k_off=2
    U6, V6, W6, r6 = _schoolbook_block(2, 1, 3)
    col = _add_block(global_U, global_V, global_W, U6, V6, W6, 0, 4, 2, 2, 1, 3, col)
    
    # Block 7: A[2:3, 0:5] * B[0:5, 0:2] -> C[2:3, 0:2]
    # n=1, m=5, p=2. i_off=2, j_off=0, k_off=0
    U7, V7, W7, r7 = _schoolbook_block(1, 5, 2)
    col = _add_block(global_U, global_V, global_W, U7, V7, W7, 2, 0, 0, 1, 5, 2, col)
    
    # Block 8: A[2:3, 0:5] * B[0:5, 2:5] -> C[2:3, 2:5]
    # n=1, m=5, p=3. i_off=2, j_off=0, k_off=2
    U8, V8, W8, r8 = _schoolbook_block(1, 5, 3)
    col = _add_block(global_U, global_V, global_W, U8, V8, W8, 2, 0, 2, 1, 5, 3, col)
    
    # Final rank is col
    final_rank = col
    
    # Slice to final rank
    U_final = global_U[:, :final_rank]
    V_final = global_V[:, :final_rank]
    W_final = global_W[:, :final_rank]
    
    # Verification
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randint(-3, 4, size=(3, 5))
        B_test = np.random.randint(-3, 4, size=(5, 5))
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((3, 5))
        for i in range(3):
            for k in range(5):
                s = 0
                for j in range(5):
                    for r in range(final_rank):
                        s += U_final[i*5+j, r] * V_final[j*5+k, r] * W_final[k*3+i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook if verification fails
            U_f, V_f, W_f, _ = _schoolbook_block(3, 5, 5)
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
            
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
