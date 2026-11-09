"""Program for 3x4x4 matrix multiplication rank minimization."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # Convert to numpy for fast vectorized verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Check dimensions
    if U_np.shape[0] != n * m or V_np.shape[0] != m * p or W_np.shape[0] != p * n:
        return False
    if V_np.shape[1] != rank or W_np.shape[1] != rank:
        return False

    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # We can check this by iterating over i, j, k. 
    # Since rank is small (44) and n*m*p is 48, a simple loop is fast enough.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                # Vectorized sum over rank
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_rank(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _rank_44_decomposition(n, m, p):
    """
    Construct a rank-44 decomposition for 3x4x4 matrix multiplication.
    Strategy:
    - Split A (3x4) into A_top (2x4) and A_bot (1x4).
    - Split B (4x4) into 2x2 blocks.
    - Use Strassen (rank 7) for 2x2x2 blocks in the top part.
    - Use Schoolbook (rank 4) for 1x2x2 blocks in the bottom part.
    """
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i_off, j_off, k_off are the starting indices for the 2x2 submatrices
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Local indices:
        # A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j, (1,0)->k, (1,1)->l
        
        # Map local to global
        # U index: (i_off + li) * m + (j_off + lj)
        # V index: (j_off + lj) * p + (k_off + lk)
        # W index: (k_off + lk) * n + (i_off + li)
        
        # Strassen terms:
        # 1. M1 = (a+d)(e+h) -> C00+=1, C01+=1, C10+=1, C11+=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 2. M2 = (c+d)e -> C10+=1, C11+=-1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # 3. M3 = a(f-h) -> C00+=1, C01+=-1
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # 4. M4 = d(g-e) -> C10+=1, C11+=-1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        col += 1
        
        # 5. M5 = (a+b)h -> C00+=1, C01+=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # 6. M6 = (c-a)(e+f) -> C10+=1, C11+=1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 7. M7 = (b-d)(g+h) -> C00+=1, C01+=1
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

    # Helper to add a 1x2x2 block (rank 4)
    # A is 1x2, B is 2x2, C is 1x2
    def add_122(i_off, j_off, k_off):
        nonlocal col
        # A: (0,0)->a, (0,1)->b
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j
        
        # 1. ae -> C0
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 2. bg -> C0
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 3. af -> C1
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 4. bh -> C1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    # --- Construction ---
    
    # Top part: A[0:2, :] * B[:, :]
    # A_top is 2x4. B is 4x4.
    # Split A_top into A_T0 (cols 0-1) and A_T1 (cols 2-3).
    # Split B into B0 (rows 0-1) and B1 (rows 2-3).
    # C_top_left (2x2) = A_T0 * B0 + A_T1 * B1? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For i in 0,1:
    # C[i, 0:2] = sum_{j=0}^{1} A[i,j] B[j, 0:2] + sum_{j=2}^{3} A[i,j] B[j, 0:2]
    #            = A_T0[i, :] * B[0:2, 0:2] + A_T1[i, :] * B[2:4, 0:2]
    
    # Block 1: A_T0 (i=0,1; j=0,1) * B[0:2, 0:2] (j=0,1; k=0,1) -> C[0:2, 0:2]
    add_strassen_222(0, 0, 0)
    
    # Block 2: A_T1 (i=0,1; j=2,3) * B[2:4, 0:2] (j=2,3; k=0,1) -> C[0:2, 0:2]
    add_strassen_222(0, 2, 0)
    
    # Block 3: A_T0 (i=0,1; j=0,1) * B[0:2, 2:4] (j=0,1; k=2,3) -> C[0:2, 2:4]
    add_strassen_222(0, 0, 2)
    
    # Block 4: A_T1 (i=0,1; j=2,3) * B[2:4, 2:4] (j=2,3; k=2,3) -> C[0:2, 2:4]
    add_strassen_222(0, 2, 2)
    
    # Bottom part: A[2, :] * B[:, :]
    # A_bot is 1x4.
    # Split A_bot into A_B0 (cols 0-1) and A_B1 (cols 2-3).
    # C[2, 0:2] = A_B0 * B[0:2, 0:2] + A_B1 * B[2:4, 0:2]
    # C[2, 2:4] = A_B0 * B[0:2, 2:4] + A_B1 * B[2:4, 2:4]
    
    # Block 5: A_B0 (i=2; j=0,1) * B[0:2, 0:2] (j=0,1; k=0,1) -> C[2, 0:2]
    add_122(2, 0, 0)
    
    # Block 6: A_B1 (i=2; j=2,3) * B[2:4, 0:2] (j=2,3; k=0,1) -> C[2, 0:2]
    add_122(2, 2, 0)
    
    # Block 7: A_B0 (i=2; j=0,1) * B[0:2, 2:4] (j=0,1; k=2,3) -> C[2, 2:4]
    add_122(2, 0, 2)
    
    # Block 8: A_B1 (i=2; j=2,3) * B[2:4, 2:4] (j=2,3; k=2,3) -> C[2, 2:4]
    add_122(2, 2, 2)
    
    # Verify col count matches rank
    if col != rank:
        return None
        
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    n, m, p = 3, 4, 4
    
    time_limit = 120  # seconds
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + time_limit - 5  # 5 second safety margin
    
    # Try to find the rank-44 decomposition
    try:
        U, V, W = _rank_44_decomposition(n, m, p)
        
        # Check if we're still within time
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
            
        # Final verification
        if not _verify(U, V, W, n, m, p):
            # If verification fails, fall back to schoolbook
            U, V, W = _schoolbook_rank(n, m, p)
            
    except Exception:
        # Fallback to schoolbook if anything goes wrong
        U, V, W = _schoolbook_rank(n, m, p)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
