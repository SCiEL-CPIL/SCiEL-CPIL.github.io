import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _add_strassen_block(U, V, W, base_col, i0, jA0, jB0, k0):
    """
    Add a rank-7 Strassen decomposition for a 2x2x2 block.
    i0: start row of A (and C)
    jA0: start col of A
    jB0: start row of B
    k0: start col of B (and C)
    """
    col = base_col
    # M1 = (A00+A11)(B00+B11)
    U[i0*4 + jA0, col] += 1
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[jB0*8 + k0, col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[(i0+1)*4 + jA0, col] += 1
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[jB0*8 + k0, col] += 1
    W[k0*3 + (i0+1), col] += 1
    W[(k0+1)*3 + (i0+1), col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[i0*4 + jA0, col] += 1
    V[jB0*8 + (k0+1), col] += 1
    V[(jB0+1)*8 + (k0+1), col] -= 1
    W[(k0+1)*3 + i0, col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[(jB0+1)*8 + k0, col] += 1
    V[jB0*8 + k0, col] -= 1
    W[k0*3 + i0, col] += 1
    W[k0*3 + (i0+1), col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[i0*4 + jA0, col] += 1
    U[i0*4 + (jA0+1), col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] -= 1
    W[(k0+1)*3 + i0, col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[(i0+1)*4 + jA0, col] += 1
    U[i0*4 + jA0, col] -= 1
    V[jB0*8 + k0, col] += 1
    V[jB0*8 + (k0+1), col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[i0*4 + (jA0+1), col] += 1
    U[(i0+1)*4 + (jA0+1), col] -= 1
    V[(jB0+1)*8 + k0, col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] += 1
    col += 1
    
    return col

def _construct_hybrid():
    """
    Construct a rank-88 decomposition for 3x4x8 matmul.
    Split 3x4 into 1x4 (schoolbook, rank 32) + 2x4 (Strassen-based, rank 56).
    """
    n, m, p = 3, 4, 8
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    col = 0
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) - Strassen blocks
    # We have 2x4x8. We can view this as 4 blocks of 2x2x2 in the B dimension (k=0..7),
    # but Strassen works on 2x2x2. 
    # Actually, let's partition B into 4 blocks of 2 columns: k=0,1; 2,3; 4,5; 6,7.
    # For each k-block, we have a 2x2 (A rows 1,2; A cols 0,1) * 2x2 (B rows 0,1; B cols k,k+1) -> 2x2 (C rows 1,2; C cols k,k+1)
    # Wait, A is 2x4. B is 4x8.
    # Let's partition A cols into 2 blocks of 2: j=0,1 and j=2,3.
    # Let's partition B rows into 2 blocks of 2: j=0,1 and j=2,3.
    # Let's partition B cols into 4 blocks of 2: k=0,1; 2,3; 4,5; 6,7.
    # C is 2x8. Partition C cols into 4 blocks of 2.
    
    # Block (A_j, B_j, C_k) where A_j is 2x2, B_j is 2x2, C_k is 2x2.
    # Sum over j=0,1.
    # So for each k-block (4 of them), we sum two 2x2x2 multiplications.
    # Each 2x2x2 multiplication can be done with rank 7.
    # Total rank for Part 2 = 4 k-blocks * 2 j-blocks * 7 = 56.
    # Total rank = 32 + 56 = 88.
    
    # A rows: 1, 2 (indices 4, 5 in U)
    # A col blocks: (0,1) and (2,3)
    # B row blocks: (0,1) and (2,3)
    # B col blocks: (0,1), (2,3), (4,5), (6,7)
    # C rows: 1, 2 (indices 1, 2 in W)
    # C col blocks: (0,1), (2,3), (4,5), (6,7)
    
    # jA0: start col of A block (0 or 2)
    # jB0: start row of B block (0 or 2)
    # k0: start col of B/C block (0, 2, 4, 6)
    # i0: start row of A/C block (1)
    
    for k0 in [0, 2, 4, 6]:
        # j-block 0: A cols 0-1, B rows 0-1
        col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=k0)
        # j-block 1: A cols 2-3, B rows 2-3
        col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=k0)
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 110  # Safety margin
    
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Schoolbook fallback (rank 96)
    R = n * m * p
    U_sb = [[0]*R for _ in range(n*m)]
    V_sb = [[0]*R for _ in range(m*p)]
    W_sb = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_h, "W": W_h}
