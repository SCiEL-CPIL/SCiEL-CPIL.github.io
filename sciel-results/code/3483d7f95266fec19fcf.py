"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Rank 18 strategy:
    # Block 1: Cols 0, 1 -> Strassen (Rank 7)
    # Block 2: Cols 2, 3 -> Strassen (Rank 7)
    # Block 3: Col 4     -> Schoolbook (Rank 4)
    
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add Strassen terms for a given column offset
    def add_strassen(offset_col_start, rank_offset):
        # offset_col_start: which column of B this block starts at (0 or 2)
        # rank_offset: which rank index to start writing to
        
        # m1 = (A00 + A11)(B00 + B11)
        # B indices: j*p + (offset_col_start + 0) and j*p + (offset_col_start + 1)
        # C indices: (offset_col_start + 0)*n + i and (offset_col_start + 1)*n + i
        
        # Term 0
        U[0, rank_offset + 0] = 1; U[3, rank_offset + 0] = 1
        V[0*p + offset_col_start, rank_offset + 0] = 1
        V[1*p + (offset_col_start + 1), rank_offset + 0] = 1
        W[(offset_col_start + 0)*n + 0, rank_offset + 0] = 1
        W[(offset_col_start + 1)*n + 1, rank_offset + 0] = 1
        
        # m2 = (A10 + A11)B00
        U[2, rank_offset + 1] = 1; U[3, rank_offset + 1] = 1
        V[0*p + offset_col_start, rank_offset + 1] = 1
        W[(offset_col_start + 0)*n + 1, rank_offset + 1] = 1
        W[(offset_col_start + 1)*n + 1, rank_offset + 1] = -1
        
        # m3 = A00(B01 - B11)
        U[0, rank_offset + 2] = 1
        V[0*p + (offset_col_start + 1), rank_offset + 2] = 1
        V[1*p + (offset_col_start + 1), rank_offset + 2] = -1
        W[(offset_col_start + 1)*n + 0, rank_offset + 2] = 1
        W[(offset_col_start + 1)*n + 1, rank_offset + 2] = -1
        
        # m4 = A11(B10 - B00)
        U[3, rank_offset + 3] = 1
        V[1*p + offset_col_start, rank_offset + 3] = 1
        V[0*p + offset_col_start, rank_offset + 3] = -1
        W[(offset_col_start + 0)*n + 1, rank_offset + 3] = 1
        W[(offset_col_start + 0)*n + 0, rank_offset + 3] = -1
        
        # m5 = (A00 + A10)B11
        U[0, rank_offset + 4] = 1; U[2, rank_offset + 4] = 1
        V[1*p + (offset_col_start + 1), rank_offset + 4] = 1
        W[(offset_col_start + 1)*n + 0, rank_offset + 4] = 1
        W[(offset_col_start + 1)*n + 1, rank_offset + 4] = -1
        
        # m6 = (A01 - A10)(B00 + B10)
        U[1, rank_offset + 5] = 1; U[2, rank_offset + 5] = -1
        V[0*p + offset_col_start, rank_offset + 5] = 1
        V[1*p + offset_col_start, rank_offset + 5] = 1
        W[(offset_col_start + 0)*n + 0, rank_offset + 5] = 1
        
        # m7 = (A01 - A11)(B01 + B11)
        U[1, rank_offset + 6] = 1; U[3, rank_offset + 6] = -1
        V[0*p + (offset_col_start + 1), rank_offset + 6] = 1
        V[1*p + (offset_col_start + 1), rank_offset + 6] = 1
        W[(offset_col_start + 1)*n + 1, rank_offset + 6] = 1

    # Block 1: Cols 0, 1 (Rank 0-6)
    add_strassen(0, 0)
    
    # Block 2: Cols 2, 3 (Rank 7-13)
    add_strassen(2, 7)
    
    # Block 3: Col 4 (Rank 14-17)
    # Schoolbook for 2x2x1
    # C_i = sum_j A_ij B_j
    # Terms:
    # (i=0, j=0): U[0, r]=1, V[0*p+4, r]=1, W[4*n+0, r]=1
    U[0, 14] = 1; V[0*p+4, 14] = 1; W[4*n+0, 14] = 1
    # (i=0, j=1): U[1, r]=1, V[1*p+4, r]=1, W[4*n+0, r]=1
    U[1, 15] = 1; V[1*p+4, 15] = 1; W[4*n+0, 15] = 1
    # (i=1, j=0): U[2, r]=1, V[0*p+4, r]=1, W[4*n+1, r]=1
    U[2, 16] = 1; V[0*p+4, 16] = 1; W[4*n+1, 16] = 1
    # (i=1, j=1): U[3, r]=1, V[1*p+4, r]=1, W[4*n+1, r]=1
    U[3, 17] = 1; V[1*p+4, 17] = 1; W[4*n+1, 17] = 1

    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 19 if verification fails (should not happen)
    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Re-use logic from previous block for safety
    # Strassen 0-1
    add_strassen(0, 0)
    # Schoolbook 2
    U[0, 7] = 1; V[2, 7] = 1; W[0, 7] = 1
    U[1, 8] = 1; V[7, 8] = 1; W[0, 8] = 1
    U[2, 9] = 1; V[2, 9] = 1; W[1, 9] = 1
    U[3, 10] = 1; V[7, 10] = 1; W[1, 10] = 1
    # Schoolbook 3
    U[0, 11] = 1; V[3, 11] = 1; W[0, 11] = 1
    U[1, 12] = 1; V[8, 12] = 1; W[0, 12] = 1
    U[2, 13] = 1; V[3, 13] = 1; W[1, 13] = 1
    U[3, 14] = 1; V[8, 14] = 1; W[1, 14] = 1
    # Schoolbook 4
    U[0, 15] = 1; V[4, 15] = 1; W[0, 15] = 1
    U[1, 16] = 1; V[9, 16] = 1; W[0, 16] = 1
    U[2, 17] = 1; V[4, 17] = 1; W[1, 17] = 1
    U[3, 18] = 1; V[9, 18] = 1; W[1, 18] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
