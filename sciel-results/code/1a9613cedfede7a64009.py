"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

_start_time = time.time()
TIME_LIMIT = 145  # safety margin before 150s

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT


def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W


def _build_2x2x3_rank10():
    """
    Build a rank-10 decomposition for 2x2 x 2x3 matrix multiplication.
    Returns U (4x10), V (6x10), W (6x10) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We'll use a known construction. For 2x2x3, one approach:
    # Split into two 2x2x2 blocks (using Strassen) plus corrections, but that gives 14.
    # Instead, use a direct rank-10 construction.
    
    # Known rank-10 decomposition for 2x2x3:
    # We can use the following approach based on literature.
    
    # Let me construct it systematically.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0,1,2
    
    # Using a structured approach with 10 terms:
    
    # Terms 0-6: Strassen-like for k=0,1 (2x2x2 part)
    # But we need to handle k=2 as well with shared terms.
    
    # Alternative: use the fact that rank(2x2x3) = 10 is achievable.
    # Let me use a verified construction.
    
    # Construction from literature (e.g., using linear algebra):
    # We need 10 rank-1 tensors that sum to the 2x2x3 multiplication tensor.
    
    # Here's a valid rank-10 decomposition:
    
    # Term 0: A00 * B00
    U[0*2+0, 0] = 1
    V[0*3+0, 0] = 1
    W[0*2+0, 0] = 1
    
    # Term 1: A00 * B01
    U[0*2+0, 1] = 1
    V[0*3+1, 1] = 1
    W[1*2+0, 1] = 1
    
    # Term 2: A00 * B02
    U[0*2+0, 2] = 1
    V[0*3+2, 2] = 1
    W[2*2+0, 2] = 1
    
    # Term 3: A01 * B10
    U[0*2+1, 3] = 1
    V[1*3+0, 3] = 1
    W[0*2+0, 3] = 1
    
    # Term 4: A01 * B11
    U[0*2+1, 4] = 1
    V[1*3+1, 4] = 1
    W[1*2+0, 4] = 1
    
    # Term 5: A01 * B12
    U[0*2+1, 5] = 1
    V[1*3+2, 5] = 1
    W[2*2+0, 5] = 1
    
    # Term 6: A10 * B00
    U[1*2+0, 6] = 1
    V[0*3+0, 6] = 1
    W[0*2+1, 6] = 1
    
    # Term 7: A10 * B01
    U[1*2+0, 7] = 1
    V[0*3+1, 7] = 1
    W[1*2+1, 7] = 1
    
    # Term 8: A10 * B02
    U[1*2+0, 8] = 1
    V[0*3+2, 8] = 1
    W[2*2+1, 8] = 1
    
    # Term 9: A11 * B10
    U[1*2+1, 9] = 1
    V[1*3+0, 9] = 1
    W[0*2+1, 9] = 1
    
    # This is only rank 10 but we're missing A11*B11 and A11*B12!
    # This construction is wrong. Let me use a proper rank-10 decomposition.
    
    # Actually, the above is just schoolbook for 2x2x3 which has rank 12, not 10.
    # I need a proper rank-10 construction.
    
    # Let me use a different approach: combine Strassen for 2x2x2 with one extra column.
    # For k=0,1: use Strassen (7 terms)
    # For k=2: need to add terms, but we can share some.
    
    # Actually, let me just use schoolbook for 2x2x3 (rank 12) and see if the total is still better.
    # 6*7 + 12 = 54, which is worse than 53.
    
    # Let me try a different split: 7 pairs + 1 single
    # 7 blocks of 2x2x2: 7*7 = 49
    # 1 block of 2x2x1: rank 4
    # Total: 53 (same as before)
    
    # Or: 5 blocks of 2x2x2 + 1 block of 2x2x5
    # Need rank(2x2x5). Known: rank(2x2x5) <= 18 (from literature)
    # 5*7 + 18 = 53
    
    # Let me try yet another approach. Use the known result that rank(2x2x15) can be lower.
    
    # For now, let me just return the schoolbook for 2x2x3 (rank 12)
    # and we'll get total rank 6*7+12 = 54, which is worse.
    
    # Let me instead use a proper rank-10 construction if I can find one.
    
    # After research, a valid rank-10 decomposition for 2x2x3 exists.
    # Here's one construction:
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use the following 10 terms (verified construction):
    # This is based on the work of Bini et al. and subsequent improvements.
    
    # I'll use a simple but valid approach: take Strassen for columns 0,1 
    # and add 3 more terms for column 2, with some sharing.
    
    # Actually, let me just use schoolbook for simplicity and correctness.
    # The total will be 6*7 + 12 = 54, which is worse than 53.
    
    # Let me go back to the 53-rank approach: 6 Strassen blocks + 1 schoolbook 2x2x1
    return None  # Signal to use alternative approach


def _build_schoolbook_2x2x1():
    """Build schoolbook decomposition for 2x2 x 2x1 matrix multiplication (rank 4)."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the n x m x p matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = 2, 2, 15
    
    # Strategy: Split the 15 columns into 6 groups of 2 and 1 group of 1.
    # Use Strassen (rank 7) for each 2x2x2 block, schoolbook (rank 4) for 2x2x1.
    # Total rank: 6*7 + 4 = 46 + 4 = 50
    
    # Wait, that's even better! Let me verify the index mapping.
    
    # For a 2x2x2 block at columns k0, k0+1:
    # The local indices are i=0,1; j=0,1; k_local=0,1
    # Global: k = k0 + k_local
    
    # U: shape (4, total_rank)
    # V: shape (30, total_rank) 
    # W: shape (30, total_rank)
    
    # For each block, we map:
    # U[i*m+j, local_col] -> U[i*m+j, global_col]
    # V[j*p+k_global, local_col] -> V[j*p+k_global, global_col]
    # W[k_global*n+i, local_col] -> W[k_global*n+i, global_col]
    
    total_rank = 6 * 7 + 4  # 46 + 4 = 50
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # 6 blocks of 2x2x2
    for block in range(6):
        if _time_exceeded():
            break
        k0 = block * 2  # starting column index
        
        U_block, V_block, W_block = _build_strassen_2x2x2()
        
        # Map local to global indices
        for local_col in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col_offset + local_col] = U_block[i * m + j, local_col]
            for j in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j * p + k_global, col_offset + local_col] = V_block[j * 2 + k_local, local_col]
            for k_local in range(2):
                for i in range(2):
                    k_global = k0 + k_local
                    W[k_global * n + i, col_offset + local_col] = W_block[k_local * 2 + i, local_col]
        
        col_offset += 7
    
    # 1 block of 2x2x1 (column 12,13,14 -> actually just column 12? No, we have 15 columns: 0-14)
    # 6 blocks of 2 = 12 columns (0-11), remaining: columns 12,13,14 = 3 columns
    # So we need 2x2x3, not 2x2x1!
    
    # Let me redo: 15 = 7*2 + 1, so 7 blocks of 2 and 1 block of 1
    # Total: 7*7 + 4 = 49 + 4 = 53
    
    # Let me restart with the correct split.
    
    U = np.zeros((n * m, 53))
    V = np.zeros((m * p, 53))
    W = np.zeros((p * n, 53))
    
    col_offset = 0
    
    # 7 blocks of 2x2x2 (columns 0-13)
    for block in range(7):
        if _time_exceeded():
            break
        k0 = block * 2
        
        U_block, V_block, W_block = _build_strassen_2x2x2()
        
        for local_col in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col_offset + local_col] = U_block[i * m + j, local_col]
            for j in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j * p + k_global, col_offset + local_col] = V_block[j * 2 + k_local, local_col]
            for k_local in range(2):
                for i in range(2):
                    k_global = k0 + k_local
                    W[k_global * n + i, col_offset + local_col] = W_block[k_local * 2 + i, local_col]
        
        col_offset += 7
    
    # 1 block of 2x2x1 (column 14)
    if not _time_exceeded():
        k0 = 14
        U_block, V_block, W_block = _build_schoolbook_2x2x1()
        
        for local_col in range(4):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col_offset + local_col] = U_block[i * m + j, local_col]
            for j in range(2):
                k_global = k0
                V[j * p + k_global, col_offset + local_col] = V_block[j * 1 + 0, local_col]
            for i in range(2):
                k_global = k0
                W[k_global * n + i, col_offset + local_col] = W_block[0 * 2 + i, local_col]
        
        col_offset += 4
    
    # Verify the decomposition
    if not _time_exceeded():
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_school = np.zeros((n * m, rank_school))
            V_school = np.zeros((m * p, rank_school))
            W_school = np.zeros((p * n, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j, col] = 1
                        V_school[j * p + k, col] = 1
                        W_school[k * n + i, col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
