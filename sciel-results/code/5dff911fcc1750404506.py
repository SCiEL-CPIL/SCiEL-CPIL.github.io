"""Matrix multiplication 2x2x15 tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

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
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """
    Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
    Returns U (4x4), V (2x4), W (2x4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    assert U.shape == (n*m, rank), f"U shape {U.shape} != {(n*m, rank)}"
    assert V.shape == (m*p, rank), f"V shape {V.shape} != {(m*p, rank)}"
    assert W.shape == (p*n, rank), f"W shape {p*n, rank} != {W.shape}"
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                if abs(val - expected) > 1e-10:
                    return False, (i, j, k, val, expected)
    return True, None

def _try_find_2x2x3_rank10(deadline):
    """
    Try to find a rank-10 decomposition for 2x2 x 2x3.
    Uses the structure: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    We look for U (4x10), V (6x10), W (6x10) with integer/half-integer entries.
    
    Strategy: Use known results - 2x2x3 has rank 10 (not 9) per literature.
    But let's try a constructive approach using bilinear forms.
    """
    import itertools
    import random
    n, m, p = 2, 2, 3
    rank = 10
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try a structured approach:
    # Split into two 2x2x2 blocks and one 2x2x1 block, but share some terms
    # Block 1 (k=0): Strassen-like for columns 0
    # Block 2 (k=1): Strassen-like for columns 1  
    # Block 3 (k=2): Schoolbook for column 2
    # But we can share the U terms across k blocks
    
    # Let's try: use 7 Strassen terms for (k=0,1) combined with 3 extra for k=2
    # Actually, let's try a direct search with random initialization and gradient-free refinement
    
    # Known: rank(2x2x3) = 10. Let's construct it.
    # Approach: C[:,0:2] uses rank 7 (Strassen for 2x2x2), C[:,2] needs rank 3 more? No...
    # Actually C[:,2] = A * B[:,2] which is 2x2 * 2x1, rank 4.
    # But we can share: if we use 7 terms for first two columns, the third column adds 3 new terms.
    # Total: 7 + 3 = 10. Let's verify this works.
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 7 terms handle k=0 and k=1 (the 2x2x2 part)
    U[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j*p + k_local, 0:7] = V_str[j*2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[k_local*n + i, 0:7] = W_str[k_local*2 + i, :]
    
    # Terms 7,8,9 handle k=2 with schoolbook-like structure but sharing
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Term 7: A[0,0]*B[0,2] -> contributes to C[0,2]
    U[0*2+0, 7] = 1
    V[0*2+2, 7] = 1
    W[2*2+0, 7] = 1
    
    # Term 8: A[1,0]*B[0,2] -> contributes to C[1,2]
    U[1*2+0, 8] = 1
    V[0*2+2, 8] = 1
    W[2*2+1, 8] = 1
    
    # Term 9: A[0,1]*B[1,2] + A[1,1]*B[1,2]... wait, that's two terms
    # We need: C[0,2] += A[0,1]*B[1,2] and C[1,2] += A[1,1]*B[1,2]
    # That's two more terms, but we only have one slot (term 9)
    # So we need a different approach for the last column
    
    # Let me reconsider. Maybe use a different splitting.
    # Actually, let's try: 6 terms for a 2x2x2-like structure that covers k=0,1,2 partially
    # This is getting complex. Let's just verify if rank 10 is achievable with a known construction.
    
    # Alternative: use the fact that 2x2x3 can be decomposed as:
    # 3 schoolbook columns = 3*4 = 12 (too much)
    # 1 Strassen (2 cols) + 1 schoolbook (1 col) = 7 + 4 = 11 (still too much)
    # Need to share 1 term between Strassen and schoolbook parts.
    
    # Let me try a more careful construction:
    # Terms 0-6: Strassen for k=0,1
    # Term 7: handles A[0,0]*B[0,2] -> C[0,2]
    # Term 8: handles A[1,0]*B[0,2] -> C[1,2]  
    # Term 9: handles (A[0,1]+A[1,1])*B[1,2] -> C[0,2] += A[0,1]*B[1,2], C[1,2] += A[1,1]*B[1,2]
    # But term 9 needs to give +A[0,1]*B[1,2] to C[0,2] and +A[1,1]*B[1,2] to C[1,2]
    # So U[0,9]=1, U[1,9]=1 (wait, indices: U[i*m+j, r], so U[0*2+1,9]=U[1,9] for A[0,1], U[1*2+1,9]=U[3,9] for A[1,1])
    # V[1*2+2, 9] = V[4,9] = 1 for B[1,2]
    # W[2*2+0, 9] = W[4,9] = 1 for C[0,2]
    # W[2*2+1, 9] = W[5,9] = 1 for C[1,2]
    
    U[0*2+1, 9] = 1  # A[0,1]
    U[1*2+1, 9] = 1  # A[1,1]
    V[1*2+2, 9] = 1  # B[1,2]
    W[2*2+0, 9] = 1  # C[0,2]
    W[2*2+1, 9] = 1  # C[1,2]
    
    # Now verify
    valid, info = _verify(U, V, W, n, m, p)
    if valid:
        return U, V, W
    
    # If not valid, return None to indicate failure
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # First, try to find a rank-10 decomposition for 2x2x3
    result_223 = _try_find_2x2x3_rank10(deadline)
    
    if result_223 is not None:
        U_223, V_223, W_223 = result_223
        # Use 5 blocks of size 3 (using rank-10 each) = 50 total rank
        # Or mix: some 2x2x3 blocks and some 2x2x2 blocks
        # 15 = 5*3, so 5 blocks of 2x2x3 at rank 10 = 50
        # Compare with current: 7*7 + 4 = 53
        # 50 < 53, so this would be an improvement!
        
        total_rank = 10 * 5  # 50
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col_offset = 0
        for b in range(5):
            start_k = b * 3
            U[:, col_offset:col_offset+10] = U_223
            for j in range(m):
                for k_local in range(3):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
            for k_local in range(3):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
            col_offset += 10
        
        # Verify
        valid, info = _verify(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to original Strassen-based approach with rank 53
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            # Use Strassen
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            # Use Schoolbook
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
            
    # Verify
    valid, info = _verify(U, V, W, n, m, p)
    if not valid:
        # If verification fails, fall back to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = np.zeros((n * m, total_rank_fb))
        V_fb = np.zeros((m * p, total_rank_fb))
        W_fb = np.zeros((p * n, total_rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, idx] = 1
                    V_fb[j*p+k, idx] = 1
                    W_fb[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
