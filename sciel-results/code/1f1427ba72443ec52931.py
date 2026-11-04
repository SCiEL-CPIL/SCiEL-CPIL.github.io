"""
4x4x4 matrix multiplication tensor decomposition via Kronecker product of Strassen.
Achieves rank 49 with exact integer entries.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_4x4():
    """Fallback: schoolbook 4x4x4 decomposition (rank 64)."""
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U2, V2, W2 as int arrays of shape (4, 7).
    Row indices: U2[i*2+j], V2[j*2+k], W2[k*2+i] (trace convention).
    """
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    return U2, V2, W2

def _verify_2x2(U2, V2, W2):
    """Verify 2x2 Strassen: sum_c U2[i*2+j,c]*V2[j*2+k,c]*W2[k*2+i,c] == 1 for all i,j,k."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j, :], V2[j*2+k, :] * W2[k*2+i, :]))
                if s != 1:
                    return False
    return True

def _build_permutation():
    """
    Build the row permutation for Kronecker product to match 4x4 linear indexing.
    perm[S] = R where S = i*4+j (target index) and R is the corresponding kron row index.
    
    For np.kron(A2, B2) with A2,B2 of shape (4,7):
    - kron row R = a*4 + b where a is row in first factor, b in second
    - a = i_hi*2 + j_hi (treating high bits as 2x2 indices)
    - b = i_lo*2 + j_lo
    - Target S = i*4 + j where i = i_hi*2+i_lo, j = j_hi*2+j_lo
    """
    perm = np.empty(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_hi, i_lo = i // 2, i % 2
        j_hi, j_lo = j // 2, j % 2
        a = i_hi * 2 + j_hi
        b = i_lo * 2 + j_lo
        R = a * 4 + b
        perm[S] = R
    return perm

def _verify_4x4(U4, V4, W4):
    """Verify 4x4 decomposition: sum_c U4[i*4+j,c]*V4[j*4+k,c]*W4[k*4+i,c] == 1 for all i,j,k."""
    # Vectorized check: build tensors and verify
    n = 4
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = int(np.dot(U4[i*n+j, :], V4[j*n+k, :] * W4[k*n+i, :]))
                if s != 1:
                    return False
    return True

def construct():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication
    using Kronecker product of Strassen rank-7 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Get Strassen 2x2 factors
    U2, V2, W2 = _strassen_2x2()
    
    # Quick verify 2x2 (very fast)
    if not _verify_2x2(U2, V2, W2):
        if time.time() > deadline:
            return _schoolbook_4x4()
        # Should not happen, but fallback
        return _schoolbook_4x4()
    
    # Build 4x4 via Kronecker product
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Apply permutation to fix row ordering
    perm = _build_permutation()
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Check deadline before expensive verification
    if time.time() > deadline:
        # Skip verification, trust the construction
        return {
            "U": U4_fixed.tolist(),
            "V": V4_fixed.tolist(),
            "W": W4_fixed.tolist()
        }
    
    # Verify 4x4 correctness
    if not _verify_4x4(U4_fixed, V4_fixed, W4_fixed):
        # Fallback if verification fails
        return _schoolbook_4x4()
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
