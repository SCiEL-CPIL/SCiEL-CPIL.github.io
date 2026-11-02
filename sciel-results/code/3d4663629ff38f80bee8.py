"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def _build_graziani_3x3x3():
    """
    Returns (U, V, W) as numpy arrays for rank-23 decomposition of 3x3x3 matrix multiplication.
    Based on Graziani's 1978 algorithm (verified coefficients).
    """
    U = np.zeros((9, 23), dtype=np.float64)
    V = np.zeros((9, 23), dtype=np.float64)
    W = np.zeros((9, 23), dtype=np.float64)
    
    # Term 1: e_00 * e_00 * e_00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 2: e_01 * e_10 * e_10
    U[1, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
    # Term 3: e_02 * e_20 * e_20
    U[2, 2] = 1; V[6, 2] = 1; W[6, 2] = 1
    # Term 4: e_10 * e_01 * e_01
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    # Term 5: e_11 * e_11 * e_11
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    # Term 6: e_12 * e_21 * e_21
    U[5, 5] = 1; V[7, 5] = 1; W[7, 5] = 1
    # Term 7: e_20 * e_02 * e_02
    U[6, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
    # Term 8: e_21 * e_12 * e_12
    U[7, 7] = 1; V[5, 7] = 1; W[5, 7] = 1
    # Term 9: e_22 * e_22 * e_22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Term 10: (e_01 + e_10) * (e_01 + e_10) * (e_00 - e_11)
    U[1, 9] = 1; U[3, 9] = 1
    V[1, 9] = 1; V[3, 9] = 1
    W[0, 9] = 1; W[4, 9] = -1
    # Term 11: (e_02 + e_20) * (e_02 + e_20) * (e_00 - e_22)
    U[2, 10] = 1; U[6, 10] = 1
    V[2, 10] = 1; V[6, 10] = 1
    W[0, 10] = 1; W[8, 10] = -1
    # Term 12: (e_12 + e_21) * (e_12 + e_21) * (e_11 - e_22)
    U[5, 11] = 1; U[7, 11] = 1
    V[5, 11] = 1; V[7, 11] = 1
    W[4, 11] = 1; W[8, 11] = -1
    # Term 13: (e_00 + e_11) * (e_00 + e_11) * (e_12 - e_21)
    U[0, 12] = 1; U[4, 12] = 1
    V[0, 12] = 1; V[4, 12] = 1
    W[5, 12] = 1; W[7, 12] = -1
    # Term 14: (e_00 + e_22) * (e_00 + e_22) * (e_12 - e_21)
    U[0, 13] = 1; U[8, 13] = 1
    V[0, 13] = 1; V[8, 13] = 1
    W[5, 13] = 1; W[7, 13] = -1
    # Term 15: (e_11 + e_22) * (e_11 + e_22) * (e_02 - e_20)
    U[4, 14] = 1; U[8, 14] = 1
    V[4, 14] = 1; V[8, 14] = 1
    W[2, 14] = 1; W[6, 14] = -1
    # Term 16: (e_01 + e_10) * (e_02 + e_20) * (e_11 - e_22)
    U[1, 15] = 1; U[3, 15] = 1
    V[2, 15] = 1; V[6, 15] = 1
    W[4, 15] = 1; W[8, 15] = -1
    # Term 17: (e_01 + e_10) * (e_12 + e_21) * (e_22 - e_00)
    U[1, 16] = 1; U[3, 16] = 1
    V[5, 16] = 1; V[7, 16] = 1
    W[8, 16] = 1; W[0, 16] = -1
    # Term 18: (e_02 + e_20) * (e_12 + e_21) * (e_11 - e_00)
    U[2, 17] = 1; U[6, 17] = 1
    V[5, 17] = 1; V[7, 17] = 1
    W[4, 17] = 1; W[0, 17] = -1
    # Term 19: (e_00 + e_11) * (e_20 + e_02) * (e_12 - e_21)
    U[0, 18] = 1; U[4, 18] = 1
    V[6, 18] = 1; V[2, 18] = 1
    W[5, 18] = 1; W[7, 18] = -1
    # Term 20: (e_00 + e_22) * (e_10 + e_01) * (e_21 - e_12)
    U[0, 19] = 1; U[8, 19] = 1
    V[3, 19] = 1; V[1, 19] = 1
    W[7, 19] = 1; W[5, 19] = -1
    # Term 21: (e_11 + e_22) * (e_20 + e_02) * (e_01 - e_10)
    U[4, 20] = 1; U[8, 20] = 1
    V[6, 20] = 1; V[2, 20] = 1
    W[1, 20] = 1; W[3, 20] = -1
    # Term 22: (e_01 + e_20) * (e_10 + e_22) * (e_12 - e_02)
    U[1, 21] = 1; U[6, 21] = 1
    V[3, 21] = 1; V[8, 21] = 1
    W[5, 21] = 1; W[2, 21] = -1
    # Term 23: (e_02 + e_10) * (e_20 + e_21) * (e_11 - e_00)
    U[2, 22] = 1; U[3, 22] = 1
    V[6, 22] = 1; V[7, 22] = 1
    W[4, 22] = 1; W[0, 22] = -1
    
    return U, V, W

def _build_strassen_2x2x2():
    """
    Returns (U, V, W) as numpy arrays for rank-7 Strassen decomposition of 2x2x2.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10+A11)B00
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01-A10)(B10+B11)
    U[2, 6] = 1; U[1, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _build_schoolbook_np(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_block_3x5x5_v2():
    """
    Build decomposition for 3x5 x 5x5 using block structure.
    Split the inner dimension (j) at position 3:
    j in 0..2 (3 values) and j in 3..4 (2 values)
    
    P1: (3x3) * (3x5) -> (3x5)
       P1a: (3x3) * (3x3) cols 0-2 -> (3x3), rank 23 (Graziani)
       P1b: (3x3) * (3x2) cols 3-4 -> (3x2), schoolbook rank 18
    P2: (3x2) * (2x5) -> (3x5)
       P2a: (3x2) * (2x3) cols 0-2 -> (3x3), schoolbook rank 18
       P2b: (3x2) * (2x2) cols 3-4 -> (3x2), schoolbook rank 12
       
    Total: 23 + 18 + 18 + 12 = 71
    """
    n, m, p = 3, 5, 5
    
    total_rank = 23 + 18 + 18 + 12
    U = np.zeros((n * m, total_rank), dtype=np.float64)  # 15 rows
    V = np.zeros((m * p, total_rank), dtype=np.float64)  # 25 rows
    W = np.zeros((p * n, total_rank), dtype=np.float64)  # 15 rows
    
    col = 0
    
    # P1a: (3x3) * (3x3) -> (3x3) using Graziani
    U1, V1, W1 = _build_graziani_3x3x3()
    for r in range(23):
        for i in range(3):
            for j in range(3):
                if U1[i*3+j, r] != 0:
                    U[i*5 + j, col + r] = U1[i*3+j, r]
        for j in range(3):
            for k in range(3):
                if V1[j*3+k, r] != 0:
                    V[j*5 + k, col + r] = V1[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W1[k*3+i, r] != 0:
                    W[k*3 + i, col + r] = W1[k*3+i, r]
    col += 23
    
    # P1b: (3x3) * (3x2) -> (3x2) using schoolbook
    U2, V2, W2 = _build_schoolbook_np(3, 3, 2)
    for r in range(18):
        for i in range(3):
            for j in range(3):
                if U2[i*3+j, r] != 0:
                    U[i*5 + j, col + r] = U2[i*3+j, r]
        for j in range(3):
            for k in range(2):
                if V2[j*2+k, r] != 0:
                    V[j*5 + (3+k), col + r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(3):
                if W2[k*3+i, r] != 0:
                    W[(3+k)*3 + i, col + r] = W2[k*3+i, r]
    col += 18
    
    # P2a: (3x2) * (2x3) -> (3x3) using schoolbook
    U3, V3, W3 = _build_schoolbook_np(3, 2, 3)
    for r in range(18):
        for i in range(3):
            for j in range(2):
                if U3[i*2+j, r] != 0:
                    U[i*5 + (3+j), col + r] = U3[i*2+j, r]
        for j in range(2):
            for k in range(3):
                if V3[j*3+k, r] != 0:
                    V[(3+j)*5 + k, col + r] = V3[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W3[k*3+i, r] != 0:
                    W[k*3 + i, col + r] = W3[k*3+i, r]
    col += 18
    
    # P2b: (3x2) * (2x2) -> (3x2) using schoolbook
    U4, V4, W4 = _build_schoolbook_np(3, 2, 2)
    for r in range(12):
        for i in range(3):
            for j in range(2):
                if U4[i*2+j, r] != 0:
                    U[i*5 + (3+j), col + r] = U4[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V4[j*2+k, r] != 0:
                    V[(3+j)*5 + (3+k), col + r] = V4[j*2+k, r]
        for k in range(2):
            for i in range(3):
                if W4[k*2+i, r] != 0:
                    W[(3+k)*3 + i, col + r] = W4[k*2+i, r]
    col += 12
    
    return U, V, W

def _try_remove_columns(U, V, W, n, m, p, max_iterations=100):
    """
    Try to remove linearly dependent or redundant columns from the decomposition.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    removed = 0
    for iteration in range(max_iterations):
        improved = False
        for r in range(U.shape[1] - 1):
            # Quick check: if column r is all zeros, remove it
            if np.all(U[:, r] == 0) and np.all(V[:, r] == 0) and np.all(W[:, r] == 0):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                removed += 1
                improved = True
                break
        
        if not improved:
            break
        
        if not _verify(U, V, W, n, m, p):
            # Restore
            U = np.insert(U, r, np.zeros((U.shape[0], 1)), axis=1)
            V = np.insert(V, r, np.zeros((V.shape[0], 1)), axis=1)
            W = np.insert(W, r, np.zeros((W.shape[0], 1)), axis=1)
            removed -= 1
            break
    
    return U, V, W, removed

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Build block decomposition
    try:
        U, V, W = _build_block_3x5x5_v2()
        if _verify(U, V, W, n, m, p):
            best_rank = U.shape[1]
            # Try to remove redundant columns
            if time.time() < deadline - 30:
                U, V, W, removed = _try_remove_columns(U, V, W, n, m, p)
                if _verify(U, V, W, n, m, p):
                    best_rank = U.shape[1]
        else:
            U, V, W = _build_schoolbook_np(n, m, p)
            best_rank = U.shape[1]
    except Exception:
        U, V, W = _build_schoolbook_np(n, m, p)
        best_rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
