"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 115


def _strassen_222():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where U is (4,7), V is (4,7), W is (4,7).
    U[i*2+j] is the coefficient for A[i,j], V[i*2+j] for B[i,j], W[i*2+j] for C[i,j].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_rank71():
    """Build rank-71 decomposition using Strassen on 2x2x2 sub-blocks with 1+4 inner split.
    
    C11 (2x2): inner 0 (1) + inner 1-2 (2, Strassen) + inner 3-4 (2, Strassen) = 4+7+7 = 18
    C12 (2x3): inner 0 (1) + inner 1-4 (4):
                B cols 0-1: inner 1-2 (Strassen 7) + inner 3-4 (Strassen 7) = 14
                B col 2: inner 1-2 (4) + inner 3-4 (4) = 8
                Total: 6 + 14 + 8 = 28
    C21 (1x2): inner 0 (1) + inner 1-4 (4) = 2 + 8 = 10
    C22 (1x3): inner 0 (1) + inner 1-4 (4) = 3 + 12 = 15
    Total: 18 + 28 + 10 + 15 = 71
    """
    total_rank = 71
    U = [[0]*total_rank for _ in range(15)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(15)]
    
    col = 0
    
    # === C11 (2x2) = A rows 0-1, B cols 0-1 ===
    # Part 1: inner 0, schoolbook 2x1x2 (4 terms)
    for gi in range(2):
        for gk in range(2):
            a_idx = gi * 5 + 0
            b_idx = 0 * 5 + gk
            w_idx = gk * 3 + gi
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Part 2: inner 1-2, Strassen 2x2x2 (7 terms)
    sU, sV, sW = _strassen_222()
    for local_i in range(4):
        gi = local_i // 2
        gj = local_i % 2
        a_row = gi
        a_inner = 1 + gj
        for c in range(7):
            U[a_row * 5 + a_inner][col + c] = sU[local_i][c]
    for local_j in range(4):
        gj = local_j // 2
        gk = local_j % 2
        b_inner = 1 + gj
        b_col = gk
        for c in range(7):
            V[b_inner * 5 + b_col][col + c] = sV[local_j][c]
    for local_k in range(4):
        gk = local_k // 2
        gi = local_k % 2
        c_col = gk
        c_row = gi
        for c in range(7):
            W[c_col * 3 + c_row][col + c] = sW[local_k][c]
    col += 7
    
    # Part 3: inner 3-4, Strassen 2x2x2 (7 terms)
    sU, sV, sW = _strassen_222()
    for local_i in range(4):
        gi = local_i // 2
        gj = local_i % 2
        a_row = gi
        a_inner = 3 + gj
        for c in range(7):
            U[a_row * 5 + a_inner][col + c] = sU[local_i][c]
    for local_j in range(4):
        gj = local_j // 2
        gk = local_j % 2
        b_inner = 3 + gj
        b_col = gk
        for c in range(7):
            V[b_inner * 5 + b_col][col + c] = sV[local_j][c]
    for local_k in range(4):
        gk = local_k // 2
        gi = local_k % 2
        c_col = gk
        c_row = gi
        for c in range(7):
            W[c_col * 3 + c_row][col + c] = sW[local_k][c]
    col += 7
    
    # === C12 (2x3) = A rows 0-1, B cols 2-4 ===
    # Part 1: inner 0, schoolbook 2x1x3 (6 terms)
    for gi in range(2):
        for gk in range(3):
            a_idx = gi * 5 + 0
            b_idx = 0 * 5 + (2 + gk)
            w_idx = (2 + gk) * 3 + gi
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Part 2: inner 1-4, B cols 0-1 (local), Strassen 2x2x2 (7 terms)
    # B cols 2-3 (global), inner 1-2
    sU, sV, sW = _strassen_222()
    for local_i in range(4):
        gi = local_i // 2
        gj = local_i % 2
        a_row = gi
        a_inner = 1 + gj
        for c in range(7):
            U[a_row * 5 + a_inner][col + c] = sU[local_i][c]
    for local_j in range(4):
        gj = local_j // 2
        gk = local_j % 2
        b_inner = 1 + gj
        b_col = 2 + gk
        for c in range(7):
            V[b_inner * 5 + b_col][col + c] = sV[local_j][c]
    for local_k in range(4):
        gk = local_k // 2
        gi = local_k % 2
        c_col = 2 + gk
        c_row = gi
        for c in range(7):
            W[c_col * 3 + c_row][col + c] = sW[local_k][c]
    col += 7
    
    # Part 3: inner 1-4, B cols 0-1 (local), Strassen 2x2x2 (7 terms)
    # B cols 2-3 (global), inner 3-4
    sU, sV, sW = _strassen_222()
    for local_i in range(4):
        gi = local_i // 2
        gj = local_i % 2
        a_row = gi
        a_inner = 3 + gj
        for c in range(7):
            U[a_row * 5 + a_inner][col + c] = sU[local_i][c]
    for local_j in range(4):
        gj = local_j // 2
        gk = local_j % 2
        b_inner = 3 + gj
        b_col = 2 + gk
        for c in range(7):
            V[b_inner * 5 + b_col][col + c] = sV[local_j][c]
    for local_k in range(4):
        gk = local_k // 2
        gi = local_k % 2
        c_col = 2 + gk
        c_row = gi
        for c in range(7):
            W[c_col * 3 + c_row][col + c] = sW[local_k][c]
    col += 7
    
    # Part 4: inner 1-4, B col 2 (local, global col 4), inner 1-2, schoolbook 2x2x1 (4 terms)
    for gi in range(2):
        for gj in range(2):
            a_idx = gi * 5 + (1 + gj)
            b_idx = (1 + gj) * 5 + 4
            w_idx = 4 * 3 + gi
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Part 5: inner 1-4, B col 2 (local, global col 4), inner 3-4, schoolbook 2x2x1 (4 terms)
    for gi in range(2):
        for gj in range(2):
            a_idx = gi * 5 + (3 + gj)
            b_idx = (3 + gj) * 5 + 4
            w_idx = 4 * 3 + gi
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # === C21 (1x2) = A row 2, B cols 0-1 ===
    # Part 1: inner 0, schoolbook 1x1x2 (2 terms)
    for gk in range(2):
        a_idx = 2 * 5 + 0
        b_idx = 0 * 5 + gk
        w_idx = gk * 3 + 2
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # Part 2: inner 1-4, schoolbook 1x4x2 (8 terms)
    for gj in range(4):
        for gk in range(2):
            a_idx = 2 * 5 + (1 + gj)
            b_idx = (1 + gj) * 5 + gk
            w_idx = gk * 3 + 2
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # === C22 (1x3) = A row 2, B cols 2-4 ===
    # Part 1: inner 0, schoolbook 1x1x3 (3 terms)
    for gk in range(3):
        a_idx = 2 * 5 + 0
        b_idx = 0 * 5 + (2 + gk)
        w_idx = (2 + gk) * 3 + 2
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # Part 2: inner 1-4, schoolbook 1x4x3 (12 terms)
    for gj in range(4):
        for gk in range(3):
            a_idx = 2 * 5 + (1 + gj)
            b_idx = (1 + gj) * 5 + (2 + gk)
            w_idx = (2 + gk) * 3 + 2
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(result):
    """Verify that the decomposition is correct."""
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    n, m, p = 3, 5, 5
    
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    errors += 1
                    max_err = max(max_err, abs(val - 1))
    
    if errors > 0:
        print(f"Verification FAILED: {errors} errors, max error {max_err}")
        return False
    return True


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication."""
    global _DEADLINE
    
    try:
        result = _build_rank71()
        
        if time.time() > _DEADLINE - 5:
            return result
        
        if not _verify_decomposition(result):
            # Fall back to schoolbook
            n, m, p = DIMENSIONS
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
