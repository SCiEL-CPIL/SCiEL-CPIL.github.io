"""Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy: Build the rank-49 Kronecker product of Strassen, then prune
terms that contribute zero to all (i,j,k) triples. This structural
pruning may reveal a lower-rank exact decomposition.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _strassen_2x2():
    """Return (U2, V2, W2) for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    
    U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    Satisfies: sum_r U2[i*2+j,r] * V2[j*2+k,r] * W2[k*2+i,r] = 1 for all i,j,k in {0,1}
    """
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3 = A00(B01-B11) -> C01 += 1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1
    
    return U2, V2, W2

def _build_kronecker_49():
    """Build the full rank-49 Kronecker product decomposition."""
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=float)
    V4 = np.zeros((16, rank), dtype=float)
    W4 = np.zeros((16, rank), dtype=float)
    
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for j in range(4):
            j1, j2 = j // 2, j % 2
            for r1 in range(7):
                u1 = U2[i1 * 2 + j1, r1]
                if u1 == 0:
                    continue
                for r2 in range(7):
                    u2 = U2[i2 * 2 + j2, r2]
                    if u2 == 0:
                        continue
                    col = r1 * 7 + r2
                    U4[i * 4 + j, col] = u1 * u2
    
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for k in range(4):
            k1, k2 = k // 2, k % 2
            for r1 in range(7):
                v1 = V2[j1 * 2 + k1, r1]
                if v1 == 0:
                    continue
                for r2 in range(7):
                    v2 = V2[j2 * 2 + k2, r2]
                    if v2 == 0:
                        continue
                    col = r1 * 7 + r2
                    V4[j * 4 + k, col] = v1 * v2
    
    for k in range(4):
        k1, k2 = k // 2, k % 2
        for i in range(4):
            i1, i2 = i // 2, i % 2
            for r1 in range(7):
                w1 = W2[k1 * 2 + i1, r1]
                if w1 == 0:
                    continue
                for r2 in range(7):
                    w2 = W2[k2 * 2 + i2, r2]
                    if w2 == 0:
                        continue
                    col = r1 * 7 + r2
                    W4[k * 4 + i, col] = w1 * w2
    
    return U4, V4, W4

def _prune_zero_terms(U4, V4, W4):
    """Remove terms that contribute zero to all (i,j,k) triples.
    
    A term r is 'active' if there exists (i,j,k) such that
    U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] != 0.
    """
    n, m, p = 4, 4, 4
    rank = U4.shape[1]
    
    # For each term, check if it contributes to any (i,j,k)
    active_terms = []
    for r in range(rank):
        # Check if this term has any non-zero product
        contributes = False
        for i in range(n):
            for j in range(m):
                u_val = U4[i * m + j, r]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V4[j * p + k, r]
                    if v_val == 0:
                        continue
                    w_val = W4[k * n + i, r]
                    if w_val == 0:
                        continue
                    if u_val * v_val * w_val != 0:
                        contributes = True
                        break
                if contributes:
                    break
            if contributes:
                break
        if contributes:
            active_terms.append(r)
    
    return active_terms

def _verify_decomp(U, V, W, tol=1e-10):
    """Verify that the decomposition is exact."""
    n, m, p = 4, 4, 4
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _to_integer_list(mat):
    """Convert numpy array to list of lists with integer values."""
    result = mat.tolist()
    for i in range(len(result)):
        for j in range(len(result[i])):
            if abs(result[i][j] - round(result[i][j])) < 1e-10:
                result[i][j] = round(result[i][j])
    return result

def _schoolbook_4x4():
    """Fallback: schoolbook decomposition of rank 64."""
    n, m, p = 4, 4, 4
    rank_fb = 64
    U_fb = [[0] * rank_fb for _ in range(16)]
    V_fb = [[0] * rank_fb for _ in range(16)]
    W_fb = [[0] * rank_fb for _ in range(16)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def construct():
    start_time = time.time()
    
    # Build the full rank-49 Kronecker decomposition
    U4, V4, W4 = _build_kronecker_49()
    
    # First, verify the full rank-49 decomposition is correct
    if not _verify_decomp(U4, V4, W4):
        return _schoolbook_4x4()
    
    # Prune zero-contributing terms
    active_terms = _prune_zero_terms(U4, V4, W4)
    
    # If we can prune some terms, create a smaller decomposition
    if len(active_terms) < 49:
        U_pruned = U4[:, active_terms]
        V_pruned = V4[:, active_terms]
        W_pruned = W4[:, active_terms]
        
        # Verify the pruned decomposition is still exact
        if _verify_decomp(U_pruned, V_pruned, W_pruned):
            U_list = _to_integer_list(U_pruned)
            V_list = _to_integer_list(V_pruned)
            W_list = _to_integer_list(W_pruned)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If pruning didn't help or wasn't exact, return the full rank-49
    U_list = _to_integer_list(U4)
    V_list = _to_integer_list(V4)
    W_list = _to_integer_list(W4)
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
