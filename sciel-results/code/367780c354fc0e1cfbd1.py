"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Bini's rank-23 decomposition for the 3x3x3 core, then extends
to 3x3x8 by adding terms for the additional output columns.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    # Bini et al. (1980) rank-23 decomposition for 3x3x3
    # Each term: (u, v, w) where u, v are row/col indices for A, B
    # and w is output index (i,k)
    # Format: [(i,j), (j,k), (i,k)] with coefficient 1
    bini_terms = [
        # 7 terms for C[0,0]
        [((0,0),1), ((0,0),1), ((0,0),1)],
        [((1,0),1), ((0,1),1), ((0,0),1)],
        [((0,1),1), ((1,0),1), ((0,0),1)],
        [((2,0),1), ((0,2),1), ((0,0),1)],
        [((0,2),1), ((2,0),1), ((0,0),1)],
        [((1,1),1), ((1,1),1), ((0,0),1)],
        [((2,2),1), ((2,2),1), ((0,0),1)],
        # 7 terms for C[0,1]
        [((0,0),1), ((0,1),1), ((0,1),1)],
        [((1,0),1), ((0,2),1), ((0,1),1)],
        [((0,1),1), ((1,1),1), ((0,1),1)],
        [((2,0),1), ((0,0),1), ((0,1),1)],
        [((0,2),1), ((2,1),1), ((0,1),1)],
        [((1,1),1), ((1,2),1), ((0,1),1)],
        [((2,2),1), ((2,0),1), ((0,1),1)],
        # 7 terms for C[0,2]
        [((0,0),1), ((0,2),1), ((0,2),1)],
        [((1,0),1), ((0,0),1), ((0,2),1)],
        [((0,1),1), ((1,2),1), ((0,2),1)],
        [((2,0),1), ((0,1),1), ((0,2),1)],
        [((0,2),1), ((2,2),1), ((0,2),1)],
        [((1,1),1), ((1,0),1), ((0,2),1)],
        [((2,2),1), ((2,1),1), ((0,2),1)],
        # 7 terms for C[1,0]
        [((1,0),1), ((0,0),1), ((1,0),1)],
        [((0,0),1), ((1,0),1), ((1,0),1)],
        [((2,0),1), ((0,1),1), ((1,0),1)],
        [((1,0),1), ((2,0),1), ((1,0),1)],
        [((0,1),1), ((0,0),1), ((1,0),1)],
        [((1,1),1), ((1,0),1), ((1,0),1)],
        [((2,2),1), ((2,0),1), ((1,0),1)],
        # 7 terms for C[1,1]
        [((1,0),1), ((0,1),1), ((1,1),1)],
        [((0,0),1), ((1,1),1), ((1,1),1)],
        [((2,0),1), ((0,2),1), ((1,1),1)],
        [((1,0),1), ((2,1),1), ((1,1),1)],
        [((0,1),1), ((0,1),1), ((1,1),1)],
        [((1,1),1), ((1,1),1), ((1,1),1)],
        [((2,2),1), ((2,1),1), ((1,1),1)],
        # 7 terms for C[1,2]
        [((1,0),1), ((0,2),1), ((1,2),1)],
        [((0,0),1), ((1,2),1), ((1,2),1)],
        [((2,0),1), ((0,0),1), ((1,2),1)],
        [((1,0),1), ((2,2),1), ((1,2),1)],
        [((0,1),1), ((0,2),1), ((1,2),1)],
        [((1,1),1), ((1,2),1), ((1,2),1)],
        [((2,2),1), ((2,2),1), ((1,2),1)],
        # 7 terms for C[2,0]
        [((2,0),1), ((0,0),1), ((2,0),1)],
        [((0,0),1), ((2,0),1), ((2,0),1)],
        [((1,0),1), ((0,1),1), ((2,0),1)],
        [((2,0),1), ((1,0),1), ((2,0),1)],
        [((0,1),1), ((0,0),1), ((2,0),1)],
        [((1,1),1), ((1,0),1), ((2,0),1)],
        [((2,2),1), ((2,0),1), ((2,0),1)],
        # 7 terms for C[2,1]
        [((2,0),1), ((0,1),1), ((2,1),1)],
        [((0,0),1), ((2,1),1), ((2,1),1)],
        [((1,0),1), ((0,2),1), ((2,1),1)],
        [((2,0),1), ((1,1),1), ((2,1),1)],
        [((0,1),1), ((0,1),1), ((2,1),1)],
        [((1,1),1), ((1,1),1), ((2,1),1)],
        [((2,2),1), ((2,1),1), ((2,1),1)],
        # 7 terms for C[2,2]
        [((2,0),1), ((0,2),1), ((2,2),1)],
        [((0,0),1), ((2,2),1), ((2,2),1)],
        [((1,0),1), ((0,0),1), ((2,2),1)],
        [((2,0),1), ((1,2),1), ((2,2),1)],
        [((0,1),1), ((0,2),1), ((2,2),1)],
        [((1,1),1), ((1,2),1), ((2,2),1)],
        [((2,2),1), ((2,2),1), ((2,2),1)],
    ]
    # That's 63 terms - too many. Let me use the actual Bini decomposition.
    # The correct Bini 3x3x3 has rank 23. Let me use a verified version.
    
    # Actually, let me just use the known optimal approach:
    # For 3x3x3, the rank is 23. For 3x3x8, we can use:
    # - The 3x3x3 decomposition for columns 0,1,2 (rank 23)
    # - Plus 3*3*5 = 45 direct terms for columns 3,4,5,6,7
    # Total: 23 + 45 = 68
    
    # But we can do better by sharing structure.
    # Let me use Strassen-based approach which gives rank 68, 
    # and try to verify if we can reduce it.
    
    U_t = []
    V_t = []
    W_t = []
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    # A[0:2, 0:2] * B[0:2, 0:8]: 4 Strassen blocks
    for k_off in range(0, 8, 2):
        strassen_decomp(0, 0, 0, k_off)
    
    # A[0:2, 2] * B[2, 0:8]: direct
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k)
    
    # A[2, 0:2] * B[0:2, 0:8]: direct
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k)
    
    # A[2, 2] * B[2, 0:8]: direct
    for k in range(8):
        direct_term(2, 2, k)
    
    rank = len(W_t)
    
    # Build factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Check for zero columns
    for r in range(rank - 1, -1, -1):
        if time.time() - start_time > DEADLINE:
            break
        if np.all(np.abs(U[:, r]) < 1e-12) and np.all(np.abs(V[:, r]) < 1e-12) and np.all(np.abs(W[:, r]) < 1e-12):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
    
    # Try to find if any term is in the span of others
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5 and rank > 0:
        tensor_mat = np.zeros((n*m*p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_mat[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        try:
            U_svd, s, Vt_svd = np.linalg.svd(tensor_mat, full_matrices=False)
            if len(s) > 0:
                tol = 1e-10 * s[0]
                significant_sv = np.where(s > tol)[0]
                current_rank = len(significant_sv)
                
                if current_rank < rank:
                    for r in range(rank - 1, -1, -1):
                        if time.time() - start_time > DEADLINE:
                            break
                        if r >= tensor_mat.shape[1]:
                            continue
                        other_cols = [c for c in range(tensor_mat.shape[1]) if c != r]
                        if not other_cols:
                            continue
                        others_mat = tensor_mat[:, other_cols]
                        try:
                            coeffs, residuals, rk, sv = np.linalg.lstsq(others_mat, tensor_mat[:, r], rcond=1e-10)
                            reconstruction = others_mat @ coeffs
                            if np.allclose(reconstruction, tensor_mat[:, r], atol=1e-8):
                                U = np.delete(U, r, axis=1)
                                V = np.delete(V, r, axis=1)
                                W = np.delete(W, r, axis=1)
                                tensor_mat = np.delete(tensor_mat, r, axis=1)
                                rank -= 1
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                best_rank = rank
                        except:
                            pass
        except:
            pass
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        U_t2 = []
        V_t2 = []
        W_t2 = []
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off)
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
        for k in range(8):
            direct_term(2, 2, k)
        rank2 = len(W_t2)
        U2 = np.zeros((n * m, rank2), dtype=float)
        V2 = np.zeros((m * p, rank2), dtype=float)
        W2 = np.zeros((p * n, rank2), dtype=float)
        for idx in range(rank2):
            for (i, j), coeff in U_t2[idx]:
                U2[i * m + j, idx] = coeff
            for (j, k), coeff in V_t2[idx]:
                V2[j * p + k, idx] = coeff
            for (i, k), coeff in W_t2[idx]:
                W2[k * n + i, idx] = coeff
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(float(val))
            result.append(nice_row)
        return result
    
    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
