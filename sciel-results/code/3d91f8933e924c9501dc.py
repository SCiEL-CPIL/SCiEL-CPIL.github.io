import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W decompose the tensor T[i,j,k]=1.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    if not U or not V or not W:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        if not np.allclose(result, 1.0, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _round_to_half_integer(arr, tol=1e-9):
    """
    Round entries to nearest half-integer.
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return rounded / 2.0

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen decomposition
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[0][5] = -1; U[2][5] = 1
    U[1][6] = 1; U[3][6] = -1
    
    V[1][0] = 1; V[2][0] = 1
    V[0][1] = 1
    V[0][2] = 1; V[2][2] = -1
    V[1][3] = -1; V[3][3] = 1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Returns U, V, W for 2x2x1 matrix multiplication (Rank 4).
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _block_decompose_2x4x7(rank_227=None):
    """
    Decomposes 2x4x7 using block structure.
    If rank_227 is provided, it should be a tuple (U, V, W) for 2x2x7.
    Otherwise, it uses the standard 3x(2x2x2) + 1x(2x2x1) construction (Rank 25).
    """
    if rank_227 is None:
        U_222, V_222, W_222 = _strassen_2x2x2()
        U_221, V_221, W_221 = _schoolbook_2x2x1()
        
        # Assemble 2x2x7
        U_227 = [[0]*25 for _ in range(4)]
        V_227 = [[0]*25 for _ in range(14)]
        W_227 = [[0]*25 for _ in range(14)]
        
        def add_sub(U_sub, V_sub, W_sub, col_offset, u_rows, v_rows, w_rows):
            r_sub = len(U_sub[0])
            for i_sub in range(len(U_sub)):
                for c in range(r_sub):
                    U_227[u_rows[i_sub]][col_offset + c] += U_sub[i_sub][c]
            for j_sub in range(len(V_sub)):
                for c in range(r_sub):
                    V_227[v_rows[j_sub]][col_offset + c] += V_sub[j_sub][c]
            for k_sub in range(len(W_sub)):
                for c in range(r_sub):
                    W_227[w_rows[k_sub]][col_offset + c] += W_sub[k_sub][c]
            return r_sub
            
        col_off = 0
        # B1_0 (cols 0-1)
        col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [0, 1, 7, 8], [0, 1, 2, 3])
        # B1_1 (cols 2-3)
        col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [2, 3, 9, 10], [4, 5, 6, 7])
        # B1_2 (cols 4-5)
        col_off += add_sub(U_222, V_222, W_222, col_off, [0, 1, 2, 3], [4, 5, 11, 12], [8, 9, 10, 11])
        # B1_3 (col 6)
        col_off += add_sub(U_221, V_221, W_221, col_off, [0, 1, 2, 3], [6, 13], [12, 13])
        
        U_227, V_227, W_227 = U_227, V_227, W_227
    else:
        U_227, V_227, W_227 = rank_227
        r_227 = len(U_227[0])
        
    # Now assemble 2x4x7
    # A (2x4) = [A1 A2]
    # B (4x7) = [B1; B2]
    # C = A1 B1 + A2 B2
    
    r_227 = len(U_227[0])
    R_total = 2 * r_227
    
    U = [[0]*R_total for _ in range(8)]
    V = [[0]*R_total for _ in range(28)]
    W = [[0]*R_total for _ in range(14)]
    
    def add_sub_global(U_sub, V_sub, W_sub, col_offset, u_rows, v_rows, w_rows):
        r_sub = len(U_sub[0])
        for i_sub in range(len(U_sub)):
            for c in range(r_sub):
                U[u_rows[i_sub]][col_offset + c] += U_sub[i_sub][c]
        for j_sub in range(len(V_sub)):
            for c in range(r_sub):
                V[v_rows[j_sub]][col_offset + c] += V_sub[j_sub][c]
        for k_sub in range(len(W_sub)):
            for c in range(r_sub):
                W[w_rows[k_sub]][col_offset + c] += W_sub[k_sub][c]
        return r_sub
        
    # Block 1: A1 * B1
    # A1 rows in U: [0, 1, 2, 3]
    # B1 rows in V: [0..13]
    # C rows in W: [0..13]
    col_off = 0
    col_off += add_sub_global(U_227, V_227, W_227, col_off, 
                              [0, 1, 2, 3], 
                              list(range(14)), 
                              list(range(14)))
    
    # Block 2: A2 * B2
    # A2 rows in U: [4, 5, 6, 7]
    # B2 rows in V: [14..27]
    # C rows in W: [0..13] (same output)
    col_off += add_sub_global(U_227, V_227, W_227, col_off, 
                              [4, 5, 6, 7], 
                              list(range(14, 28)), 
                              list(range(14)))
    
    return U, V, W

def _als_decompose_2x2x7(rank, max_iter=500, tol=1e-8, seed=42, time_limit=10.0, start_time=0.0):
    """
    ALS for 2x2x7 tensor decomposition.
    T[i,j,k] = 1.
    """
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((4, rank))
    V = rng.standard_normal((14, rank))
    W = rng.standard_normal((14, rank))
    
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    T = np.ones((2, 2, 7), dtype=np.float64)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        for i in range(2):
            for j in range(2):
                A = np.zeros((rank, 7), dtype=np.float64)
                for k in range(7):
                    A[:, k] = V[j * 7 + k, :] * W[k * 2 + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, T[i, j, :].T, rcond=None)
                    U[i * 2 + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(2):
            for k in range(7):
                B = np.zeros((rank, 2), dtype=np.float64)
                for i in range(2):
                    B[:, i] = U[i * 2 + j, :] * W[k * 2 + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(B, T[:, j, k].T, rcond=None)
                    V[j * 7 + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(7):
            for i in range(2):
                C = np.zeros((rank, 2), dtype=np.float64)
                for j in range(2):
                    C[:, j] = U[i * 2 + j, :] * V[j * 7 + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(C, T[i, :, k].T, rcond=None)
                    W[k * 2 + i, :] = x
                except:
                    pass
        
        if it % 10 == 0:
            U3 = U.reshape(2, 2, rank)
            V3 = V.reshape(2, 7, rank)
            W3 = W.reshape(7, 2, rank)
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
            err = np.linalg.norm(recon - T)
            if err < tol:
                break
                
    U3 = U.reshape(2, 2, rank)
    V3 = V.reshape(2, 7, rank)
    W3 = W.reshape(7, 2, rank)
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    err = np.linalg.norm(recon - T)
    
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0
    
    best_result = None
    best_rank = float('inf')
    
    # 1. Try to find a lower rank decomposition for M(2,2,7)
    # Current known is 25. Try 24, 23...
    found_227 = None
    for rank_227 in range(24, 22, -1):
        if time.time() - start_time > time_limit - 10.0:
            break
        for restart in range(5):
            if time.time() - start_time > time_limit - 10.0:
                break
            res = _als_decompose_2x2x7(rank_227, max_iter=200, seed=rank_227*10+restart, time_limit=5.0, start_time=start_time)
            if res is not None:
                U, V, W = res
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                # Verify 2x2x7
                Ua = np.array(U_r, dtype=np.float64)
                Va = np.array(V_r, dtype=np.float64)
                Wa = np.array(W_r, dtype=np.float64)
                U3 = Ua.reshape(2, 2, rank_227)
                V3 = Va.reshape(2, 7, rank_227)
                W3 = Wa.reshape(7, 2, rank_227)
                recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
                if np.allclose(recon, 1.0, atol=1e-9):
                    found_227 = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    break
        if found_227:
            break
            
    # 2. Assemble 2x4x7
    if found_227:
        U, V, W = _block_decompose_2x4x7(found_227)
        rank = 2 * len(found_227[0][0])
    else:
        # Fallback to Rank 50
        U, V, W = _block_decompose_2x4x7(None)
        rank = 50
        
    # Verify full decomposition
    if _verify(U, V, W, n, m, p, tol=1e-9):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    else:
        # Fallback to Schoolbook 56
        U_sb = [[0] * 56 for _ in range(8)]
        V_sb = [[0] * 56 for _ in range(28)]
        W_sb = [[0] * 56 for _ in range(14)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = 56

    return best_result
