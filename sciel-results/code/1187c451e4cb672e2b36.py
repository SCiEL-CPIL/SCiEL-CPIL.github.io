import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Output: C[i,k] = sum_r W[k*2+i, r] * M_r
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a rank-51 decomposition for 3x3 * 3x6 using block Strassen."""
    U_list = []
    V_list = []
    W_list = []
    
    U_s, V_s, W_s = _strassen_2x2()
    
    for block in range(3):
        b_start = 2 * block
        
        # Strassen for A[0:2,0:2] * B[0:2, b_start:b_start+2]
        for r in range(7):
            U_term = np.zeros(9)
            V_term = np.zeros(18)
            W_term = np.zeros(18)
            
            for af in range(4):
                ai, aj = af // 2, af % 2
                U_term[ai * 3 + aj] = U_s[af, r]
            
            for bf in range(4):
                bj, bk = bf // 2, bf % 2
                V_term[bj * 6 + b_start + bk] = V_s[bf, r]
            
            for cf in range(4):
                ci, ck = cf // 2, cf % 2
                W_term[(b_start + ck) * 3 + ci] = W_s[cf, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # A[0:2,2] * B[2, b_start:b_start+2]: 2x1 * 1x2 -> 4 terms
        for i in range(2):
            for k in range(2):
                U_term = np.zeros(9)
                V_term = np.zeros(18)
                W_term = np.zeros(18)
                gk = b_start + k
                U_term[i * 3 + 2] = 1.0
                V_term[2 * 6 + gk] = 1.0
                W_term[gk * 3 + i] = 1.0
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # A[2,0:2] * B[0:2, b_start:b_start+2]: 1x2 * 2x2 -> 4 terms
        for jj in range(2):
            for k in range(2):
                U_term = np.zeros(9)
                V_term = np.zeros(18)
                W_term = np.zeros(18)
                gk = b_start + k
                U_term[2 * 3 + jj] = 1.0
                V_term[jj * 6 + gk] = 1.0
                W_term[gk * 3 + 2] = 1.0
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # A[2,2] * B[2, b_start:b_start+2]: 1x1 * 1x2 -> 2 terms
        for k in range(2):
            U_term = np.zeros(9)
            V_term = np.zeros(18)
            W_term = np.zeros(18)
            gk = b_start + k
            U_term[2 * 3 + 2] = 1.0
            V_term[2 * 6 + gk] = 1.0
            W_term[gk * 3 + 2] = 1.0
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all zero entries
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                i1, j1 = a // m, a % m
                j2, k1 = b // p, b % p
                k2, i2 = c // n, c % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s) > tol:
                    return False
    
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=10000, tol=1e-9):
    """Try to reduce rank by finding linear dependencies."""
    R = U.shape[1]
    
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_R = R
    
    eliminated = set()
    
    for target in range(R):
        if target in eliminated:
            continue
        
        others = [r for r in range(R) if r != target and r not in eliminated]
        
        if not others:
            continue
        
        n_eq = a_dim * b_dim * c_dim
        n_unk = len(others)
        
        if n_unk > 40:
            continue
        
        A_sys = np.zeros((n_eq, n_unk))
        rhs = np.zeros(n_eq)
        
        for a in range(a_dim):
            for b in range(b_dim):
                for c in range(c_dim):
                    eq = (a * b_dim + b) * c_dim + c
                    rhs[eq] = U[a, target] * V[b, target] * W[c, target]
                    for col_idx, r in enumerate(others):
                        A_sys[eq, col_idx] = U[a, r] * V[b, r] * W[c, r]
        
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
        except:
            continue
        
        residual = np.linalg.norm(A_sys @ alpha - rhs)
        norm_rhs = np.linalg.norm(rhs)
        
        if norm_rhs > 0 and residual / norm_rhs < 1e-8:
            eliminated.add(target)
            current_R -= 1
    
    if current_R < R:
        keep = [r for r in range(R) if r not in eliminated]
        new_U = U[:, keep]
        new_V = V[:, keep]
        new_W = W[:, keep]
        return new_U, new_V, new_W
    
    return U, V, W

def _try_shared_block_decomposition(n, m, p, deadline):
    """
    Try to construct a lower-rank decomposition by splitting B into two 3x3 blocks
    and sharing U factors. For each 3x3*3x3, we build a decomposition and try to
    reduce its rank. Total rank = rank_of_3x3_decomp (shared U, W replicated in V).
    """
    if time.time() > deadline - 30:
        return None
    
    # Build schoolbook 3x3*3x3 decomposition (rank 27)
    R3 = 27
    U3 = np.zeros((9, R3))
    V3 = np.zeros((9, R3))
    W3 = np.zeros((9, R3))
    
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, r] = 1.0
                V3[j*3+k, r] = 1.0
                W3[k*3+i, r] = 1.0
                r += 1
    
    # Try to reduce the 3x3*3x3 rank
    if time.time() > deadline - 25:
        # Just use schoolbook for 3x3*3x3, total rank = 27
        # Construct full decomposition
        U_full = np.zeros((9, R3))
        V_full = np.zeros((18, R3))
        W_full = np.zeros((18, R3))
        
        U_full[:] = U3
        # V: first 9 rows for B1 (cols 0-2), next 9 rows for B2 (cols 3-5)
        for r_idx in range(R3):
            for j in range(3):
                for k in range(3):
                    V_full[j*3+k, r_idx] = V3[j*3+k, r_idx]
                    V_full[9 + j*3 + k, r_idx] = V3[j*3+k, r_idx]
        
        # W: first 9 rows for C1 (cols 0-2), next 9 rows for C2 (cols 3-5)
        for r_idx in range(R3):
            for k in range(3):
                for i in range(3):
                    W_full[k*3+i, r_idx] = W3[k*3+i, r_idx]
                    W_full[9 + k*3 + i, r_idx] = W3[k*3+i, r_idx]
        
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            return U_full, V_full, W_full
        return None
    
    # Try to reduce 3x3*3x3 rank
    U3_red, V3_red, W3_red = _try_reduce_rank(U3, V3, W3, 3, 3, 3)
    R3_red = U3_red.shape[1]
    
    if R3_red >= R3:
        return None
    
    # Construct full decomposition with reduced 3x3*3x3
    U_full = np.zeros((9, R3_red))
    V_full = np.zeros((18, R3_red))
    W_full = np.zeros((18, R3_red))
    
    U_full[:] = U3_red
    for r_idx in range(R3_red):
        for j in range(3):
            for k in range(3):
                V_full[j*3+k, r_idx] = V3_red[j*3+k, r_idx]
                V_full[9 + j*3 + k, r_idx] = V3_red[j*3+k, r_idx]
    
    for r_idx in range(R3_red):
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, r_idx] = W3_red[k*3+i, r_idx]
                W_full[9 + k*3 + i, r_idx] = W3_red[k*3+i, r_idx]
    
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return U_full, V_full, W_full
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Block Strassen (rank 51)
    try:
        U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            rank1 = U.shape[1]
            if rank1 < best_rank:
                best_rank = rank1
                best_result = (U, V, W)
            # Try to reduce
            if time.time() < deadline - 40:
                U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p)
                if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                    rank1r = U_red.shape[1]
                    if rank1r < best_rank:
                        best_rank = rank1r
                        best_result = (U_red, V_red, W_red)
    except Exception:
        pass
    
    # Approach 2: Shared block decomposition (split B into two 3x3, share U)
    try:
        result2 = _try_shared_block_decomposition(n, m, p, deadline)
        if result2 is not None:
            U2, V2, W2 = result2
            rank2 = U2.shape[1]
            if rank2 < best_rank:
                best_rank = rank2
                best_result = (U2, V2, W2)
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 54)
    if best_result is None:
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
