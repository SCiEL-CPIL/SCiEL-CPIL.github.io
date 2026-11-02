# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import lstsq

def _build_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Left half (k=0,1) - Strassen
    # M1_L = (A00+A11)(B00+B11) -> C00+=, C11+=
    c = 0
    U[0, c] = 1; U[3, c] = 1
    V[0, c] = 1; V[5, c] = 1
    W[0, c] = 1; W[3, c] = 1
    
    # M2_L = (A10+A11)B00 -> C10+=, C11-=
    c = 1
    U[2, c] = 1; U[3, c] = 1
    V[0, c] = 1
    W[1, c] = 1; W[3, c] = -1
    
    # M3_L = A00(B01-B11) -> C01+=, C11+=
    c = 2
    U[0, c] = 1
    V[1, c] = 1; V[5, c] = -1
    W[2, c] = 1; W[3, c] = 1
    
    # M4_L = A11(B10-B00) -> C00+=, C10+=
    c = 3
    U[3, c] = 1
    V[4, c] = 1; V[0, c] = -1
    W[0, c] = 1; W[1, c] = 1
    
    # M5_L = (A00+A01)B11 -> C00-=, C01+=
    c = 4
    U[0, c] = 1; U[1, c] = 1
    V[5, c] = 1
    W[0, c] = -1; W[2, c] = 1
    
    # M6_L = (A10-A00)(B00+B01) -> C11+=
    c = 5
    U[2, c] = 1; U[0, c] = -1
    V[0, c] = 1; V[1, c] = 1
    W[3, c] = 1
    
    # M7_L = (A01-A11)(B10+B11) -> C00+=
    c = 6
    U[1, c] = 1; U[3, c] = -1
    V[4, c] = 1; V[5, c] = 1
    W[0, c] = 1
    
    # Right half (k=2,3) - Strassen
    # M1_R = (A00+A11)(B02+B13) -> C02+=, C13+=
    c = 7
    U[0, c] = 1; U[3, c] = 1
    V[2, c] = 1; V[7, c] = 1
    W[4, c] = 1; W[7, c] = 1
    
    # M2_R = (A10+A11)B02 -> C12+=, C13-=
    c = 8
    U[2, c] = 1; U[3, c] = 1
    V[2, c] = 1
    W[5, c] = 1; W[7, c] = -1
    
    # M3_R = A00(B03-B13) -> C03+=, C13+=
    c = 9
    U[0, c] = 1
    V[3, c] = 1; V[7, c] = -1
    W[6, c] = 1; W[7, c] = 1
    
    # M4_R = A11(B12-B02) -> C02+=, C12+=
    c = 10
    U[3, c] = 1
    V[6, c] = 1; V[2, c] = -1
    W[4, c] = 1; W[5, c] = 1
    
    # M5_R = (A00+A01)B13 -> C02-=, C03+=
    c = 11
    U[0, c] = 1; U[1, c] = 1
    V[7, c] = 1
    W[4, c] = -1; W[6, c] = 1
    
    # M6_R = (A10-A00)(B02+B03) -> C13+=
    c = 12
    U[2, c] = 1; U[0, c] = -1
    V[2, c] = 1; V[3, c] = 1
    W[7, c] = 1
    
    # M7_R = (A01-A11)(B12+B13) -> C02+=
    c = 13
    U[1, c] = 1; U[3, c] = -1
    V[6, c] = 1; V[7, c] = 1
    W[4, c] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    # Target tensor T[i, k, i', m', m'', l''] = 1 if i==i', k==l'', m==m'' else 0
    # Flatten indices:
    # i in 0..1, k in 0..3 -> idx_c = k*n + i (0..7)
    # i' in 0..1, m' in 0..1 -> idx_a = i'*m + m' (0..3)
    # m'' in 0..1, l'' in 0..3 -> idx_b = m''*p + l'' (0..7)
    
    T = np.zeros((p*n, n*m, m*p))
    for i in range(n):
        for k in range(p):
            for m in range(m):
                # C[i,k] depends on A[i,m] and B[m,k]
                # So T[idx_c, idx_a, idx_b] = 1
                idx_c = k * n + i
                idx_a = i * m + m
                idx_b = m * p + k
                T[idx_c, idx_a, idx_b] = 1
                
    T_flat = T.flatten()
    
    # Decomposition: sum_r W_r (U_r^T A) (V_r^T B)
    # Flatten: sum_r W_r (U_r \otimes V_r)
    # We need to match T_flat.
    # T_flat index is (idx_c, idx_a, idx_b).
    # Term r contributes W_r[idx_c] * U_r[idx_a] * V_r[idx_b].
    
    error = 0.0
    for r in range(rank):
        pass # We can just compute the full tensor sum
    
    # Compute sum_r W_r \otimes U_r \otimes V_r
    # This is a tensor of shape (8, 4, 8)
    S = np.zeros((p*n, n*m, m*p))
    for r in range(rank):
        S += np.outer(W[:, r], np.outer(U[:, r], V[:, r]))
        # Wait, np.outer(np.outer(U, V), W) gives (4, 8, 8)
        # We need (8, 4, 8) which is (C, A, B)
        # S[idx_c, idx_a, idx_b] = W_r[idx_c] * U_r[idx_a] * V_r[idx_b]
        
    S_flat = S.flatten()
    error = np.linalg.norm(T_flat - S_flat)
    return error

def _try_rank13(deadline):
    n, m, p = 2, 2, 4
    rank = 13
    
    # Initialize from rank 14, drop last column, randomize slightly
    U14, V14, W14 = _build_rank14()
    U = U14[:, :rank].copy()
    V = V14[:, :rank].copy()
    W = W14[:, :rank].copy()
    
    # Add random noise
    U += np.random.normal(0, 0.1, U.shape)
    V += np.random.normal(0, 0.1, V.shape)
    W += np.random.normal(0, 0.1, W.shape)
    
    # Precompute target tensor flat
    T = np.zeros((p*n, n*m, m*p))
    for i in range(n):
        for k in range(p):
            for m in range(m):
                idx_c = k * n + i
                idx_a = i * m + m
                idx_b = m * p + k
                T[idx_c, idx_a, idx_b] = 1
    T_flat = T.flatten()
    
    best_error = float('inf')
    best_U, best_V, best_W = U, V, W
    
    for it in range(500):
        if time.time() > deadline:
            break
            
        # Update U
        # Minimize || T - sum_r W_r (U_r \otimes V_r) ||
        # Linear in U_r.
        # Construct matrix M_U such that M_U vec(U) = T_flat
        # M_U has shape (256, 4*13)
        # Column for U[i', m', r] is T_flat index (idx_c, i'*m+m', idx_b) mapped to W_r[idx_c] * V_r[idx_b]
        
        M_U = np.zeros((p*n * n*m * m*p, n*m * rank))
        for r in range(rank):
            for ia in range(n*m):
                for ic in range(p*n):
                    for ib in range(m*p):
                        idx_flat = (ic * n*m + ia) * m*p + ib
                        col_idx = r * n*m + ia
                        M_U[idx_flat, col_idx] = W[ic, r] * V[ib, r]
                        
        # Solve M_U vec(U) = T_flat
        sol_U, residuals, rank_u, s = lstsq(M_U, T_flat, cond=None)
        U = sol_U.reshape(rank, n*m).T
        
        # Update V
        M_V = np.zeros((p*n * n*m * m*p, m*p * rank))
        for r in range(rank):
            for ib in range(m*p):
                for ic in range(p*n):
                    for ia in range(n*m):
                        idx_flat = (ic * n*m + ia) * m*p + ib
                        col_idx = r * m*p + ib
                        M_V[idx_flat, col_idx] = W[ic, r] * U[ia, r]
                        
        sol_V, residuals, rank_v, s = lstsq(M_V, T_flat, cond=None)
        V = sol_V.reshape(rank, m*p).T
        
        # Update W
        M_W = np.zeros((p*n * n*m * m*p, p*n * rank))
        for r in range(rank):
            for ic in range(p*n):
                for ia in range(n*m):
                    for ib in range(m*p):
                        idx_flat = (ic * n*m + ia) * m*p + ib
                        col_idx = r * p*n + ic
                        M_W[idx_flat, col_idx] = U[ia, r] * V[ib, r]
                        
        sol_W, residuals, rank_w, s = lstsq(M_W, T_flat, cond=None)
        W = sol_W.reshape(rank, p*n).T
        
        error = np.linalg.norm(T_flat - (W[:, :, None, None] * U[None, :, None, :] * V[None, None, :, :]).sum(axis=0).flatten())
        # Wait, the multiplication order in numpy broadcasting:
        # W is (8, 13), U is (4, 13), V is (8, 13)
        # We want S[ic, ia, ib] = sum_r W[ic, r] U[ia, r] V[ib, r]
        # S = (W[:, :, None, None] * U[None, :, None, :] * V[None, None, :, :]).sum(axis=0)
        # This works.
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error < 1e-10:
            break
            
    if best_error < 1e-10:
        # Check for rationality
        # Try to snap to integers or half-integers
        U_int = np.round(best_U * 2) / 2
        V_int = np.round(best_V * 2) / 2
        W_int = np.round(best_W * 2) / 2
        
        error_int = np.linalg.norm(T_flat - (W_int[:, :, None, None] * U_int[None, :, None, :] * V_int[None, None, :, :]).sum(axis=0).flatten())
        
        if error_int < 1e-10:
            return U_int, V_int, W_int, True
        else:
            # Maybe integers?
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            error_int = np.linalg.norm(T_flat - (W_int[:, :, None, None] * U_int[None, :, None, :] * V_int[None, None, :, :]).sum(axis=0).flatten())
            if error_int < 1e-10:
                return U_int, V_int, W_int, True
            else:
                return best_U, best_V, best_W, False
    else:
        return None, None, None, False

def construct():
    deadline = time.time() + 140
    
    # Try rank 13
    U13, V13, W13, found = _try_rank13(deadline)
    if found:
        return {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}
        
    # Fallback to rank 14
    U14, V14, W14 = _build_rank14()
    return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
# EVOLVE-BLOCK-END
