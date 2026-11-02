"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: Use Kronecker product T_{2,2,4} (rank 8) ⊗ T_{2,2,2} (rank 7) = T_{4,4,8} (rank 56).
Also attempt peeling: fix U,V to sparse patterns, solve linear system for W.
"""

DIMENSIONS = [4, 4, 8]

import time
import numpy as np

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W


def _schoolbook_2x2x4():
    """
    Return U, V, W for 2x2 * 2x4 matrix multiplication with rank 8.
    U shape: (4, 8), V shape: (8, 8), W shape: (8, 8)
    n=2, m=2, p=4
    """
    n, m, p = 2, 2, 4
    rank = n * m * p  # 8
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _kronecker_4x4x8():
    """
    Construct rank-56 decomposition for 4x4 * 4x8 using Kronecker product:
    T_{4,4,8} = T_{2,2,4} ⊗ T_{2,2,2}
    
    Outer: n1=2, m1=2, p1=4, rank r1=8
    Inner: n2=2, m2=2, p2=2, rank r2=7
    Total rank = 8*7 = 56
    
    Index mapping:
    i = 2*i1 + i2 (i1 in 0..1, i2 in 0..1) -> i in 0..3
    j = 2*j1 + j2 (j1 in 0..1, j2 in 0..1) -> j in 0..3
    k = 4*k1 + k2 (k1 in 0..1, k2 in 0..3) -> k in 0..7
    """
    U222, V222, W222 = _strassen_2x2()
    U224, V224, W224 = _schoolbook_2x2x4()
    
    n1, m1, p1, r1 = 2, 2, 4, 8
    n2, m2, p2, r2 = 2, 2, 2, 7
    n, m, p = n1*n2, m1*m2, p1*p2  # 4, 4, 8
    rank = r1 * r2  # 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = n1 * i1 + i2
                            j = m1 * j1 + j2
                            u_val = U224[i1 * m1 + j1, c1] * U222[i2 * m2 + j2, c2]
                            if u_val != 0:
                                U[i * m + j, c] += u_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = m1 * j1 + j2
                            k = p1 * k1 + k2
                            v_val = V224[j1 * p1 + k1, c1] * V222[j2 * p2 + k2, c2]
                            if v_val != 0:
                                V[j * p + k, c] += v_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = p1 * k1 + k2
                            i = n1 * i1 + i2
                            w_val = W224[k1 * n1 + i1, c1] * W222[k2 * n2 + i2, c2]
                            if w_val != 0:
                                W[k * n + i, c] += w_val
    
    return U, V, W


def _verify_4x4x8(U, V, W, rank):
    """Verify the 4x4*4x8 decomposition using vectorized operations."""
    n, m, p = 4, 4, 8
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=1e-8))


def _als_reduce_rank(U0, V0, W0, target_rank, max_iters=30, tol=1e-8, deadline=None):
    """
    Try to reduce rank from rank(U0) to target_rank using ALS.
    Fix two factors, solve for the third via least squares (vectorized).
    """
    n, m, p = 4, 4, 8
    r0 = U0.shape[1]
    
    if target_rank >= r0:
        return U0, V0, W0, False
    
    U = U0[:, :target_rank].copy()
    V = V0[:, :target_rank].copy()
    W = W0[:, :target_rank].copy()
    r = target_rank
    
    # Precompute index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    N = n * m * p
    
    u_row = i_idx * m + j_idx
    v_row = j_idx * p + k_idx
    w_row = k_idx * n + i_idx
    
    def solve_W(U, V):
        A = np.zeros((N, p * n * r), dtype=float)
        for c in range(r):
            u3 = U[:, c].reshape(n, m)
            v3 = V[:, c].reshape(m, p)
            coeff = np.einsum('ij,jk->ijk', u3, v3).flatten()
            wfi = (k_idx * n + i_idx) * r + c
            A[np.arange(N), wfi] = coeff
        try:
            Wf, _, _, _ = np.linalg.lstsq(A, np.ones(N), rcond=None)
            return Wf.reshape(p * n, r)
        except:
            return W
    
    def solve_U(V, W):
        A = np.zeros((N, n * m * r), dtype=float)
        for c in range(r):
            v3 = V[:, c].reshape(m, p)
            w3 = W[:, c].reshape(p, n)
            coeff = np.einsum('jk,ki->ijk', v3, w3).flatten()
            ufi = (i_idx * m + j_idx) * r + c
            A[np.arange(N), ufi] = coeff
        try:
            Uf, _, _, _ = np.linalg.lstsq(A, np.ones(N), rcond=None)
            return Uf.reshape(n * m, r)
        except:
            return U
    
    def solve_V(U, W):
        A = np.zeros((N, m * p * r), dtype=float)
        for c in range(r):
            u3 = U[:, c].reshape(n, m)
            w3 = W[:, c].reshape(p, n)
            coeff = np.einsum('ij,ki->ijk', u3, w3).flatten()
            vfi = (j_idx * p + k_idx) * r + c
            A[np.arange(N), vfi] = coeff
        try:
            Vf, _, _, _ = np.linalg.lstsq(A, np.ones(N), rcond=None)
            return Vf.reshape(m * p, r)
        except:
            return V
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        W = solve_W(U, V)
        U = solve_U(V, W)
        V = solve_V(U, W)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = float(np.max(np.abs(T - 1.0)))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    success = best_err < 1e-6
    return best_U, best_V, best_W, success


def _peeling_approach():
    """
    Peeling approach: Fix U and V to sparse canonical patterns, then solve
    linear systems for W. Try to find a low-rank solution.
    """
    n, m, p = 4, 4, 8
    best_result = None
    
    for r in range(40, 57):
        U = np.zeros((n * m, r), dtype=float)
        for idx in range(n * m):
            U[idx, idx % r] = 1.0
        
        V = np.zeros((m * p, r), dtype=float)
        for idx in range(m * p):
            V[idx, idx % r] = 1.0
        
        A = np.zeros((n * m * p, p * n * r), dtype=float)
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if u_val != 0 and v_val != 0:
                            w_idx = (k * n + i) * r + c
                            A[eq_idx, w_idx] = u_val * v_val
                    eq_idx += 1
        
        try:
            w_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W_flat = w_flat.reshape(p * n, r)
            
            valid = _verify_4x4x8(U, V, W_flat, r)
            
            if valid:
                W_check = W_flat * 2
                if np.allclose(W_check, np.round(W_check), atol=1e-6):
                    W_flat = np.round(W_check) / 2.0
                    if best_result is None or r < best_result[0]:
                        best_result = (r, U.copy(), V.copy(), W_flat.copy())
        
        except Exception:
            pass
    
    return best_result


def construct():
    t_start = time.time()
    DEADLINE = t_start + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try Kronecker product decomposition (rank 56)
    U_kron = V_kron = W_kron = None
    try:
        if time.time() > DEADLINE:
            raise TimeoutError("Deadline exceeded before Kronecker construction")
        
        U_kron, V_kron, W_kron = _kronecker_4x4x8()
        rank = W_kron.shape[1]
        
        if _verify_4x4x8(U_kron, V_kron, W_kron, rank):
            # Try to reduce rank via ALS
            if time.time() < DEADLINE - 20:
                for target_r in range(48, 57):
                    if time.time() > DEADLINE - 5:
                        break
                    U_r, V_r, W_r, success = _als_reduce_rank(
                        U_kron, V_kron, W_kron, target_r,
                        max_iters=15, deadline=DEADLINE - 5
                    )
                    if success and _verify_4x4x8(U_r, V_r, W_r, target_r):
                        # Check for half-integer entries
                        ok = True
                        for mat in [U_r, V_r, W_r]:
                            doubled = mat * 2
                            if not np.allclose(doubled, np.round(doubled), atol=1e-6):
                                ok = False
                                break
                        if ok:
                            U_r = np.round(U_r * 2) / 2.0
                            V_r = np.round(V_r * 2) / 2.0
                            W_r = np.round(W_r * 2) / 2.0
                            if _verify_4x4x8(U_r, V_r, W_r, target_r):
                                return {
                                    "U": U_r.tolist(),
                                    "V": V_r.tolist(),
                                    "W": W_r.tolist()
                                }
            
            return {
                "U": U_kron.tolist(),
                "V": V_kron.tolist(),
                "W": W_kron.tolist()
            }
        
    except Exception:
        pass
    
    # Try peeling approach
    if time.time() < DEADLINE - 10:
        try:
            result = _peeling_approach()
            if result is not None:
                r, U, V, W = result
                if _verify_4x4x8(U, V, W, r):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        except Exception:
            pass
    
    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
