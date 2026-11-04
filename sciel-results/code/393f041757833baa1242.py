"""Rank-minimal decomposition for 2x3 * 3x3 matrix multiplication.
Uses Strassen-block rank-17 as baseline, searches for rank-16 via
laser method (structured entries) and high-precision ALS polish."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    """Full matmul tensor T[i,j,k]=1 iff (i//M)==(k%N) and (i%M)==(j//P)."""
    T = np.zeros((U_DIM, V_DIM, W_DIM), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T


def _validate(U, V, W, T, atol=1e-10):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    """Valid rank-17 decomposition using Strassen for 2x2x2 sub-block."""
    rank = 17
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    # Strassen columns 0..6 for the 2x2x2 sub-problem (rows 0,1 of A; cols 0,1 of B; output cols 0,1)
    # A is 2x3, B is 3x3. Take A[:,0:2] (2x2) and B[0:2,0:2] (2x2) -> C[0:2,0:2] (2x2)
    # Strassen: M1=(A11-A22)(B21+B22), M2=(A21+A22)B11, M3=A11(B11-B22),
    #           M4=(A11+A22)(B22-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
    # C11=M3+M4, C12=M5+M6, C21=M2+M7, C22=M1+M4
    # Mapping: A11=A[0,0]=idx0, A12=A[0,1]=idx1, A21=A[1,0]=idx3, A22=A[1,1]=idx4
    # B11=B[0,0]=idx0, B12=B[0,1]=idx1, B21=B[1,0]=idx3, B22=B[1,1]=idx4
    # C11=C[0,0]=idx0, C12=C[0,1]=idx1, C21=C[1,0]=idx3, C22=C[1,1]=idx4
    
    # M1 = (A11-A22)(B21+B22) -> contributes to C22
    U[0, 0] = 1; U[4, 0] = -1; V[3, 0] = 1; V[4, 0] = 1; W[4, 0] = 1
    # M2 = (A21+A22)B11 -> contributes to C21
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[3, 1] = 1
    # M3 = A11(B11-B22) -> contributes to C11
    U[0, 2] = 1; V[0, 2] = 1; V[4, 2] = -1; W[0, 2] = 1
    # M4 = (A11+A22)(B22-B11) -> contributes to C11 and C22
    U[0, 3] = 1; U[4, 3] = 1; V[4, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[4, 3] = 1
    # M5 = (A11+A12)B22 -> contributes to C12
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
    # M6 = (A21-A11)(B11+B12) -> contributes to C12
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[1, 5] = 1
    # M7 = (A12-A22)(B21+B22) -> contributes to C21
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[3, 6] = 1
    
    # Schoolbook for remaining terms involving column 2 of B (indices 5,6,7 in V)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    U[0, 7] = 1; V[2, 7] = 1; W[2, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[2, 8] = 1
    U[2, 9] = 1; V[8, 9] = 1; W[2, 9] = 1
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    U[3, 10] = 1; V[2, 10] = 1; W[5, 10] = 1
    U[4, 11] = 1; V[5, 11] = 1; W[5, 11] = 1
    U[5, 12] = 1; V[8, 12] = 1; W[5, 12] = 1
    # C[0,0] already handled by Strassen? Let me check...
    # Actually the Strassen handles C[0,0], C[0,1], C[1,0], C[1,1] using B[0:2,0:2]
    # But we also need terms from B[:,2] for C[:,2] which we did above.
    # Wait, I think I made an error. Let me reconsider.
    
    # Actually, let me just use the known correct rank-17 from the current program
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
    U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    return U, V, W


def _als_decompose(T, rank, U_init=None, V_init=None, W_init=None, max_iter=200, deadline=None, tol=1e-12):
    """Alternating least squares for tensor decomposition."""
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    
    if U_init is None:
        U = np.random.randn(U_DIM, rank) * 0.1
    else:
        U = np.array(U_init, dtype=float)
    if V_init is None:
        V = np.random.randn(V_DIM, rank) * 0.1
    else:
        V = np.array(V_init, dtype=float)
    if W_init is None:
        W = np.random.randn(W_DIM, rank) * 0.1
    else:
        W = np.array(W_init, dtype=float)
    
    prev_err = np.inf
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        
        # Solve for V
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        
        # Solve for W
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
        
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        
        if err < tol:
            return U, V, W, err
        if abs(prev_err - err) < 1e-15 * max(1.0, prev_err):
            break
        prev_err = err
    
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = np.linalg.norm(recon - T)
    return U, V, W, err


def _polish(U, V, W, T, max_iter=500, tol=1e-14, deadline=None):
    """High-precision ALS polish with tight tolerance."""
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = U.shape[1]
    prev_err = np.inf
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < tol:
            break
        if abs(prev_err - err) < 1e-16 * max(1.0, prev_err):
            break
        prev_err = err
    return U, V, W, err


def _laser_search_rank16(T, deadline):
    """
    Laser method: fix factor entries to small integers {-1, 0, 1} with
    structured sparsity patterns, then polish with high-precision ALS.
    
    Key insight: for matmul tensors, rank-minimal decompositions often
    have entries in {-1, 0, 1}. We enumerate structured patterns where
    each factor column has a specific sparsity pattern, then refine.
    """
    U17, V17, W17 = _rank17()
    best_err = np.inf
    best_result = None
    
    # Strategy A: Drop one column from rank-17, snap to integers, polish
    for drop_col in range(17):
        if time.time() > deadline:
            break
        U = np.delete(U17, drop_col, axis=1)
        V = np.delete(V17, drop_col, axis=1)
        W = np.delete(W17, drop_col, axis=1)
        # Snap to nearest integer (laser constraint)
        U_int = np.round(U).astype(float)
        V_int = np.round(V).astype(float)
        W_int = np.round(W).astype(float)
        # Polish from integer initialization
        U_p, V_p, W_p, err = _polish(U_int, V_int, W_int, T, max_iter=300, tol=1e-13, deadline=deadline)
        if err < best_err:
            best_err = err
            best_result = (U_p, V_p, W_p, err)
        if err < 1e-12:
            return best_result
    
    # Strategy B: Structured random initialization with integer entries
    # Each column of U has 2-3 nonzeros, V has 2-3 nonzeros, W has 2-3 nonzeros
    rng = np.random.RandomState(42)
    for trial in range(80):
        if time.time() > deadline:
            break
        U = np.zeros((U_DIM, 16))
        V = np.zeros((V_DIM, 16))
        W = np.zeros((W_DIM, 16))
        for c in range(16):
            # Random sparse pattern for U
            nz_u = rng.choice(U_DIM, size=rng.randint(2, 4), replace=False)
            for idx in nz_u:
                U[idx, c] = rng.choice([-1, 1])
            nz_v = rng.choice(V_DIM, size=rng.randint(2, 4), replace=False)
            for idx in nz_v:
                V[idx, c] = rng.choice([-1, 1])
            nz_w = rng.choice(W_DIM, size=rng.randint(2, 4), replace=False)
            for idx in nz_w:
                W[idx, c] = rng.choice([-1, 1])
        U_p, V_p, W_p, err = _polish(U, V, W, T, max_iter=200, tol=1e-13, deadline=deadline)
        if err < best_err:
            best_err = err
            best_result = (U_p, V_p, W_p, err)
        if err < 1e-12:
            return best_result
    
    # Strategy C: Start from rank-17, merge two columns into one, polish
    for i in range(17):
        if time.time() > deadline:
            break
        for j in range(i + 1, 17):
            if time.time() > deadline:
                break
            U = np.array(U17, dtype=float)
            V = np.array(V17, dtype=float)
            W = np.array(W17, dtype=float)
            # Merge column j into column i
            U[:, i] += U[:, j]
            V[:, i] += V[:, j]
            W[:, i] += W[:, j]
            # Remove column j
            U = np.delete(U, j, axis=1)
            V = np.delete(V, j, axis=1)
            W = np.delete(W, j, axis=1)
            # Snap to integers
            U_int = np.round(U).astype(float)
            V_int = np.round(V).astype(float)
            W_int = np.round(W).astype(float)
            U_p, V_p, W_p, err = _polish(U_int, V_int, W_int, T, max_iter=200, tol=1e-13, deadline=deadline)
            if err < best_err:
                best_err = err
                best_result = (U_p, V_p, W_p, err)
            if err < 1e-12:
                return best_result
    
    # Strategy D: General ALS from multiple initializations (fallback)
    for trial in range(30):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(30000 + trial)
        U = rng.randn(U_DIM, 16) * 0.5
        V = rng.randn(V_DIM, 16) * 0.5
        W = rng.randn(W_DIM, 16) * 0.5
        U_p, V_p, W_p, err = _polish(U, V, W, T, max_iter=200, tol=1e-13, deadline=deadline)
        if err < best_err:
            best_err = err
            best_result = (U_p, V_p, W_p, err)
        if err < 1e-12:
            return best_result
    
    if best_result is not None and best_result[3] < 1e-10:
        return best_result[:3]
    return None


def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    T = _build_tensor()
    
    # Baseline: rank-17
    U17, V17, W17 = _rank17()
    if not _validate(U17, V17, W17, T):
        # Fallback to schoolbook rank-18
        rank = N * M * P
        U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
        col = 0
        for ia in range(N):
            for ja in range(M):
                for kb in range(P):
                    U[ia * M + ja, col] = 1.0
                    V[ja * P + kb, col] = 1.0
                    W[kb * N + ia, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    base = (U17, V17, W17)
    
    # Try rank-16 with laser method + high-precision polish
    try:
        r16 = _laser_search_rank16(T, deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass
    
    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
