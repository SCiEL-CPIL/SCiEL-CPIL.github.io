"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]
import time
import random
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill

def _elapsed():
    return time.time() - _T0

def _remaining():
    return _DEADLINE - _elapsed()

def _verify_full(U, V, W, n, m, p, tol=1e-8):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # F[ia, ja, jb, kb, k, i] = sum_r U3[ia, ja, r] * V3[jb, kb, r] * W3[k, i, r]
    # Using einsum with distinct labels:
    # U: a b r  (a=ia, b=ja)
    # V: c d r  (c=jb, d=kb)
    # W: e f r  (e=k, f=i)
    # Output: a b c d e f
    F = np.einsum('abr,cdr,efr->abcdef', U3, V3, W3)
    
    # Expected[ia, ja, jb, kb, k, i] = 1 if ia==i AND ja==jb AND kb==k else 0
    expected = np.zeros((n, m, m, p, p, n))
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for k in range(p):
                        for i in range(n):
                            if ia == i and ja == jb and kb == k:
                                expected[ia, ja, jb, kb, k, i] = 1.0
    
    if np.max(np.abs(F - expected)) > tol:
        return False
    return True

def construct():
    if _remaining() < 5.0:
        # Not enough time, return the known rank-22 solution immediately
        return _rank22()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with the known rank-22 solution
    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22
    best_verified = _verify_full(best_U, best_V, best_W, n, m, p)
    if not best_verified:
        # Something wrong with our rank-22, use pure schoolbook as fallback
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p  # 24
    
    # Try to reduce rank using border rank heuristic:
    # 1. Start from rank-22 factors
    # 2. Apply iterative truncated SVD on unfolded matrices
    # 3. Project onto half-integer lattice
    # 4. Verify if we get a valid rank-21 or lower decomposition
    
    if _remaining() > 10.0:
        # Try border rank approach for rank 21
        success = _try_border_rank(21, best_U, best_V, best_W, n, m, p)
        if success:
            best_rank = 21
            best_verified = True
        
        # If still time, try rank 20
        if best_rank == 22 and _remaining() > 20.0:
            success = _try_border_rank(20, best_U, best_V, best_W, n, m, p)
            if success:
                best_rank = 20
                best_verified = True
    
    return _to_dict(best_U, best_V, best_W)

def _try_border_rank(target_rank, U, V, W, n, m, p, max_iters=50):
    """
    Try to reduce rank using border rank heuristic.
    Uses iterative truncated SVD on unfolded matrices.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    for iteration in range(max_iters):
        if _remaining() < 5.0:
            return False
        
        # Unfold along different modes and apply truncated SVD
        # Mode 1: U (n*m x r)
        # We want to reduce the rank, so we truncate the SVD
        
        # First, let's try to find a low-rank approximation
        # by projecting onto a lower-dimensional subspace
        
        # Strategy: randomly perturb and then do SVD-based compression
        U_pert = U + 0.01 * np.random.randn(*U.shape)
        V_pert = V + 0.01 * np.random.randn(*V.shape)
        W_pert = W + 0.01 * np.random.randn(*W.shape)
        
        # Truncate SVD to target_rank
        U_trunc, sU, _ = np.linalg.svd(U_pert, full_matrices=False)
        V_trunc, sV, _ = np.linalg.svd(V_pert, full_matrices=False)
        W_trunc, sW, _ = np.linalg.svd(W_pert, full_matrices=False)
        
        # Keep only target_rank components
        U_new = U_trunc[:, :target_rank] * sU[:target_rank]
        V_new = V_trunc[:, :target_rank] * sV[:target_rank]
        W_new = W_trunc[:, :target_rank] * sW[:target_rank]
        
        # Project onto half-integer lattice
        U_proj = np.round(U_new * 2) / 2
        V_proj = np.round(V_new * 2) / 2
        W_proj = np.round(W_new * 2) / 2
        
        # Verify
        if _verify_full(U_proj, V_proj, W_proj, n, m, p, tol=1e-6):
            # Found a valid decomposition!
            # Update the global best
            best_U, best_V, best_W = _rank22_list(n, m, p)
            best_U = U_proj.tolist()
            best_V = V_proj.tolist()
            best_W = W_proj.tolist()
            return True
        
        # If not valid, try a different perturbation
        # Use a smaller perturbation
        U_pert = U + 0.001 * np.random.randn(*U.shape)
        V_pert = V + 0.001 * np.random.randn(*V.shape)
        W_pert = W + 0.001 * np.random.randn(*W.shape)
        
        U_trunc, sU, _ = np.linalg.svd(U_pert, full_matrices=False)
        V_trunc, sV, _ = np.linalg.svd(V_pert, full_matrices=False)
        W_trunc, sW, _ = np.linalg.svd(W_pert, full_matrices=False)
        
        U_new = U_trunc[:, :target_rank] * sU[:target_rank]
        V_new = V_trunc[:, :target_rank] * sV[:target_rank]
        W_new = W_trunc[:, :target_rank] * sW[:target_rank]
        
        U_proj = np.round(U_new * 2) / 2
        V_proj = np.round(V_new * 2) / 2
        W_proj = np.round(W_new * 2) / 2
        
        if _verify_full(U_proj, V_proj, W_proj, n, m, p, tol=1e-6):
            best_U, best_V, best_W = _rank22_list(n, m, p)
            best_U = U_proj.tolist()
            best_V = V_proj.tolist()
            best_W = W_proj.tolist()
            return True
    
    return False

def _rank22():
    n, m, p = DIMENSIONS
    return _to_dict(*_rank22_list(n, m, p))

def _rank22_list(n, m, p):
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Block 1: A_L (2x2) * B_T1 (2x2) -> C1 (2x2)
    # A_L: A[0:2, 0:2], B_T1: B[0:2, 0:2], C1: C[0:2, 0:2]
    # Strassen 7 products
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=4, B11=5
    # C/W indices: C00=W[0], C01=W[2], C10=W[1], C11=W[3]

    col = 0
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13
    
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}
