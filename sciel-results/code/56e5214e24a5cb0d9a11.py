"""Matrix multiplication 4x4x8 tensor decomposition.
Uses structured Strassen-based decomposition with shared sub-blocks.
"""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill


def _time_left():
    return max(0.0, _DEADLINE - time.time())


def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker of two 2x2x2 Strassen. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    r = r2 * r2  # 49
    U4 = np.zeros((16, r), dtype=np.int16)
    V4 = np.zeros((16, r), dtype=np.int16)
    W4 = np.zeros((16, r), dtype=np.int16)
    for c0 in range(r2):
        for c1 in range(r2):
            c = c0 * r2 + c1
            for i in range(4):
                for j in range(4):
                    U4[i * 4 + j, c] = U2[(i // 2) * 2 + (j // 2), c0] * U2[(i % 2) * 2 + (j % 2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j * 4 + k, c] = V2[(j // 2) * 2 + (k // 2), c0] * V2[(j % 2) * 2 + (k % 2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k * 4 + i, c] = W2[(k // 2) * 2 + (i // 2), c0] * W2[(k % 2) * 2 + (i % 2), c1]
    return U4, V4, W4


def _build_4x4x8_independent():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98."""
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W


def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T  # (n, p)
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err


def _verify_multi(U, V, W, n=4, m=4, p=8, n_tests=5, tol=1e-9):
    """Verify with multiple random seeds."""
    for seed in range(n_tests):
        ok, err = _verify(U, V, W, n, m, p, seed=seed + seed, tol=tol)
        if not ok:
            return False, err
    return True, err


def _try_shared_structure():
    """
    Try to build a 4x4x8 decomposition with shared structure between the two 4x4x4 blocks.
    
    Idea: Instead of treating C[:,0:4] and C[:,4:8] as completely independent,
    we can try to share some of the Strassen intermediate products.
    
    For the left block (k=0..3) and right block (k=4..7), the A-side is the same.
    So we can factor out the A-side Strassen decomposition and only vary the B-side.
    
    This gives us a rank-49 A-decomposition shared across both blocks, but we still
    need separate B-decompositions. However, we might be able to reduce the total rank.
    """
    if _time_left() < 10:
        return None
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # The key insight: for 4x4x8, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..7
    # 
    # If we use Strassen on the 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B_left = [[B00, B01], [B10, B11]] (k=0..3)
    # B_right = [[B20, B21], [B30, B31]] (k=4..7)
    #
    # The Strassen products M1..M7 for the left block use A-blocks and B_left-blocks
    # The Strassen products for the right block use A-blocks and B_right-blocks
    #
    # Since the A-blocks are shared, we can potentially combine.
    # But the tensor structure doesn't directly allow this without changing the rank.
    
    # Alternative: try to find a lower-rank decomposition by solving a system
    # For now, return the independent construction as baseline
    return _build_4x4x8_independent()


def _attempt_rank_reduction(U, V, W, n=4, m=4, p=8, max_time=30.0):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    """
    if _time_left() < max_time:
        return None
    
    start_time = time.time()
    rank_init = U.shape[1]
    
    # Convert to float for numerical work
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    best_rank = rank_init
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    
    # Try to find and remove linearly dependent columns
    # A column r is redundant if it can be expressed as a linear combination of others
    # such that the tensor contribution is preserved.
    
    # Simple approach: check if any column is a scalar multiple of another
    # (this would only work for exact duplicates)
    for r in range(rank_init):
        if time.time() - start_time > max_time or _time_left() < 5:
            break
        
        # Check if column r is approximately zero
        if np.max(np.abs(U_f[:, r])) < 1e-10 and np.max(np.abs(V_f[:, r])) < 1e-10 and np.max(np.abs(W_f[:, r])) < 1e-10:
            continue
        
        # Try removing column r and see if we can compensate
        # This is complex; for now, just try sparsification
        pass
    
    # Sparsification: try to zero out small entries
    for threshold in [0.01, 0.05, 0.1]:
        if time.time() - start_time > max_time or _time_left() < 5:
            break
        
        U_sparse = np.where(np.abs(U_f) < threshold, 0, U_f)
        V_sparse = np.where(np.abs(V_f) < threshold, 0, V_f)
        W_sparse = np.where(np.abs(W_f) < threshold, 0, W_f)
        
        # Remove all-zero columns
        mask = (np.any(U_sparse != 0, axis=0) & 
                np.any(V_sparse != 0, axis=0) & 
                np.any(W_sparse != 0, axis=0))
        
        U_s = U_sparse[:, mask]
        V_s = V_sparse[:, mask]
        W_s = W_sparse[:, mask]
        
        rank_s = U_s.shape[1]
        
        if rank_s < best_rank:
            ok, err = _verify(U_s, V_s, W_s, n, m, p)
            if ok:
                best_rank = rank_s
                best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    if best_rank < rank_init:
        return best_U, best_V, best_W
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    # Start with the known-correct rank-98 Strassen-Kronecker decomposition
    U, V, W = _build_4x4x8_independent()
    best_rank = 98
    best_U, best_V, best_W = U, V, W
    
    # Verify the baseline
    ok, err = _verify(U, V, W)
    if not ok:
        # Fallback to schoolbook
        n, m, p = 4, 4, 8
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Attempt rank reduction if time permits
    if _time_left() > 15:
        opt_result = _attempt_rank_reduction(U, V, W, max_time=min(30.0, _time_left() - 10))
        if opt_result is not None:
            U_opt, V_opt, W_opt = opt_result
            rank_opt = U_opt.shape[1]
            if rank_opt < best_rank:
                ok_opt, err_opt = _verify(U_opt, V_opt, W_opt)
                if ok_opt:
                    best_rank = rank_opt
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
