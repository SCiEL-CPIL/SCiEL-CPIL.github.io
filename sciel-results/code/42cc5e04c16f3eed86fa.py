"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T_reconstructed[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    T_rec = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_rec, T, atol=1e-10):
        return False
    return True

def _hoi_decompose(T, rank, max_iter=200, tol=1e-10):
    """
    Higher-Order Orthogonal Iteration to find rank-r decomposition of tensor T.
    T shape: (I, J, K). Returns U (I,r), V (J,r), W (K,r) such that
    T ~= sum_d U[:,d] outer V[:,d] outer W[:,d]
    """
    I, J, K = T.shape
    
    # Initialize with random or SVD-based initialization
    np.random.seed(42)
    U = np.random.randn(I, rank)
    V = np.random.randn(J, rank)
    W = np.random.randn(K, rank)
    
    # SVD-based initialization
    T_mat1 = T.reshape(I, J*K)
    _, _, Vt1 = np.linalg.svd(T_mat1, full_matrices=False)
    U[:, :min(rank, I)] = Vt1[:min(rank, I)].T
    
    T_mat2 = T.reshape(J, I*K)
    _, _, Vt2 = np.linalg.svd(T_mat2, full_matrices=False)
    V[:, :min(rank, J)] = Vt2[:min(rank, J)].T
    
    T_mat3 = T.reshape(K, I*J)
    _, _, Vt3 = np.linalg.svd(T_mat3, full_matrices=False)
    W[:, :min(rank, K)] = Vt3[:min(rank, K)].T
    
    prev_err = float('inf')
    
    for it in range(max_iter):
        # Update U: U[:,d] = SVD of (V^T W) contracted with T
        # T contracted with V and W: (V^T W)[b,c] * T[a,b,c] -> matrix (I, rank*rank)
        VW = np.einsum('bd,cd->bc', V, W)  # (J, K)
        # Contract T with VW over b,c: result is (I, J, K) * (J, K) -> (I,)
        # Actually: U_new[:,d] should be SVD of T contracted with V and W
        # Standard HOOI: U = SVD( T x_2 V x_3 W )[:, :rank]
        
        # T x_3 W: (I, J, K) x (K, r) -> (I, J, r)
        TW = np.einsum('abc,cd->abd', T, W)
        # (I, J, r) x_2 V: (I, J, r) x (J, r) -> (I, r)
        TVW = np.einsum('abd,bd->ad', TW, V)
        _, _, U_new_t = np.linalg.svd(TVW, full_matrices=False)
        U_new = U_new_t.T[:, :rank]
        
        # T x_2 V: (I, J, K) x (J, r) -> (I, r, K)
        TV = np.einsum('abc,bd->acd', T, V)
        # (I, r, K) x_3 W: (I, r, K) x (K, r) -> (I, r)
        TVW2 = np.einsum('acd,cd->ad', TV, W)
        # Wait, that's wrong. Let me redo.
        # TVW2 should be T x_2 V x_3 W contracted differently
        # Actually for updating U: U = SVD(T x_2 V x_3 W)
        # T x_2 V: (I,J,K) x (J,r) -> (I, r, K)  [contract over J]
        # Then (I,r,K) x_3 W: (I,r,K) x (K,r) -> (I, r) [contract over K]
        # But that gives (I,r) which is what we want for SVD.
        # Hmm, actually the standard HOOI is:
        # U = SVD( (V^T T W) )  where V^T T W is (r, I, r) -> reshape to (r*r, I)
        
        # Let me use the cleaner formulation:
        # U_new = SVD( T x_2 V x_3 W )[:, :rank]
        # where T x_2 V x_3 W means: first contract T with V on mode 2, then with W on mode 3
        # Result is a matrix of shape (I, rank)
        
        # T x_2 V: einsum('abc,bd->acd', T, V) -> (I, r, K)
        # Then x_3 W: einsum('acd,cd->ad', TV_result, W) -> (I, r)
        TV_step1 = np.einsum('abc,bd->acd', T, V)  # (I, r, K)
        TVW_step = np.einsum('acd,cd->ad', TV_step1, W)  # (I, r)
        _, _, U_new_t = np.linalg.svd(TVW_step, full_matrices=False)
        U_new = U_new_t.T[:, :rank]
        
        # Update V: V = SVD( T x_1 U x_3 W )[:, :rank]
        # T x_1 U: (I,J,K) x (I,r) -> (r, J, K)
        TU_step1 = np.einsum('abc,ad->bdc', T, U)  # (J, r, K)
        # x_3 W: (J,r,K) x (K,r) -> (J, r)
        TUW_step = np.einsum('bdc,cd->bd', TU_step1, W)  # (J, r)
        _, _, V_new_t = np.linalg.svd(TUW_step, full_matrices=False)
        V_new = V_new_t.T[:, :rank]
        
        # Update W: W = SVD( T x_1 U x_2 V )[:, :rank]
        # T x_1 U: (I,J,K) x (I,r) -> (r, J, K)
        TU_step2 = np.einsum('abc,ad->bdc', T, U)  # (J, r, K)
        # x_2 V: (J,r,K) x (J,r) -> (K, r)
        TUV_step = np.einsum('bdc,bd->cd', TU_step2, V)  # (K, r)
        _, _, W_new_t = np.linalg.svd(TUV_step, full_matrices=False)
        W_new = W_new_t.T[:, :rank]
        
        # Compute error
        T_rec = np.einsum('ad,bd,cd->abc', U_new, V_new, W_new)
        err = np.linalg.norm(T_rec - T) / np.linalg.norm(T)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
        U, V, W = U_new, V_new, W_new
    
    return U, V, W

def _round_to_half_int(arr, max_val=10.0):
    """Round array to nearest half-integer, clipping to reasonable range."""
    rounded = np.round(arr * 2.0) / 2.0
    # Clip to avoid huge values
    rounded = np.clip(rounded, -max_val, max_val)
    return rounded

def _try_rank(n, m, p, rank, T, max_hoi_iter=100):
    """Try to find a valid rank-r decomposition. Returns (U,V,W) or None."""
    try:
        U, V, W = _hoi_decompose(T, rank, max_iter=max_hoi_iter)
    except Exception:
        return None
    
    # Round to half-integers
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)
    
    # Verify
    T_rec = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
    if np.allclose(T_rec, T, atol=1e-8):
        # Convert to lists
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        # Clean up: convert 0.0 to 0, 0.5 stays, etc.
        def clean(val):
            if abs(val) < 1e-10:
                return 0
            if abs(val * 2 - round(val * 2)) < 1e-10:
                return round(val * 2) / 2
            return val
        U_list = [[clean(v) for v in row] for row in U_list]
        V_list = [[clean(v) for v in row] for row in V_list]
        W_list = [[clean(v) for v in row] for row in W_list]
        return U_list, V_list, W_list
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Upper bound: n*m*p = 48 (schoolbook)
    # Try from 24 upward
    
    best_result = None
    
    for rank in range(24, 49):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(n, m, p, rank, T, max_hoi_iter=80)
        if result is not None:
            U, V, W = result
            # Double-check with exact verification
            if _verify(U, V, W, n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                break
    
    # If HOOI didn't find anything, fall back to schoolbook
    if best_result is None:
        if time.time() > deadline - 10:
            # Emergency: return schoolbook immediately
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
        
        # Try a few more ranks with more iterations
        for rank in range(24, 49):
            if time.time() > deadline - 10:
                break
            result = _try_rank(n, m, p, rank, T, max_hoi_iter=150)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, rank):
                    best_result = {"U": U, "V": V, "W": W}
                    break
    
    if best_result is None:
        # Final fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
