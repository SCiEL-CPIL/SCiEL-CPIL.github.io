"""Matrix multiplication 4x4 x 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 Kronecker-Strassen decomposition.
  2. Compress by identifying linearly dependent rank-1 terms via SVD
     of the factor matrices, aiming to reduce rank below 77.
  3. Use coordinate descent to refine and potentially reduce rank further.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},
    {0: 0, 1: 0, 2: 1, 3: 1},
    {0: 1, 1: 0, 2: 0, 3: 0},
    {0: 0, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 1, 2: 0, 3: 0},
    {0: -1, 1: 0, 2: 1, 3: 0},
    {0: 0, 1: 1, 2: 0, 3: -1},
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 0, 2: 0, 3: 0},
    {0: 0, 1: 1, 2: 0, 3: -1},
    {0: -1, 1: 0, 2: 1, 3: 0},
    {0: 0, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 1, 2: 0, 3: 0},
    {0: 0, 1: 0, 2: 1, 3: 1},
]
recon = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 0, -1), (0, 1, 1)],
    [(1, 1, 1)],
    [(0, 0, 1)],
]


def _build_rank77(n, m, p):
    """Build the rank-77 Kronecker-Strassen decomposition."""
    R = 77
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)

    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 x 2x2 (cols 0,1 of local 2x3)
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * p + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * n + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Schoolbook for 2x2 x 2x1 (col 2 of local 2x3)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * m + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * p + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * n + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    return U, V, W


def _compress_factors(U, V, W, T, target_rank, atol=1e-9):
    """Try to compress factors to target_rank by removing linearly dependent terms."""
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W, True
    
    # Try to identify which columns are linearly dependent
    # Use SVD on concatenated factor matrices
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    
    # Stack factor matrices as columns
    stacked = np.column_stack([U_flat, V_flat, W_flat])
    
    # SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, False
    
    # Find effective rank
    tol = max(stacked.shape) * np.max(s) * np.finfo(float).eps
    effective_rank = np.sum(s > tol)
    
    if effective_rank >= current_rank:
        return U, V, W, False
    
    # Try to project to lower-dimensional subspace
    # Keep only the top target_rank singular vectors
    keep = min(target_rank, effective_rank)
    if keep < target_rank:
        return U, V, W, False
    
    # Project factors onto the top-k singular subspace
    Vt_keep = Vt[:keep, :]
    
    # Reconstruct factors in the lower-dimensional space
    U_new = U.flatten().reshape(-1, 1) @ Vt_keep
    V_new = V.flatten().reshape(-1, 1) @ Vt_keep
    W_new = W.flatten().reshape(-1, 1) @ Vt_keep
    
    # This approach doesn't directly work for tensor decomposition
    # Instead, try a different strategy: iteratively remove columns
    # that contribute least to the error
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for _ in range(current_rank - target_rank):
        # Find the column with smallest norm
        norms = np.linalg.norm(U_cur, axis=0)
        min_idx = np.argmin(norms)
        
        # Try removing this column
        U_test = np.delete(U_cur, min_idx, axis=1)
        V_test = np.delete(V_cur, min_idx, axis=1)
        W_test = np.delete(W_cur, min_idx, axis=1)
        
        # Check if still valid
        recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
        if np.allclose(recon, T, atol=atol):
            U_cur, V_cur, W_cur = U_test, V_test, W_test
        else:
            # Try other columns
            for idx in range(len(norms)):
                if idx == min_idx:
                    continue
                U_test = np.delete(U_cur, idx, axis=1)
                V_test = np.delete(V_cur, idx, axis=1)
                W_test = np.delete(W_cur, idx, axis=1)
                
                recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
                if np.allclose(recon, T, atol=atol):
                    U_cur, V_cur, W_cur = U_test, V_test, W_test
                    break
            else:
                break
    
    if U_cur.shape[1] == target_rank:
        return U_cur, V_cur, W_cur, True
    return U, V, W, False


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)
    
    # Build rank-77 decomposition
    U77, V77, W77 = _build_rank77(n, m, p)
    
    if not check(U77, V77, W77, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
        V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
        W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()
    best_rank = 77
    
    # Try to compress to lower ranks
    for target_rank in range(76, 50, -1):
        if time.time() > deadline - 20:
            break
        
        U_compressed, V_compressed, W_compressed, success = _compress_factors(
            best_U, best_V, best_W, T, target_rank
        )
        
        if success and check(U_compressed, V_compressed, W_compressed, atol=1e-9):
            best_U, best_V, best_W = U_compressed, V_compressed, W_compressed
            best_rank = target_rank
    
    # Try coordinate descent to refine and potentially reduce rank further
    if time.time() < deadline - 30 and best_rank > 50:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        err = err_of(U, V, W)
        
        for it in range(50):
            if time.time() > deadline - 15:
                break
            if err < 1e-12:
                break
            
            improved = False
            for c in range(best_rank):
                if time.time() > deadline - 15:
                    break
                
                Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                
                # Small random perturbation
                U[:, c] += np.random.randn(n * m) * 0.001
                V[:, c] += np.random.randn(m * p) * 0.001
                W[:, c] += np.random.randn(p * n) * 0.001
                
                new_err = err_of(U, V, W)
                if new_err < err:
                    err = new_err
                    improved = True
                else:
                    U[:, c], V[:, c], W[:, c] = Uo, Vo, Wo
            
            if not improved:
                break
        
        if err < 1e-10:
            best_U, best_V, best_W = U, V, W
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        err = err_of(U, V, W)
        
        for it in range(100):
            if time.time() > deadline - 3:
                break
            if err < 1e-14:
                break
            
            improved = False
            for c in range(best_rank):
                Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                
                U[:, c] += np.random.randn(n * m) * 0.0001
                V[:, c] += np.random.randn(m * p) * 0.0001
                W[:, c] += np.random.randn(p * n) * 0.0001
                
                new_err = err_of(U, V, W)
                if new_err < err:
                    err = new_err
                    improved = True
                else:
                    U[:, c], V[:, c], W[:, c] = Uo, Vo, Wo
            
            if not improved:
                break
        
        if err < 1e-12:
            best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    print(f"Rank: {U.shape[1]}")
    
    n, m, p = 4, 4, 6
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T - recon))
    print(f"Max error: {max_err}")
    print(f"Valid: {np.allclose(recon, T, atol=1e-9)}")
