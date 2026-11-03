"""Matrix multiplication 2x3x6 rank-minimal tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n=2, m=3, p=6):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = 2, 3, 6
    target = np.ones((n, m, p), dtype=float)
    
    rank = U.shape[1]
    recon = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    
    return np.allclose(recon, target, atol=1e-9)

def _snap_to_half_int(matrix):
    """Snap values to nearest half-integer if close."""
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            snapped = round(val * 2) / 2
            if abs(val - snapped) < 1e-6:
                new_row.append(snapped)
            else:
                new_row.append(val)
        result.append(new_row)
    return result

def _als_decompose(n, m, p, rank, max_iters=100, tol=1e-8, seed=42):
    """Use ALS to find a rank-rank decomposition of the all-ones tensor."""
    np.random.seed(seed)
    
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Precompute index mappings for efficiency
    idx_U = [(i, j) for i in range(n) for j in range(m)]
    idx_V = [(j, k) for j in range(m) for k in range(p)]
    idx_W = [(k, i) for k in range(p) for i in range(n)]
    
    for it in range(max_iters):
        # Update U: for each (i,j), minimize ||target[i,j,:] - V_j * W_i||
        # where V_j is V[j*p:(j+1)*p, :] and W_i is W[:, i]
        for i in range(n):
            for j in range(m):
                # Target for this (i,j): all ones of length p
                target_vec = np.ones(p)
                # Build system: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed i,j: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # This is linear in U[i*m+j,:]
                # Let a_c = V[j*p+k,c] * W[k*n+i,c] for each k
                # We need: U[i*m+j,:] @ A = target_vec where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve least squares
                sol, _, _, _ = np.linalg.lstsq(A, target_vec, rcond=None)
                U[i*m+j, :] = sol
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                target_vec = np.ones(n)
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                sol, _, _, _ = np.linalg.lstsq(A, target_vec, rcond=None)
                V[j*p+k, :] = sol
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                target_vec = np.ones(m)
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                sol, _, _, _ = np.linalg.lstsq(A, target_vec, rcond=None)
                W[k*n+i, :] = sol
        
        # Check convergence
        if it > 0 and it % 10 == 0:
            recon = np.zeros((n, m, p))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            err = np.abs(recon - 1.0).max()
            if err < tol:
                break
    
    # Compute final error
    recon = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    err = np.abs(recon - 1.0).max()
    
    return U, V, W, err

def _try_rank_search(n=2, m=3, p=6, max_rank=30, timeout=140):
    """Try to find a decomposition with rank < n*m*p using ALS."""
    start_time = time.time()
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook as fallback
    def schoolbook():
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, n*m*p
    
    # Try ranks from low to high
    for rank in range(1, min(n*m*p, max_rank + 1)):
        if time.time() - start_time > timeout - 10:
            break
        
        # Try multiple random seeds
        for trial in range(5):
            if time.time() - start_time > timeout - 5:
                break
            
            seed = 42 + trial * 100 + rank
            U, V, W, err = _als_decompose(n, m, p, rank, max_iters=50, tol=1e-8, seed=seed)
            
            if err < 1e-6:
                # Try to snap to half-integers
                U_snapped = _snap_to_half_int(U.tolist())
                V_snapped = _snap_to_half_int(V.tolist())
                W_snapped = _snap_to_half_int(W.tolist())
                
                if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped, rank
                
                # Also try without snapping
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U.tolist(), V.tolist(), W.tolist(), rank
    
    # Return schoolbook
    U, V, W, rk = schoolbook()
    return U, V, W, rk

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a lower-rank decomposition using ALS
    U, V, W, rk = _try_rank_search(n, m, p, max_rank=30, timeout=110)
    
    # Verify before returning
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
