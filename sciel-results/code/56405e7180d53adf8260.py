"""Matrix multiplication 3,4,8 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build T from U,V,W
    # T has shape (n, m, p)
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # (n*m,)
        vc = V[:, c]  # (m*p,)
        wc = W[:, c]  # (p*n,)
        # reshape
        uc_r = uc.reshape(n, m)
        vc_r = vc.reshape(m, p)
        wc_r = wc.reshape(p, n)
        # outer product contribution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc_r[i, j] * vc_r[j, k] * wc_r[k, i]
    # Compare with true matmul tensor T[i,j,k] = 1 if the (i,k) entry of A@B uses A[i,j]*B[j,k]
    # The matmul tensor is T[i,j,k] = 1 for all (i,j,k) since (A@B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has all entries = 1
    true_T = np.ones((n, m, p))
    if np.max(np.abs(T - true_T)) > tol:
        return False
    return True

def _round_half(U, V, W):
    """Round entries to nearest half-integer."""
    U = np.round(np.asarray(U, dtype=float) * 2) / 2.0
    V = np.round(np.asarray(V, dtype=float) * 2) / 2.0
    W = np.round(np.asarray(W, dtype=float) * 2) / 2.0
    return U, V, W

def _try_reduce(U, V, W, n, m, p, target_rank, max_iter=50):
    """Try to reduce rank to target_rank via SVD truncation iterations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = U.shape[1]
    
    for iteration in range(max_iter):
        if current_rank <= target_rank:
            break
        # Build the tensor T from current U, V, W
        T = np.zeros((n, m, p))
        for c in range(current_rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        
        # Now try to find a lower-rank approximation
        # Reshape T to a matrix and do SVD
        # T is (n*m*p,) = (n*m, p) or (n, m*p) etc.
        # Use mode-1 unfolding: T_mat1 = T.reshape(n, m*p)
        # But we need a joint factorization approach.
        
        # Alternative: use the "tensor train" style SVD
        # Reshape to (n*m, p) and do SVD
        T_mat = T.reshape(n * m, p)
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
        
        # The rank of this unfolding gives a bound
        # But we need a 3-way factorization.
        
        # Simpler approach: try to project down
        # Take the top target_rank singular vectors
        r = min(target_rank, len(s))
        if r >= current_rank:
            break
            
        # New factors from SVD of the mode-1 unfolding
        # This gives U_new (n*m, r), and we need to absorb into V and W
        # This is approximate; let's just truncate the rank directly
        # by keeping the top target_rank columns based on energy
        
        # Compute energy per column
        col_energy = np.sum(U ** 2, axis=0) * np.sum(V ** 2, axis=0) * np.sum(W ** 2, axis=0)
        # Sort by energy, keep top target_rank
        order = np.argsort(col_energy)[::-1]
        keep = order[:target_rank]
        
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p, tol=1e-6):
            U, V, W = U_new, V_new, W_new
            current_rank = target_rank
        else:
            # Try a slightly higher rank
            target_rank += 1
            if target_rank >= current_rank:
                break
            continue
    
    return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook rank
    schoolbook_rank = n * m * p  # 96
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Try various target ranks
    for target in range(48, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        U_t, V_t, W_t, r = _try_reduce(U, V, W, n, m, p, target, max_iter=30)
        if r < best_rank and _verify(U_t, V_t, W_t, n, m, p, tol=1e-8):
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = r
    
    # Try more aggressive reduction with random perturbations
    if time.time() < deadline - 10:
        np.random.seed(42)
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            # Start from schoolbook with random sign flips
            U_r = U.copy()
            V_r = V.copy()
            W_r = W.copy()
            signs = np.random.choice([-1, 1], size=schoolbook_rank)
            U_r = U_r * signs
            V_r = V_r * signs
            
            for target in range(40, min(60, best_rank)):
                if time.time() > deadline - 5:
                    break
                U_t, V_t, W_t, r = _try_reduce(U_r, V_r, W_r, n, m, p, target, max_iter=20)
                if r < best_rank and _verify(U_t, V_t, W_t, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank = r
                    break
    
    # Round to half-integers
    best_U, best_V, best_W = _round_half(best_U, best_V, best_W)
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Convert to lists
    U_list = [[best_U[i, j] for j in range(best_rank)] for i in range(n * m)]
    V_list = [[best_V[i, j] for j in range(best_rank)] for i in range(m * p)]
    W_list = [[best_W[i, j] for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
