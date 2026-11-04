import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U, V, W exactly decompose tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _als_decompose(T, rank, max_iter=200, tol=1e-8, seed=0):
    """Alternating least squares to find rank-r decomposition of tensor T."""
    n, m, p = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n, rank)
    V = rng.randn(m, rank)
    W = rng.randn(p, rank)
    
    prev_err = float('inf')
    for it in range(max_iter):
        # Update U: for each i, solve (V^T W^T) U[:,i] = T[i,:,:].reshape(-1)
        # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # For fixed i: T[i,:,:] = U[i,:] @ (V^T W^T) -- no, that's not right
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # Let M = V^T * W (element-wise product summed over c)... 
        # Actually: for fixed i, T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
        # So T[i,:,:] = sum_c U[i,c] * outer(V[:,c], W[:,c])
        # This is: vec(T[i,:,:]) = sum_c U[i,c] * vec(outer(V[:,c], W[:,c]))
        # = (V ⊙ W) @ U[i,:]  where ⊙ is Khatri-Rao product
        # So U[i,:] = ((V ⊙ W)^T (V ⊙ W))^{-1} (V ⊙ W)^T vec(T[i,:,:])
        
        # Khatri-Rao product of V and W: column c is V[:,c] * W[:,c]
        VW = V * W  # element-wise, shape (m*p? no, m and p may differ)
        # Actually V is (m, rank), W is (p, rank), so V*W doesn't work directly
        # We need the Khatri-Rao product: for each c, outer(V[:,c], W[:,c])
        # The matrix whose c-th column is vec(outer(V[:,c], W[:,c])) = (W[:,c] ⊗ V[:,c])
        # So VW_kron = np.kron(W, V) -- shape (p*m, rank) but we need (m*p, rank)
        # Actually vec(outer(V[:,c], W[:,c])) in column-major is V[:,c] ⊗ W[:,c]
        # In row-major (C-order) it's W[:,c] ⊗ V[:,c]
        # numpy uses row-major, so vec(outer(V[:,c], W[:,c])) = (W[:,c] ⊗ V[:,c])
        
        # Let's build the Khatri-Rao matrix properly
        # KR[c, j*p+k] = V[j,c] * W[k,c]
        KR_VW = np.zeros((m*p, rank))
        for c in range(rank):
            KR_VW[:, c] = np.outer(V[:, c], W[:, c]).flatten()  # row-major flatten
        
        # Now for each i: vec(T[i,:,:]) = KR_VW @ U[i,:]
        # U = (KR_VW^T KR_VW)^{-1} KR_VW^T T_reshaped
        T_flat = T.reshape(n, m*p)  # T[i, j*p+k]
        KR_T = KR_VW.T @ KR_VW  # (rank, rank)
        KR_T_inv = np.linalg.pinv(KR_T)
        U_new = KR_T_inv @ KR_VW.T @ T_flat.T  # (rank, n)
        U_new = U_new.T  # (n, rank)
        
        # Similarly for V
        # T[i,j,k] = sum_c V[j,c] * (U[i,c] * W[k,c])
        KR_UW = np.zeros((n*p, rank))
        for c in range(rank):
            KR_UW[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        T_flat2 = T.reshape(m, n*p)  # T[j, i*p+k]? No...
        # T[i,j,k], for fixed j: T[:,j,:] has shape (n,p)
        # We want T_reshaped[j, i*p+k] = T[i,j,k]
        T_reshaped_j = np.transpose(T, (1, 0, 2)).reshape(m, n*p)
        KR_UW_T = KR_UW.T @ KR_UW
        KR_UW_T_inv = np.linalg.pinv(KR_UW_T)
        V_new = KR_UW_T_inv @ KR_UW.T @ T_reshaped_j.T
        V_new = V_new.T  # (m, rank)
        
        # Similarly for W
        # T[i,j,k] = sum_c W[k,c] * (U[i,c] * V[j,c])
        KR_UV = np.zeros((n*m, rank))
        for c in range(rank):
            KR_UV[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        T_reshaped_k = np.transpose(T, (2, 0, 1)).reshape(p, n*m)
        KR_UV_T = KR_UV.T @ KR_UV
        KR_UV_T_inv = np.linalg.pinv(KR_UV_T)
        W_new = KR_UV_T_inv @ KR_UV.T @ T_reshaped_k.T
        W_new = W_new.T  # (p, rank)
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T_rec - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def construct():
    deadline = time.time() + 120 - 15  # Safety margin
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    T = _make_tensor(n, m, p)
    
    # Schoolbook decomposition
    U = np.zeros((n*m, rank_schoolbook))
    V = np.zeros((m*p, rank_schoolbook))
    W = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using ALS
    # Start from reasonable lower bounds
    # Known: rank(2×2×2) = 7, so for 4×5×5 we expect rank < 100
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Also, rank >= n*m*p / min(n*m, m*p, p*n) ... not a tight bound
    
    # Try ranks from 50 down to 30
    for target_rank in range(60, 24, -1):
        if time.time() > deadline:
            break
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            U_try, V_try, W_try, err = _als_decompose(T, target_rank, max_iter=100, seed=trial*100)
            if err < 1e-5:
                # Verify
                if _verify_decomposition(U_try, V_try, W_try, T, tol=1e-4):
                    best_result = {'U': U_try.tolist(), 'V': V_try.tolist(), 'W': W_try.tolist()}
                    best_rank = target_rank
                    found = True
                    break
        if found:
            # Try to go lower
            continue
    
    return best_result
