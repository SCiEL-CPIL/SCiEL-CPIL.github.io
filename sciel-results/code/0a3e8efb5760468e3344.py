"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k, else 0.
    The convention is W[k*n + i, r].
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _decompose_tensor(T, rank, method='CP'):
    """Attempt to decompose T into rank factors.
    Returns (U, V, W) each of shape (dim, rank) as lists, or None on failure.
    """
    shape = T.shape
    # For a 3-way tensor of shape (I, J, K), a CP rank-r decomposition is:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # Here I=n*m, J=m*p, K=p*n
    I, J, K = shape
    # Matrix unfoldings:
    # T_I (I x J*K): reshape T to (I, J*K)
    # T_J (J x I*K): reshape to (J, I*K) with appropriate transposition
    # T_K (K x I*J): reshape to (K, I*J)
    
    T_I = T.reshape(I, J * K)
    # For J unfolding: T_J[j, i*K + k] = T[i, j, k]
    T_J = np.zeros((J, I * K), dtype=np.int64)
    T_K = np.zeros((K, I * J), dtype=np.int64)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T_J[j, i * K + k] = T[i, j, k]
                T_K[k, i * J + j] = T[i, j, k]
    
    # Rank-r factorization via SVD of T_I:
    # U * (V * W)^T where (V*W) has shape (J*K, r)
    # This is a rank-r approximation, but we need exact.
    
    # Let's try: use SVD of T_I, take top r components, then check if the full tensor is exactly recovered.
    # For exact CP, we need T_i = U * (V ⊙ W)^T where ⊙ is Khatri-Rao.
    # This is hard. Let's try a different approach.
    
    # Use the fact that the schoolbook is a CP decomposition. For lower rank, we can try random rank-r factors and check.
    # But exact recovery by chance is unlikely.
    
    # Instead, let's try to solve the system of equations.
    # We have I*J*K equations, and 3*I*rank unknowns.
    # This is a polynomial system, hard to solve exactly.
    
    # For now, return the schoolbook decomposition which is always valid.
    return None

def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    I, J, K = T.shape
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.array_equal(reconstructed, T)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition (rank = n*m*p = 48)
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    # For 2x4x6 matrix multiplication, we can use a block-based approach
    # Since n=2 is small, we can exploit the structure
    
    import numpy as np
    
    # Try ranks from a reasonable lower bound up to schoolbook-1
    # Lower bound: at least max(n, m, p) * 2 = 12, but realistically higher
    # For 2x4x6, known results suggest rank around 30-40
    
    start_rank = 30
    for rank in range(start_rank, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Use ALS (Alternating Least Squares) to try to find exact decomposition
        # Initialize with random values, then iterate
        
        best_solution = None
        n_restarts = 5  # Multiple random restarts
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U_curr = np.random.randn(n * m, rank) * 0.1
            V_curr = np.random.randn(m * p, rank) * 0.1
            W_curr = np.random.randn(p * n, rank) * 0.1
            
            # ALS iterations
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: For each i, T[i,:,:] = U[i,:] * (V ⊙ W).T
                # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                # Fix V and W, solve for U
                # T_i = U_i * (V ⊙ W).T
                # where (V ⊙ W)[(j,p_idx), c] = V[j,c] * W[p_idx,c]
                
                # Build Khatri-Rao product of V and W
                # Shape: (m*p, p*n) -> need to reshape properly
                # Actually, for fixed i: T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
                # Let M_i be matrix where M_i[j,k] = T[i,j,k]
                # Let Q be matrix where Q[(j,k), c] = V[j,c] * W[k,c]
                # Then M_i = U[i,:] @ Q.T
                
                # Build Q: shape (m*p, p*n) doesn't work directly
                # Q has entries Q[j*p_idx, c] where j in [0,m), p_idx in [0,p*n)
                # But we need Q[(j,k), c] = V[j,c] * W[k,c]
                # The index (j,k) corresponds to position j*(p*n) + k in a flattened array?
                # No, T is indexed as T[i*m+j, j*p+k, k*n+i]
                # For fixed i, we have T[i*m+j, j*p+k, k*n+i] for j in [0,m), k in [0,p)
                # But T is a 3D array of shape (n*m, m*p, p*n)
                # So T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
                
                # Let's reshape T for fixed i
                # For fixed i, we want M_i[j,k] = T[i*m+j, j*p+k, k*n+i]
                M_i = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        M_i[j, k] = T[i * m + j, j * p + k, k * n + i]
                
                # Q[j, k, c] = V[j, c] * W[k, c] but W is indexed by (k*n+i)
                # Wait, W has shape (p*n, rank), so W[k*n+i, c]
                # So for fixed i, Q[j, k, c] = V[j, c] * W[k*n+i, c]
                
                # We need to solve: M_i[j, k] = sum_c U[i, c] * V[j, c] * W[k*n+i, c]
                # This is: M_i = sum_c U[i, c] * (V[:, c] * W[i_idx[:, c]])
                # where i_idx maps k to k*n+i
                
                # Let's build the design matrix for U[i,:]
                # M_i.flatten() = U[i,:] @ D_i.T
                # where D_i is (m*p, rank) matrix with D_i[j*p+k, c] = V[j, c] * W[k*n+i, c]
                
                D_i = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            D_i[j * p + k, c] = V_curr[j, c] * W_curr[k * n + i, c]
                
                # Solve: M_i.flatten() = D_i.T @ U[i,:]
                # Use least squares
                try:
                    U_curr[i, :] = np.linalg.lstsq(D_i.T, M_i.flatten(), rcond=None)[0]
                except:
                    pass
                
                # Update V: For each j, similar approach
                # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j, c] * W[k*n+i, c]
                # For fixed j: T[i*m+j, j*p+k, k*n+i] for i in [0,n), k in [0,p)
                # = sum_c U[i*m+j, c] * V[j, c] * W[k*n+i, c]
                
                for j in range(m):
                    # Build design matrix for V[j,:]
                    # M_j[i, k] = T[i*m+j, j*p+k, k*n+i]
                    M_j = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            M_j[i, k] = T[i * m + j, j * p + k, k * n + i]
                    
                    # D_j[i*p+k, c] = U[i*m+j, c] * W[k*n+i, c]
                    D_j = np.zeros((n * p, rank))
                    for i in range(n):
                        for k in range(p):
                            for c in range(rank):
                                D_j[i * p + k, c] = U_curr[i * m + j, c] * W_curr[k * n + i, c]
                    
                    try:
                        V_curr[j, :] = np.linalg.lstsq(D_j.T, M_j.flatten(), rcond=None)[0]
                    except:
                        pass
                
                # Update W: For each (k,i) pair
                # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j, c] * W[k*n+i, c]
                # For fixed (k,i): T[i*m+j, j*p+k, k*n+i] for j in [0,m)
                # = sum_c U[i*m+j, c] * V[j, c] * W[k*n+i, c]
                
                for k in range(p):
                    for i in range(n):
                        # Build design matrix for W[k*n+i,:]
                        # M_w[j] = T[i*m+j, j*p+k, k*n+i]
                        M_w = np.zeros(m)
                        for j in range(m):
                            M_w[j] = T[i * m + j, j * p + k, k * n + i]
                        
                        # D_w[j, c] = U[i*m+j, c] * V[j, c]
                        D_w = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                D_w[j, c] = U_curr[i * m + j, c] * V_curr[j, c]
                        
                        try:
                            W_curr[k * n + i, :] = np.linalg.lstsq(D_w.T, M_w, rcond=None)[0]
                        except:
                            pass
            
            # Check if solution is exact
            reconstructed = np.einsum('ic,jc,kc->ijk', U_curr, V_curr, W_curr)
            if np.allclose(reconstructed, T, atol=1e-6):
                # Check if entries are close to integers or half-integers
                max_dev = max(np.max(np.abs(U_curr - np.round(U_curr))),
                             np.max(np.abs(V_curr - np.round(V_curr))),
                             np.max(np.abs(W_curr - np.round(W_curr))))
                if max_dev < 1e-4:
                    # Convert to integers
                    U_int = np.round(U_curr).astype(int)
                    V_int = np.round(V_curr).astype(int)
                    W_int = np.round(W_curr).astype(int)
                    
                    # Verify exactness with integers
                    recon_int = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                    if np.array_equal(recon_int, T):
                        best_solution = (U_int, V_int, W_int)
                        break
        
        if best_solution is not None:
            U_best, V_best, W_best = best_solution
            best_rank = rank
            best_U = U_best.tolist()
            best_V = V_best.tolist()
            best_W = W_best.tolist()
            break  # Found a solution, stop searching
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
