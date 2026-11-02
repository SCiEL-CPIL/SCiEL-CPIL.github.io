import numpy as np
import time
from itertools import combinations

_start_time = time.time()
TIME_LIMIT = 145.0

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = T[i*m+j, j*p+k, k*n+i]
                max_err = max(max_err, abs(actual - expected))
    
    for idx_i in range(n*m):
        for idx_j in range(m*p):
            for idx_k in range(p*n):
                i = idx_i // m
                j = idx_i % m
                j2 = idx_j // p
                k = idx_j % p
                k2 = idx_k // n
                i2 = idx_k % n
                if j == j2 and i == i2 and k == k2:
                    continue
                max_err = max(max_err, abs(T[idx_i, idx_j, idx_k]))
    
    return max_err < 1e-10

def _build_base_decomposition():
    """Build the rank-44 Strassen-based decomposition for 3x4 by 4x4."""
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m2 = (c+d)e
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            W[(b_c*2+1)*n + 1][col] = -1
            col += 1
            
            # m3 = a(f-h)
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m4 = d(g-e)
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1
            col += 1
            
            # m5 = (a+b)h
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1
            col += 1
            
            # m6 = (c-a)(e+f)
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1
            col += 1
            
            # m7 = (b-d)(g+h)
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return U, V, W

def _try_random_cp_decomposition(n, m, p, target_rank, n_iters=200, n_restarts=5, seed=42):
    """Try to find a CP decomposition of rank target_rank for the matrix multiplication tensor.
    
    Uses alternating least squares with random restarts.
    """
    np.random.seed(seed)
    U_dim = n * m  # 12
    V_dim = m * p  # 16
    W_dim = p * n  # 12
    
    # Build the target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(n_restarts):
        if _time_exceeded():
            break
            
        # Initialize randomly
        U = np.random.randn(U_dim, target_rank) * 0.5
        V = np.random.randn(V_dim, target_rank) * 0.5
        W = np.random.randn(W_dim, target_rank) * 0.5
        
        for it in range(n_iters):
            if _time_exceeded():
                break
                
            # Update U: for each i, solve least squares
            for i in range(U_dim):
                # T[i,:,:] = sum_r U[i,r] * V[:,r] * W[:,r]
                # Fix U[i,:], solve for U[i,r]
                # T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
                # This is: T[i].reshape(-1) = U[i,:] @ (V*W).reshape(target_rank, -1).T
                VW = (V * W).reshape(target_rank, -1)  # (R, V_dim*W_dim)
                rhs = T[i].reshape(-1)
                # Solve: VW.T @ U[i,:] = rhs  ->  (VW.T @ VW) @ U[i,:] = VW.T @ rhs
                A_mat = VW.T @ VW  # (R, R)
                b_vec = VW.T @ rhs  # (R,)
                try:
                    U[i,:] = np.linalg.solve(A_mat + 1e-10*np.eye(target_rank), b_vec)
                except:
                    U[i,:] = np.linalg.lstsq(A_mat + 1e-10*np.eye(target_rank), b_vec, rcond=None)[0]
            
            # Update V
            for j in range(V_dim):
                UW = (U * W).reshape(target_rank, -1)  # (R, U_dim*W_dim)
                # T[:,j,:] = sum_r V[j,r] * U[:,r] * W[:,r]
                # T[:,j,:].reshape(-1) = V[j,:] @ (U*W).reshape(R, -1).T
                A_mat = UW.T @ UW
                b_vec = UW.T @ T[:,j,:].reshape(-1)
                try:
                    V[j,:] = np.linalg.solve(A_mat + 1e-10*np.eye(target_rank), b_vec)
                except:
                    V[j,:] = np.linalg.lstsq(A_mat + 1e-10*np.eye(target_rank), b_vec, rcond=None)[0]
            
            # Update W
            for k in range(W_dim):
                UV = (U * V).reshape(target_rank, -1)  # (R, U_dim*V_dim)
                # T[:,:,k] = sum_r W[k,r] * U[:,r] * V[:,r]
                A_mat = UV.T @ UV
                b_vec = UV.T @ T[:,:,k].reshape(-1)
                try:
                    W[k,:] = np.linalg.solve(A_mat + 1e-10*np.eye(target_rank), b_vec)
                except:
                    W[k,:] = np.linalg.lstsq(A_mat + 1e-10*np.eye(target_rank), b_vec, rcond=None)[0]
        
        # Check error
        T_recon = np.zeros((U_dim, V_dim, W_dim))
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-8:
            break
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = best_result
        # Round to integers/half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Verify rounded version
        U_list = U_round.tolist()
        V_list = V_round.tolist()
        W_list = W_round.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list, target_rank
    
    return None

def _search_rank_reduction(U, V, W, n, m, p):
    """Search for a lower-rank decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_keep = list(range(rank))
    
    # First pass: try removing single columns
    for col_to_remove in range(rank):
        if _time_exceeded():
            break
        keep = [c for c in range(rank) if c != col_to_remove]
        # Fast check
        T = np.zeros((n*m, m*p, p*n))
        for r in keep:
            T += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    max_err = max(max_err, abs(T[i*m+j, j*p+k, k*n+i] - 1))
        
        if max_err < 1e-10:
            # Check zeros
            for idx_i in range(n*m):
                for idx_j in range(m*p):
                    for idx_k in range(p*n):
                        i = idx_i // m
                        j = idx_i % m
                        j2 = idx_j // p
                        k = idx_j % p
                        k2 = idx_k // n
                        i2 = idx_k % n
                        if j == j2 and i == i2 and k == k2:
                            continue
                        max_err = max(max_err, abs(T[idx_i, idx_j, idx_k]))
            
            if max_err < 1e-10:
                best_rank = rank - 1
                best_keep = keep
                break
    
    if best_rank < rank:
        new_U = [[U_arr[i, r] for r in best_keep] for i in range(n*m)]
        new_V = [[V_arr[i, r] for r in best_keep] for i in range(m*p)]
        new_W = [[W_arr[i, r] for r in best_keep] for i in range(p*n)]
        for mat in [new_U, new_V, new_W]:
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    mat[i][j] = round(mat[i][j] * 2) / 2
        return new_U, new_V, new_W
    
    # Try random CP decomposition for lower ranks
    for target_rank in range(30, best_rank):
        if _time_exceeded():
            break
        result = _try_random_cp_decomposition(n, m, p, target_rank, n_iters=100, n_restarts=3, seed=42+target_rank)
        if result is not None:
            U_list, V_list, W_list, r = result
            return U_list, V_list, W_list
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 4
    
    # Build the rank-44 decomposition
    U, V, W = _build_base_decomposition()
    
    # Try to reduce rank
    if not _time_exceeded():
        U_reduced, V_reduced, W_reduced = _search_rank_reduction(U, V, W, n, m, p)
        if len(U_reduced[0]) < len(U[0]):
            U, V, W = U_reduced, V_reduced, W_reduced
    
    return {"U": U, "V": V, "W": W}
